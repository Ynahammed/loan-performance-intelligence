"""
Unsupervised anomaly detection: Isolation Forest on the engineered
numeric feature matrix.

TARGET-BLIND ON PURPOSE
-----------------------
The detector never sees exception_required, exception_type, or any
performance target. "Unusual" and "bad" are different questions, and
conflating them is the standard way this component goes wrong: a
detector trained toward the exception label stops being an independent
signal and becomes a worse copy of the rule engine. Keeping it blind is
what lets us claim, honestly, that when it agrees with the rules that
agreement is evidence.

WHY DEVIATION-BASED REASONS RATHER THAN SHAP
--------------------------------------------
SHAP can be coaxed onto an Isolation Forest via TreeExplainer, but the
resulting attributions explain a path-length score that nobody --
reviewer or judge -- has intuition for. Robust deviation from the
population is directly checkable: "loan amount sits in the 99.6th
percentile" can be verified by sorting a column. Explanations a reviewer
can audit beat explanations that are merely more sophisticated.

Median/MAD rather than mean/std, because the outliers we are hunting are
in the sample being summarised, and they drag a mean and inflate a
standard deviation enough to hide themselves.

LANGUAGE
--------
Everything this module emits says "unusual pattern requiring review".
Never "fraud", never "bad loan". The detector observes that a record is
statistically unlike its peers; it does not know why, and the difference
matters to the person who has to act on it.

PHASE: 7
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import LOAN_ID_COLUMN, RANDOM_SEED, TIME_COLUMN
from src.explainability.labels import label

logger = logging.getLogger(__name__)

# Columns the detector must never see: targets, identifiers, and the rule
# outputs (which would make it a laundered rule engine rather than an
# independent signal).
DETECTOR_EXCLUDE = {
    "exception_required", "exception_type", "next_state",
    "next_3m_delinquency_flag", "next_6m_delinquency_flag",
    "next_12m_default_flag", "next_12m_prepayment_flag",
    "default_flag", "prepayment_flag", "loss_severity_band",
    "rule_violation_count", "data_quality_score",
    "month_index", LOAN_ID_COLUMN, TIME_COLUMN,
}

DEVIATION_THRESHOLD = 3.0  # robust z beyond which a feature is "unusual"


@dataclass
class AnomalyResult:
    scores: pd.Series             # 0-1, higher = more unusual
    raw_scores: pd.Series         # IsolationForest.decision_function
    flagged: pd.Series            # bool, by contamination threshold
    feature_columns: list
    reference: pd.DataFrame       # median/MAD per feature, from training
    model: object = None
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join([
            "records scored     : {:,}".format(len(self.scores)),
            "features used      : {}".format(len(self.feature_columns)),
            "flagged as unusual : {:,} ({:.2f}%)".format(
                int(self.flagged.sum()),
                100.0 * self.flagged.mean() if len(self.flagged) else 0.0),
            "score range        : {:.4f} to {:.4f}".format(
                float(self.scores.min()), float(self.scores.max())),
        ])


def detector_features(df: pd.DataFrame) -> list:
    """Numeric, non-excluded, non-constant columns."""
    cols = []
    for c in df.columns:
        if c in DETECTOR_EXCLUDE:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


def _reference_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Median and MAD per feature, plus the quantile grid for percentiles."""
    median = X.median()
    mad = (X - median).abs().median()
    # A MAD of zero means the feature is constant across the middle of the
    # distribution; fall back to a scaled IQR so the deviation is defined
    # rather than infinite.
    iqr = (X.quantile(0.75) - X.quantile(0.25)) / 1.349
    scale = mad.where(mad > 0, iqr).replace(0, np.nan)
    return pd.DataFrame({"median": median, "mad": mad, "scale": scale})


def fit_isolation_forest(
    X: pd.DataFrame,
    contamination="auto",
    n_estimators: int = 300,
    random_state: int = RANDOM_SEED,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def detect_anomalies(
    df: pd.DataFrame,
    fit_on: pd.DataFrame = None,
    contamination="auto",
    random_state: int = RANDOM_SEED,
) -> AnomalyResult:
    """Score every record for how unusual it is.

    `fit_on` lets the population be defined by the training window while
    later records are scored against it -- the deployment shape. Defaults
    to fitting on `df` itself for a one-shot portfolio review.
    """
    notes = []
    cols = detector_features(df)
    if not cols:
        raise ValueError("no usable numeric features for the anomaly detector")

    fit_df = df if fit_on is None else fit_on
    fit_X = fit_df[cols].astype(float)
    # Impute with the training median, not the scoring median: a record
    # should be judged against the population, not against its own batch.
    medians = fit_X.median()
    fit_X = fit_X.fillna(medians)
    score_X = df[cols].astype(float).fillna(medians)

    model = fit_isolation_forest(fit_X, contamination, random_state=random_state)
    raw = pd.Series(model.decision_function(score_X), index=df.index)

    # decision_function is higher = more normal. Invert and min-max so the
    # published score reads the intuitive way, and record that the scale is
    # relative to this population rather than absolute.
    lo, hi = float(raw.min()), float(raw.max())
    if hi > lo:
        scores = (hi - raw) / (hi - lo)
    else:
        scores = pd.Series(0.0, index=df.index)
        notes.append("all records scored identically; anomaly score is uninformative")

    flagged = pd.Series(model.predict(score_X) == -1, index=df.index)

    return AnomalyResult(
        scores=scores.rename("anomaly_score"),
        raw_scores=raw.rename("anomaly_raw"),
        flagged=flagged.rename("anomaly_flagged"),
        feature_columns=cols,
        reference=_reference_stats(fit_X),
        model=model,
        notes=notes,
    )


def explain_anomaly(
    row: pd.Series,
    reference: pd.DataFrame,
    population: pd.DataFrame = None,
    top_n: int = 3,
) -> list:
    """Why this record looks unusual, in checkable terms.

    Returns up to `top_n` dicts, each naming a feature, its value, how far
    it sits from the population median in robust deviations, and its
    percentile. A reviewer can verify any of it by sorting one column.
    """
    rows = []
    for feature in reference.index:
        if feature not in row.index:
            continue
        value = row[feature]
        if pd.isna(value):
            continue
        scale = reference.loc[feature, "scale"]
        median = reference.loc[feature, "median"]
        if not np.isfinite(scale) or scale == 0:
            continue
        deviation = abs(float(value) - float(median)) / float(scale)
        if deviation < DEVIATION_THRESHOLD:
            continue
        entry = {
            "feature": feature,
            "label": label(feature),
            "value": float(value),
            "population_median": float(median),
            "robust_deviations": round(float(deviation), 2),
            "direction": "above" if value > median else "below",
        }
        if population is not None and feature in population.columns:
            pct = float((population[feature] < value).mean())
            entry["percentile"] = round(100 * pct, 1)
        rows.append(entry)

    rows.sort(key=lambda r: r["robust_deviations"], reverse=True)
    return rows[:top_n]


def phrase_reason(entry: dict) -> str:
    """One reviewer-readable sentence for a single driver."""
    if "percentile" in entry:
        return (
            "{} is {} the typical range ({:.4g} against a median of {:.4g}, "
            "{}th percentile)".format(
                entry["label"], entry["direction"], entry["value"],
                entry["population_median"], entry["percentile"],
            )
        )
    return (
        "{} is {} the typical range ({:.4g} against a median of {:.4g}, "
        "{} robust deviations out)".format(
            entry["label"], entry["direction"], entry["value"],
            entry["population_median"], entry["robust_deviations"],
        )
    )


def anomaly_reasons_frame(
    df: pd.DataFrame,
    result: AnomalyResult,
    index,
    top_n: int = 3,
) -> pd.DataFrame:
    """Drivers for a selected set of records, one row per record."""
    population = df[result.feature_columns]
    rows = []
    for idx in index:
        drivers = explain_anomaly(
            df.loc[idx], result.reference, population, top_n=top_n
        )
        rows.append({
            "index": idx,
            "anomaly_score": round(float(result.scores.loc[idx]), 4),
            "n_drivers": len(drivers),
            "drivers": "; ".join(
                "{} {} median by {:.1f} robust deviations".format(
                    d["label"], d["direction"], d["robust_deviations"])
                for d in drivers
            ) or "no single feature is individually extreme; the record is "
                 "unusual in combination",
        })
    return pd.DataFrame(rows)
