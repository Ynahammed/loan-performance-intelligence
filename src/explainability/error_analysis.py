"""
Error analysis and model uncertainty.

FALSE POSITIVES AND FALSE NEGATIVES
-----------------------------------
Aggregate metrics say how often the model is wrong. They do not say WHO
it is wrong about, and that is the question a reviewer and a model-risk
function both actually ask. Two things are reported:

  PROFILE   how the false positives differ from the true negatives, and
            the false negatives from the true positives, feature by
            feature, in standardised units so columns on different scales
            are comparable.

  CONCENTRATION  error rates by segment (credit band, state, servicer,
            vintage). A model that is 8% wrong overall but 30% wrong on
            one servicer has a fixable problem; a model that is uniformly
            8% wrong has a data problem. Only one of those is actionable.

UNCERTAINTY
-----------
Not a bootstrap. Bootstrapping rows measures sampling noise, which is
almost never what breaks a loan model in deployment -- what breaks it is
that the world moved. So uncertainty here is the SPREAD ACROSS
WALK-FORWARD FOLDS: refit the champion recipe on each expanding time
window, score the same evaluation rows with each, and report the
dispersion. It answers "how much does this loan's score depend on which
period the model happened to be trained on", which is the risk that
actually materialised in phase 3 when prepayment signal collapsed across
the rate-environment shift.

The ensemble uses a lighter model configuration than the champion, since
it exists to estimate dispersion rather than to set headline numbers.
That is stated wherever its output appears.

PHASE: 8
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, TIME_COLUMN
from src.explainability.labels import label
from src.models.train import build_preprocessor, recall_at_precision
from src.models.validation import rolling_time_series_splits

logger = logging.getLogger(__name__)

SEGMENT_COLUMNS = ("credit_score_band", "state", "servicer_name", "vintage",
                   "loan_purpose", "current_status")
MIN_SEGMENT_ROWS = 150


@dataclass
class ErrorAnalysis:
    threshold: float
    confusion: pd.DataFrame
    fp_profile: pd.DataFrame
    fn_profile: pd.DataFrame
    segment_errors: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        c = self.confusion.set_index("outcome")["n"]
        tp, fp, fn, tn = (int(c.get(k, 0)) for k in
                          ("true positive", "false positive",
                           "false negative", "true negative"))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return "\n".join([
            "threshold      : {:.4f}".format(self.threshold),
            "true positives : {:,}".format(tp),
            "false positives: {:,}".format(fp),
            "false negatives: {:,}".format(fn),
            "true negatives : {:,}".format(tn),
            "precision      : {:.4f}".format(precision),
            "recall         : {:.4f}".format(recall),
        ])


def choose_threshold(y_true, y_score, target_precision: float = 0.5) -> tuple:
    """Operating point at the requested precision, if reachable.

    Defaults to 0.5 probability only as a fallback, and says so -- a 0.5
    cut on a 0.9%-prevalence target is an arbitrary choice dressed up as
    a default.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ok = precision[:-1] >= target_precision
    if ok.any():
        best = np.argmax(recall[:-1] * ok)
        return float(thresholds[best]), (
            "threshold set at the highest-recall point where precision "
            "reaches {:.0%}".format(target_precision)
        )
    return 0.5, (
        "precision of {:.0%} is unreachable at any threshold; falling back "
        "to a 0.5 probability cut, which is arbitrary for a target at this "
        "prevalence".format(target_precision)
    )


def _standardised_gap(a: pd.DataFrame, b: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Mean difference between two groups in pooled standard deviations."""
    rows = []
    for col in cols:
        x, y = a[col].dropna(), b[col].dropna()
        if len(x) < 5 or len(y) < 5:
            continue
        pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        if not np.isfinite(pooled) or pooled == 0:
            continue
        rows.append({
            "feature": col,
            "label": label(col),
            "error_group_mean": round(float(x.mean()), 4),
            "reference_mean": round(float(y.mean()), 4),
            "std_gap": round(float((x.mean() - y.mean()) / pooled), 3),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (out.reindex(out["std_gap"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


def analyse_errors(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    target_precision: float = 0.5,
    top_n: int = 10,
) -> ErrorAnalysis:
    """Confusion breakdown, error profiles, and where errors concentrate."""
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_score = pd.Series(np.asarray(y_score, dtype=float)).reset_index(drop=True)
    data = df.reset_index(drop=True)

    threshold, reason = choose_threshold(y_true, y_score, target_precision)
    pred = (y_score >= threshold).astype(int)

    tp = (pred == 1) & (y_true == 1)
    fp = (pred == 1) & (y_true == 0)
    fn = (pred == 0) & (y_true == 1)
    tn = (pred == 0) & (y_true == 0)

    confusion = pd.DataFrame([
        {"outcome": "true positive", "n": int(tp.sum())},
        {"outcome": "false positive", "n": int(fp.sum())},
        {"outcome": "false negative", "n": int(fn.sum())},
        {"outcome": "true negative", "n": int(tn.sum())},
    ])

    numeric = [c for c in data.columns
               if pd.api.types.is_numeric_dtype(data[c])
               and data[c].nunique(dropna=True) > 2]

    fp_profile = _standardised_gap(data[fp], data[tn], numeric).head(top_n)
    fn_profile = _standardised_gap(data[fn], data[tp], numeric).head(top_n)

    segment_errors = {}
    for col in SEGMENT_COLUMNS:
        if col not in data.columns:
            continue
        frame = pd.DataFrame({
            "segment": data[col].astype(str),
            "y": y_true, "pred": pred,
        })
        agg = frame.groupby("segment").apply(
            lambda g: pd.Series({
                "n": len(g),
                "positives": int(g.y.sum()),
                "fp_rate": float(((g.pred == 1) & (g.y == 0)).sum()
                                 / max((g.y == 0).sum(), 1)),
                "fn_rate": float(((g.pred == 0) & (g.y == 1)).sum()
                                 / max((g.y == 1).sum(), 1)),
            }),
            include_groups=False,
        ).reset_index()
        agg = agg[agg.n >= MIN_SEGMENT_ROWS]
        if len(agg):
            segment_errors[col] = agg.round(4).sort_values(
                "fn_rate", ascending=False
            ).reset_index(drop=True)

    notes = [reason]
    if fp.sum() == 0:
        notes.append("no false positives at this threshold; FP profile is empty")
    if fn.sum() == 0:
        notes.append("no false negatives at this threshold; FN profile is empty")

    return ErrorAnalysis(
        threshold=threshold,
        confusion=confusion,
        fp_profile=fp_profile,
        fn_profile=fn_profile,
        segment_errors=segment_errors,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyResult:
    per_row: pd.DataFrame
    fold_summary: pd.DataFrame
    n_folds: int
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        if self.per_row.empty:
            return "no folds produced usable predictions"
        return "\n".join([
            "folds                    : {}".format(self.n_folds),
            "mean prediction spread   : {:.5f}".format(
                self.per_row["spread"].mean()),
            "median spread            : {:.5f}".format(
                self.per_row["spread"].median()),
            "rows where spread exceeds",
            "  the mean prediction    : {:,} ({:.2f}%)".format(
                int((self.per_row.spread > self.per_row.mean_prediction).sum()),
                100.0 * (self.per_row.spread > self.per_row.mean_prediction).mean()),
        ])


def temporal_ensemble_uncertainty(
    df: pd.DataFrame,
    target: str,
    columns: list,
    eval_df: pd.DataFrame,
    n_splits: int = 4,
    purge_months: int = 0,
    max_iter: int = 150,
    random_state: int = RANDOM_SEED,
) -> UncertaintyResult:
    """Spread of predictions across models trained on different periods.

    Each walk-forward fold produces one model; all of them score the same
    evaluation rows. A loan whose predictions agree across folds is one
    the model understands stably; a loan whose predictions swing is one
    where the answer depends on when you asked.
    """
    notes = [
        "Uncertainty is the dispersion across models fitted on different "
        "expanding time windows, not a bootstrap. Sampling noise is not "
        "what breaks a loan model in deployment; the world moving is.",
        "The ensemble uses a lighter configuration ({} iterations) than the "
        "champion, because it estimates spread rather than setting headline "
        "metrics.".format(max_iter),
    ]

    try:
        folds = rolling_time_series_splits(
            df, n_splits=n_splits, purge_months=purge_months, min_train_months=12
        )
    except ValueError as exc:
        return UncertaintyResult(pd.DataFrame(), pd.DataFrame(), 0,
                                 notes + ["fold construction failed: {}".format(exc)])

    predictions = []
    fold_rows = []
    for i, fold in enumerate(folds):
        y = fold.train[target].dropna().astype(int)
        if y.nunique() < 2 or len(y) < 500:
            continue
        model = Pipeline([
            ("pre", build_preprocessor(fold.train, columns, scale=False)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=max_iter, learning_rate=0.08,
                random_state=random_state)),
        ])
        model.fit(fold.train.loc[y.index, columns], y)
        p = model.predict_proba(eval_df[columns])[:, 1]
        predictions.append(p)
        fold_rows.append({
            "fold": i + 1,
            "train_rows": len(y),
            "train_end": str(fold.train_end.date()),
            "mean_prediction": round(float(p.mean()), 5),
        })

    if len(predictions) < 2:
        return UncertaintyResult(pd.DataFrame(), pd.DataFrame(fold_rows),
                                 len(predictions),
                                 notes + ["fewer than two usable folds; "
                                          "no spread can be computed"])

    stacked = np.vstack(predictions)
    per_row = pd.DataFrame({
        "mean_prediction": stacked.mean(axis=0),
        "spread": stacked.std(axis=0),
        "min_prediction": stacked.min(axis=0),
        "max_prediction": stacked.max(axis=0),
    }, index=eval_df.index)
    per_row["relative_spread"] = (
        per_row["spread"] / per_row["mean_prediction"].clip(lower=1e-6)
    )

    return UncertaintyResult(
        per_row=per_row,
        fold_summary=pd.DataFrame(fold_rows),
        n_folds=len(predictions),
        notes=notes,
    )


def confidence_band(
    spread: pd.Series,
    mean_prediction: pd.Series,
    method: str = "quantile",
    absolute_bins=(0.25, 0.75),
) -> pd.Series:
    """Turn dispersion into a three-level label for the reviewer UI.

    Relative spread rather than absolute: a swing of 0.02 is noise around
    a prediction of 0.40 and enormous around a prediction of 0.01.

    Bands are cut on QUANTILES of relative spread by default, not fixed
    constants. Measured on this data, fixed cuts of 0.25/0.75 put 91% of
    rows into "low confidence" -- technically defensible, since a
    0.9%-prevalence target really does have relative spread above 0.75
    almost everywhere, and completely useless to a reviewer who needs the
    label to discriminate. Terciles guarantee the bands separate the
    portfolio, and the report checks that they track real accuracy rather
    than assuming it.

    `method="absolute"` keeps the fixed thresholds for when an externally
    agreed cutoff matters more than usable bucket sizes.
    """
    relative = spread / mean_prediction.clip(lower=1e-6)
    labels = ["high confidence", "moderate confidence", "low confidence"]

    if method == "absolute":
        return pd.cut(
            relative, bins=[-np.inf, absolute_bins[0], absolute_bins[1], np.inf],
            labels=labels,
        ).astype(object)

    try:
        return pd.qcut(relative, 3, labels=labels, duplicates="drop").astype(object)
    except ValueError:
        # Too few distinct spread values to cut into three; fall back
        # rather than raise, and let the caller see one band.
        logger.warning("relative spread has too few distinct values to band")
        return pd.Series("moderate confidence", index=spread.index)
