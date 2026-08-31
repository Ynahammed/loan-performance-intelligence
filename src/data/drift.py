"""
Train vs. test distribution drift.

Two complementary views, because they fail in opposite directions:

  PSI, per feature. Univariate, interpretable, and the industry-standard
  number a credit reviewer already knows how to read (>0.25 = material
  shift). Blind to joint shifts: two features can each look stable while
  their relationship inverts.

  ADVERSARIAL VALIDATION. Train a classifier to tell a train row from a
  test row. If it cannot (AUC ~ 0.5), the two samples are exchangeable and
  no amount of per-feature staring will find drift. If it can, its feature
  importances name exactly what gives the game away -- including joint
  shifts PSI cannot see. It is also a leakage detector: an AUC near 1.0
  usually means an ID-like or timestamp column is in the feature set.

On a panel split chronologically, SOME drift is expected -- the test
period is genuinely later. The question this module answers is not
"is there drift" but "is it larger than the passage of time explains".

PHASE: 2
STATUS: implemented.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src.config import RANDOM_SEED, TIME_INDEX_COLUMNS

logger = logging.getLogger(__name__)

PSI_BANDS = [
    (0.10, "stable"),
    (0.25, "moderate shift"),
    (float("inf"), "material shift"),
]


def psi_band(value: float) -> str:
    for threshold, label in PSI_BANDS:
        if value < threshold:
            return label
    return "material shift"


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index between two samples of one feature.

    Numeric features are bucketed on the EXPECTED sample's quantiles, so
    the bins describe the reference population rather than being redrawn
    for each comparison. Categorical features compare category shares
    directly. Empty buckets are floored rather than dropped -- dropping
    them silently understates drift, which is the wrong direction to be
    wrong in.
    """
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    if pd.api.types.is_numeric_dtype(expected) and expected.nunique() > buckets:
        edges = np.unique(
            np.quantile(expected, np.linspace(0, 1, buckets + 1))
        )
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e_share = np.histogram(expected, bins=edges)[0] / len(expected)
        a_share = np.histogram(actual, bins=edges)[0] / len(actual)
    else:
        levels = sorted(set(expected.unique()) | set(actual.unique()))
        e_counts = expected.value_counts()
        a_counts = actual.value_counts()
        e_share = np.array([e_counts.get(v, 0) for v in levels], dtype=float) / len(expected)
        a_share = np.array([a_counts.get(v, 0) for v in levels], dtype=float) / len(actual)

    floor = 1e-6
    e_share = np.clip(e_share, floor, None)
    a_share = np.clip(a_share, floor, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def drift_report(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list = None
) -> pd.DataFrame:
    """Per-feature PSI between two samples, sorted worst first."""
    columns = columns or [c for c in train_df.columns if c in test_df.columns]
    rows = []
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(train_df[col]):
            continue  # a chronological split shifts dates by construction
        value = compute_psi(train_df[col], test_df[col])
        rows.append(
            {
                "column": col,
                "psi": round(value, 4) if pd.notna(value) else None,
                "band": psi_band(value) if pd.notna(value) else "not comparable",
                "train_missing_pct": round(100 * train_df[col].isna().mean(), 2),
                "test_missing_pct": round(100 * test_df[col].isna().mean(), 2),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list = None,
    n_splits: int = 3,
    max_rows: int = 40_000,
    exclude_time_like: bool = True,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Can a model tell train rows from test rows?

    Returns the cross-validated AUC plus the features that gave it away.
    Interpretation:
        ~0.50  samples are exchangeable; no detectable drift
        ~0.60  mild drift, usually just the passage of time
        >0.80  substantial drift, or a leaking identifier in the features
        ~1.00  almost certainly an ID/timestamp column, not real drift
    """
    columns = columns or [c for c in train_df.columns if c in test_df.columns]
    # Datetime columns always go: encoded as ordinals they separate a
    # chronological split perfectly and tell you only that time passed.
    columns = [c for c in columns
               if not pd.api.types.is_datetime64_any_dtype(train_df[c])]
    if exclude_time_like:
        columns = [c for c in columns if c not in TIME_INDEX_COLUMNS]
    if not columns:
        raise ValueError("no comparable columns left after exclusions")

    a = train_df[columns].copy()
    b = test_df[columns].copy()
    rng = np.random.default_rng(random_state)
    if len(a) > max_rows:
        a = a.iloc[rng.choice(len(a), max_rows, replace=False)]
    if len(b) > max_rows:
        b = b.iloc[rng.choice(len(b), max_rows, replace=False)]

    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)

    model = HistGradientBoostingClassifier(
        max_iter=150, random_state=random_state, categorical_features=None
    )
    proba = cross_val_predict(model, X, y, cv=n_splits, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))

    model.fit(X, y)
    importance = _permutation_importance(model, X, y, random_state)

    if auc < 0.55:
        verdict = "samples are exchangeable; no material drift detected"
    elif auc < 0.70:
        verdict = "mild drift, consistent with a chronological split"
    elif auc < 0.90:
        verdict = "substantial drift; review the top features below"
    else:
        verdict = ("near-perfect separation; check for an identifier or "
                   "timestamp-like column in the feature set before "
                   "concluding this is real drift")

    return {"auc": round(auc, 4), "verdict": verdict, "top_features": importance}


def _permutation_importance(model, X, y, random_state, n_top=10) -> pd.DataFrame:
    """Which columns the discriminator actually relies on.

    Permutation rather than a built-in importance so the number means
    "how much worse does it get without this", which is the question.
    """
    rng = np.random.default_rng(random_state)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for col in X.columns:
        saved = X[col].copy()
        X[col] = rng.permutation(saved.to_numpy())
        shuffled = roc_auc_score(y, model.predict_proba(X)[:, 1])
        X[col] = saved
        rows.append({"column": col, "auc_drop": round(base - shuffled, 4)})
    return (
        pd.DataFrame(rows)
        .sort_values("auc_drop", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )
