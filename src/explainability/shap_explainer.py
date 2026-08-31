"""
SHAP explanations for the supervised champions.

THE PIPELINE PROBLEM
--------------------
Every champion is a sklearn Pipeline: a ColumnTransformer feeding a
classifier. SHAP explains the classifier, so it sees the TRANSFORMED
matrix -- one-hot columns like `cat__state_CA`, not `state`. Explaining
at that level produces a global importance chart with forty near-zero
state indicators and no row for "state", which is both unreadable and
misleading about how much the column matters.

So attributions are computed on the transformed matrix and then summed
back to their source columns. Summing (not averaging) is the right
aggregation: the one-hot columns for a single categorical are mutually
exclusive per row, so their contributions add up to that column's total
contribution for that row.

EXPLAINER CHOICE
----------------
Routed by estimator type rather than by hope. TreeExplainer for the
gradient-boosted champions, LinearExplainer for the logistic one (which
won `next_12m_default`), and a model-agnostic fallback with an explicit
warning if a future champion is neither -- because a silent fallback to
a permutation explainer on 47,000 rows would look like a hang.

SAMPLING
--------
Global SHAP is computed on a sample, not the full panel. This is stated
in the report rather than hidden: with a stable global ranking, a few
thousand rows give the same ordering as fifty thousand at a fraction of
the cost, and the sample size travels with the output so the reader can
judge it.

PHASE: 8
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.explainability.labels import label

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE = 3000


@dataclass
class ShapResult:
    """Attributions at both the transformed and source-column level."""

    global_importance: pd.DataFrame        # per source column
    transformed_importance: pd.DataFrame   # per encoded column
    values: np.ndarray
    transformed_names: list
    source_of: list                        # source column per encoded column
    sample_index: pd.Index
    explainer_type: str
    base_value: float = 0.0
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join([
            "explainer      : {}".format(self.explainer_type),
            "sampled rows   : {:,}".format(len(self.sample_index)),
            "encoded columns: {}".format(len(self.transformed_names)),
            "source columns : {}".format(len(self.global_importance)),
        ])


def source_column_map(preprocessor, transformed_names: list) -> list:
    """Map each encoded column back to the raw column it came from.

    Matching is by LONGEST prefix against the transformer's declared input
    columns, so `credit_score_band_740-779` resolves to
    `credit_score_band` rather than to a shorter accidental prefix.
    """
    inputs = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        inputs.extend(list(cols))

    out = []
    for encoded in transformed_names:
        stripped = encoded.split("__", 1)[-1]
        candidates = [c for c in inputs if stripped == c or stripped.startswith(c + "_")]
        out.append(max(candidates, key=len) if candidates else stripped)
    return out


def _build_explainer(clf, background: np.ndarray):
    """Route to the right SHAP explainer for this estimator."""
    import shap

    name = type(clf).__name__
    if "HistGradientBoosting" in name or "Forest" in name or "XGB" in name:
        return shap.TreeExplainer(clf), "TreeExplainer"
    if "Logistic" in name or "Linear" in name or "Ridge" in name:
        return shap.LinearExplainer(clf, background), "LinearExplainer"
    logger.warning(
        "No specialised SHAP explainer for %s; falling back to the "
        "model-agnostic explainer, which is much slower.", name
    )
    return shap.Explainer(clf, background), "Explainer (model-agnostic)"


def _positive_class_values(values: np.ndarray) -> np.ndarray:
    """Normalise SHAP output to a 2-D (rows, features) array for class 1.

    Different explainers return different shapes for binary problems --
    (n, f), (n, f, 2), or a list of two (n, f) arrays -- and picking the
    wrong one silently explains the negative class.
    """
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def explain_model(
    model,
    X: pd.DataFrame,
    columns: list,
    sample: int = DEFAULT_SAMPLE,
    random_state: int = RANDOM_SEED,
) -> ShapResult:
    """Global SHAP attributions for one fitted pipeline."""
    notes = []
    rng = np.random.default_rng(random_state)

    if len(X) > sample:
        idx = X.index[rng.choice(len(X), sample, replace=False)]
        notes.append(
            "Global attributions computed on a random sample of {:,} of "
            "{:,} rows.".format(sample, len(X))
        )
    else:
        idx = X.index

    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    X_trans = pre.transform(X.loc[idx, columns])
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = ["f{}".format(i) for i in range(X_trans.shape[1])]
        notes.append("preprocessor exposed no feature names; using positional labels")

    background = X_trans[: min(200, len(X_trans))]
    explainer, explainer_type = _build_explainer(clf, background)
    raw = explainer.shap_values(X_trans) if hasattr(explainer, "shap_values") else explainer(X_trans).values
    values = _positive_class_values(raw)

    base = getattr(explainer, "expected_value", 0.0)
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])

    sources = source_column_map(pre, names)

    transformed = pd.DataFrame({
        "encoded_column": names,
        "source_column": sources,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Sum within a source column per row, THEN take the mean absolute
    # value. Summing the per-column means instead would double-count a
    # categorical whose one-hot pieces push in opposite directions.
    per_source = {}
    for src in dict.fromkeys(sources):
        cols_idx = [i for i, s in enumerate(sources) if s == src]
        per_source[src] = np.abs(values[:, cols_idx].sum(axis=1)).mean()

    global_importance = (
        pd.DataFrame({
            "source_column": list(per_source),
            "label": [label(c) for c in per_source],
            "mean_abs_shap": list(per_source.values()),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    total = global_importance["mean_abs_shap"].sum()
    global_importance["share_pct"] = (
        100 * global_importance["mean_abs_shap"] / max(total, 1e-12)
    ).round(2)
    global_importance["mean_abs_shap"] = global_importance["mean_abs_shap"].round(6)

    return ShapResult(
        global_importance=global_importance,
        transformed_importance=transformed,
        values=values,
        transformed_names=names,
        source_of=sources,
        sample_index=idx,
        explainer_type=explainer_type,
        base_value=float(base),
        notes=notes,
    )


def explain_one(
    model,
    row: pd.Series,
    columns: list,
    result: ShapResult,
    top_n: int = 5,
) -> pd.DataFrame:
    """Local attribution for a single record, aggregated to source columns.

    Returns signed contributions: positive pushes the probability up,
    negative pulls it down. Reviewers need the direction, not just the
    magnitude, and a bar chart of absolute values hides it.
    """
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    X_trans = pre.transform(row[columns].to_frame().T)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    explainer, _ = _build_explainer(clf, X_trans)
    values = _positive_class_values(
        explainer.shap_values(X_trans) if hasattr(explainer, "shap_values")
        else explainer(X_trans).values
    )[0]

    rows = []
    for src in dict.fromkeys(result.source_of):
        idx = [i for i, s in enumerate(result.source_of) if s == src]
        contribution = float(values[idx].sum())
        rows.append({
            "source_column": src,
            "label": label(src),
            "value": row.get(src),
            "contribution": round(contribution, 6),
            "direction": "increases risk" if contribution > 0 else "reduces risk",
        })

    out = pd.DataFrame(rows)
    out["abs"] = out["contribution"].abs()
    return (out.sort_values("abs", ascending=False).drop(columns="abs")
            .head(top_n).reset_index(drop=True))


def phrase_local_explanation(local: pd.DataFrame, probability: float) -> str:
    """Plain-language reviewer note from a local attribution table.

    Deliberately hedged: SHAP reports what moved THIS MODEL'S output, not
    what causes default. The phrasing says "the model weighted", never
    "because".
    """
    if local.empty:
        return "No individual factor stood out for this loan."
    up = local[local.contribution > 0].head(3)
    down = local[local.contribution < 0].head(2)
    parts = ["The model put this loan at {:.1%}.".format(probability)]
    if len(up):
        parts.append(
            "Weighing toward higher risk: " + ", ".join(up["label"]) + "."
        )
    if len(down):
        parts.append(
            "Weighing the other way: " + ", ".join(down["label"]) + "."
        )
    parts.append(
        "These are the factors the model leaned on, not established causes."
    )
    return " ".join(parts)
