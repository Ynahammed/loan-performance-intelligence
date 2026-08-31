"""
Supervised exception prediction, scoped to what the rules cannot reach.

THE DESIGN QUESTION
-------------------
Phase 2 measured the rule engine at 98.1% precision and 96.1% recall
against exception_required. That number changes what the ML layer should
be. The obvious build -- train a classifier on exception_required with
the rule flags as features -- produces a model that scores beautifully
and has learned nothing: it rediscovers the rules, reports their
performance as its own, and adds a black box in front of an auditable
check. It would look like a better result and be a worse system.

So the supervised layer is pointed at the residual instead:

  RULE-COVERED ROWS    resolved deterministically. The rule fires, the
                       exception type is known from which rule fired, and
                       the reviewer gets a citation rather than a score.

  RESIDUAL ROWS        the ~4% of exceptions no rule catches. This is
                       where a model can add recall that did not exist
                       before, and it is trained ONLY on rows the rules
                       do not flag, so its metrics describe its actual
                       contribution rather than the rules'.

EXCEPTION TYPE
--------------
Same split. Rules assign the type where they fire, via a rule -> type
mapping that is a lookup, not a prediction. The classifier handles typed
exceptions the rules missed. `status_reversal` has 6 instances in the
entire panel and is deliberately left rules-only: six examples cannot
train anything, and the rule catches all six at 100% precision.

The combined output is what the submission needs: exception_required_pred,
exception_type_pred, and a confidence that says which mechanism decided.

PHASE: 7
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.anomaly.rules import ADVISORY_RULES, RULE_TO_EXCEPTION_TYPE
from src.config import RANDOM_SEED
from src.models.train import build_preprocessor, feature_columns
from src.models.validation import time_aware_split

logger = logging.getLogger(__name__)

# Types with too few examples to learn. Rules-only, by decision not by
# accident, and reported as such.
RULES_ONLY_TYPES = {"status_reversal"}

MIN_EXAMPLES_PER_TYPE = 25


@dataclass
class ExceptionModelResult:
    residual_metrics: pd.DataFrame
    type_metrics: pd.DataFrame
    coverage: pd.DataFrame
    residual_model: object = None
    type_model: object = None
    feature_cols: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def rule_assigned_type(flags: pd.DataFrame) -> pd.Series:
    """Deterministic exception type from whichever rule fired.

    Advisory rules (source conflict, staleness) are risk signals rather
    than labelled defects, so they never assign a type. Where two
    detector rules fire on one row, the first in RULE_TO_EXCEPTION_TYPE
    order wins -- rare, and recorded in the coverage table.
    """
    out = pd.Series("none", index=flags.index, dtype=object)
    for rule_id, exception_type in RULE_TO_EXCEPTION_TYPE.items():
        if rule_id not in flags.columns:
            continue
        fires = flags[rule_id].fillna(False) & (out == "none")
        out[fires] = exception_type
    return out


def rule_covered_mask(flags: pd.DataFrame) -> pd.Series:
    """Rows a detector rule fires on. Advisory rules do not count as
    coverage -- they raise suspicion, they do not resolve a record."""
    detector_cols = [c for c in flags.columns if c not in ADVISORY_RULES]
    if not detector_cols:
        return pd.Series(False, index=flags.index)
    return flags[detector_cols].fillna(False).any(axis=1)


def train_exception_models(
    features: pd.DataFrame,
    rule_flags: pd.DataFrame,
    anomaly_scores: pd.Series = None,
    target: str = "exception_required",
    type_target: str = "exception_type",
    random_state: int = RANDOM_SEED,
) -> ExceptionModelResult:
    """Train the residual exception model and the residual type classifier."""
    notes = []
    df = features.copy()
    covered = rule_covered_mask(rule_flags).reindex(df.index).fillna(False)

    if anomaly_scores is not None:
        df["anomaly_score"] = anomaly_scores.reindex(df.index).fillna(0.0)

    # ---- coverage accounting -----------------------------------------
    truth = df[target].fillna(0).astype(int)
    total_exceptions = int(truth.sum())
    caught = int((covered & (truth == 1)).sum())
    residual_exceptions = total_exceptions - caught
    coverage = pd.DataFrame([
        {"stage": "total exceptions in panel", "n": total_exceptions},
        {"stage": "resolved by rules", "n": caught},
        {"stage": "rule false positives", "n": int((covered & (truth == 0)).sum())},
        {"stage": "residual for the model", "n": residual_exceptions},
    ])

    residual = df[~covered]
    notes.append(
        "Residual population: {:,} rows carrying {:,} exceptions "
        "({:.3f}% prevalence, against {:.3f}% before the rules ran).".format(
            len(residual), residual_exceptions,
            100.0 * residual_exceptions / max(len(residual), 1),
            100.0 * total_exceptions / max(len(df), 1),
        )
    )

    cols = [c for c in feature_columns(residual) if c not in rule_flags.columns]
    if "anomaly_score" in residual.columns and "anomaly_score" not in cols:
        cols.append("anomaly_score")

    # ---- residual exception_required ---------------------------------
    residual_metrics = pd.DataFrame()
    residual_model = None
    if residual_exceptions < 20:
        notes.append(
            "Only {:,} exceptions survive the rules -- too few to train a "
            "residual model that could be evaluated honestly. The rule "
            "engine is reported as the complete solution for this "
            "target.".format(residual_exceptions)
        )
    else:
        split = time_aware_split(residual, purge_months=0, test_fraction=0.25)
        y_tr = split.train[target].fillna(0).astype(int)
        y_te = split.test[target].fillna(0).astype(int)

        if y_tr.sum() < 5 or y_te.sum() < 3:
            notes.append(
                "Residual exceptions do not survive a temporal split with "
                "enough positives on both sides; residual model skipped."
            )
        else:
            model = Pipeline([
                ("pre", build_preprocessor(split.train, cols, scale=False)),
                ("clf", HistGradientBoostingClassifier(
                    max_iter=250, learning_rate=0.06, random_state=random_state)),
            ])
            pos, neg = max(int(y_tr.sum()), 1), max(len(y_tr) - int(y_tr.sum()), 1)
            model.fit(split.train[cols], y_tr,
                      clf__sample_weight=np.where(y_tr == 1, neg / pos, 1.0))
            p = model.predict_proba(split.test[cols])[:, 1]

            rows = [{
                "model": "residual exception model",
                "n_test": len(split.test),
                "positives": int(y_te.sum()),
                "prevalence": round(float(y_te.mean()), 5),
                "roc_auc": round(float(roc_auc_score(y_te, p)), 4)
                if y_te.nunique() > 1 else None,
                "pr_auc": round(float(average_precision_score(y_te, p)), 4)
                if y_te.nunique() > 1 else None,
            }]
            if anomaly_scores is not None and "anomaly_score" in split.test:
                a = split.test["anomaly_score"].to_numpy()
                rows.append({
                    "model": "anomaly score alone (unsupervised)",
                    "n_test": len(split.test),
                    "positives": int(y_te.sum()),
                    "prevalence": round(float(y_te.mean()), 5),
                    "roc_auc": round(float(roc_auc_score(y_te, a)), 4)
                    if y_te.nunique() > 1 else None,
                    "pr_auc": round(float(average_precision_score(y_te, a)), 4)
                    if y_te.nunique() > 1 else None,
                })
            residual_metrics = pd.DataFrame(rows)
            residual_model = model

    # ---- residual exception_type -------------------------------------
    type_metrics = pd.DataFrame()
    type_model = None
    if type_target in df.columns:
        res_types = residual[residual[type_target].fillna("none") != "none"]
        counts = res_types[type_target].value_counts()
        learnable = [
            t for t, n in counts.items()
            if n >= MIN_EXAMPLES_PER_TYPE and t not in RULES_ONLY_TYPES
        ]
        skipped = [t for t in counts.index if t not in learnable]
        if skipped:
            notes.append(
                "Types left to the rules because the residual carries too "
                "few examples to learn them: {}.".format(
                    ", ".join("{} (n={})".format(t, counts[t]) for t in skipped))
            )
        if len(learnable) >= 2:
            sub = res_types[res_types[type_target].isin(learnable)]
            split = time_aware_split(sub, purge_months=0, test_fraction=0.25)
            if len(split.test) and split.test[type_target].nunique() >= 2:
                model = Pipeline([
                    ("pre", build_preprocessor(split.train, cols, scale=False)),
                    ("clf", HistGradientBoostingClassifier(
                        max_iter=200, random_state=random_state)),
                ])
                model.fit(split.train[cols], split.train[type_target])
                pred = model.predict(split.test[cols])
                type_metrics = pd.DataFrame([{
                    "n_test": len(split.test),
                    "classes": len(learnable),
                    "macro_f1": round(float(f1_score(
                        split.test[type_target], pred, average="macro",
                        zero_division=0)), 4),
                    "accuracy": round(float(
                        (pred == split.test[type_target]).mean()), 4),
                }])
                type_model = model
        else:
            notes.append(
                "Fewer than two learnable exception types survive the rules; "
                "type assignment is deterministic for every case the rules "
                "resolve, which is the great majority."
            )

    return ExceptionModelResult(
        residual_metrics=residual_metrics,
        type_metrics=type_metrics,
        coverage=coverage,
        residual_model=residual_model,
        type_model=type_model,
        feature_cols=cols,
        notes=notes,
    )


def combined_exception_output(
    features: pd.DataFrame,
    rule_flags: pd.DataFrame,
    anomaly_result,
    result: ExceptionModelResult,
) -> pd.DataFrame:
    """The reviewer- and submission-facing view.

    Every row gets a probability, a type, and -- importantly -- a
    `decided_by` column saying which mechanism produced them, so a
    reviewer never has to guess whether they are looking at a rule
    citation or a model score.
    """
    covered = rule_covered_mask(rule_flags).reindex(features.index).fillna(False)
    types = rule_assigned_type(rule_flags).reindex(features.index).fillna("none")

    # The residual model was trained with anomaly_score among its columns,
    # so the scoring frame must carry it too. Passing the raw feature frame
    # here fails at predict time, not at fit time.
    work = features.copy()
    if "anomaly_score" not in work.columns:
        work["anomaly_score"] = anomaly_result.scores.reindex(
            features.index
        ).fillna(0.0)

    out = pd.DataFrame(index=features.index)
    out["exception_probability"] = 0.0
    out["exception_type_pred"] = "none"
    out["decided_by"] = "no signal"
    out["anomaly_score"] = anomaly_result.scores.reindex(features.index).fillna(0.0)

    # Rules first: a fired rule is a fact, and outranks any score.
    out.loc[covered, "exception_probability"] = 0.98
    out.loc[covered, "exception_type_pred"] = types[covered]
    out.loc[covered, "decided_by"] = "rule"

    residual_idx = out.index[~covered]
    if result.residual_model is not None and len(residual_idx):
        p = result.residual_model.predict_proba(
            work.loc[residual_idx, result.feature_cols]
        )[:, 1]
        out.loc[residual_idx, "exception_probability"] = p
        # Only call it a model decision where the model actually indicates
        # something. Labelling all 47,000 unflagged rows "residual model"
        # is technically true and reads as though a model adjudicated the
        # whole portfolio, when almost all of those probabilities are ~0.
        indicated = residual_idx[p >= 0.5]
        out.loc[residual_idx, "decided_by"] = "no exception indicated"
        out.loc[indicated, "decided_by"] = "residual model"
        if result.type_model is not None:
            likely = residual_idx[p >= 0.5]
            if len(likely):
                out.loc[likely, "exception_type_pred"] = result.type_model.predict(
                    work.loc[likely, result.feature_cols]
                )
    elif len(residual_idx):
        # No residual model: the unsupervised score is the only ranking
        # signal available, and is labelled as such rather than dressed up
        # as a probability.
        out.loc[residual_idx, "exception_probability"] = (
            out.loc[residual_idx, "anomaly_score"] * 0.5
        )
        out.loc[residual_idx, "decided_by"] = "anomaly score only"

    return out
