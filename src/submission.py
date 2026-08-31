"""
Assemble and validate submission.csv.

Malformed output is a disqualification-adjacent failure: a judge who
cannot load the file cannot score anything else that was built. So the
schema is read from `submission_template.csv` rather than hardcoded --
if the organizer ships a revised template, this picks it up instead of
producing a confidently wrong file against a stale column list -- and
`validate_submission` runs every check that could make the file
unreadable or internally inconsistent before it is written.

The validator is deliberately strict about things that are easy to get
wrong and invisible afterwards:

  - column set AND order match the template exactly
  - one output row per input row, in the input's order, including
    duplicated (loan_id, reporting_month) pairs
  - probabilities inside [0, 1] with no NaN or infinity
  - `next_state_pred` drawn from the known state vocabulary
  - `model_confidence` actually varies (a constant column is the classic
    silent placeholder that survives to submission)

RECOMMENDED ACTION IS A RECOMMENDATION
--------------------------------------
`recommended_action` is produced by a deterministic policy over the model
outputs, not by a model, and every value it can take is a review
instruction rather than a decision about the loan. That is the challenge
rule about LLM and model output being advisory, applied to the one
column where it would be easiest to quietly cross the line.

PHASE: 12
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, LOAN_ID_COLUMN, STATE_ORDER, TIME_COLUMN

logger = logging.getLogger(__name__)

TEMPLATE_PATH = DATA_DIR / "submission_template.csv"

PROBABILITY_COLUMNS = (
    "next_3m_delinquency_prob",
    "next_6m_delinquency_prob",
    "next_12m_default_prob",
    "next_12m_prepayment_prob",
    "next_state_confidence",
    "exception_confidence",
    "anomaly_score",
    "model_confidence",
)

# The four scored target probabilities. Held to a ranking check that the
# confidence/score columns are not: those may legitimately be coarse
# (a fired rule gives every flagged record the same confidence), while a
# target probability that ties half the portfolio has lost the ordering
# it is scored on.
TARGET_PROBABILITY_COLUMNS = (
    "next_3m_delinquency_prob",
    "next_6m_delinquency_prob",
    "next_12m_default_prob",
    "next_12m_prepayment_prob",
)
MAX_TIE_SHARE = 0.25
MIN_DISTINCT_RATIO = 0.05

# Thresholds for the action policy. Configurable rather than buried in an
# if-chain, so a reviewer can see and change the operating points.
ACTION_THRESHOLDS = {
    "default_review": 0.05,
    "delinquency_review": 0.25,
    "delinquency_monitor": 0.10,
    "anomaly_review": 0.90,
    "prepayment_note": 0.40,
}


def load_template_columns(path: Path | str = None) -> list:
    """Read the required column order from the organizer's template."""
    path = Path(path or TEMPLATE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "submission template not found at {}; the required schema must "
            "come from the organizer's file, not from a hardcoded list".format(path)
        )
    return list(pd.read_csv(path, nrows=0).columns)


@dataclass
class ValidationReport:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    checks: pd.DataFrame = None

    def describe(self) -> str:
        lines = ["PASS" if self.ok else "FAIL"]
        if self.checks is not None:
            lines.append("")
            lines.append(self.checks.to_string(index=False))
        for e in self.errors:
            lines.append("ERROR   : " + e)
        for w in self.warnings:
            lines.append("WARNING : " + w)
        return "\n".join(lines)


def recommend_action(row: pd.Series) -> str:
    """Deterministic review routing. A recommendation, never a decision.

    Ordered by severity: a data exception outranks a risk score, because a
    record that may be wrong should be fixed before it is acted on.
    """
    t = ACTION_THRESHOLDS
    if row.get("exception_required_pred", 0) == 1:
        return "REVIEW - data exception, verify record before relying on scores"
    if row.get("anomaly_score", 0) >= t["anomaly_review"]:
        return "REVIEW - unusual record pattern, confirm data quality"
    if row.get("next_12m_default_prob", 0) >= t["default_review"]:
        return "REVIEW - elevated 12-month default risk"
    if row.get("next_3m_delinquency_prob", 0) >= t["delinquency_review"]:
        return "REVIEW - elevated near-term delinquency risk"
    if row.get("next_3m_delinquency_prob", 0) >= t["delinquency_monitor"]:
        return "MONITOR - early delinquency signal"
    if row.get("next_12m_prepayment_prob", 0) >= t["prepayment_note"]:
        return "MONITOR - elevated prepayment likelihood"
    return "NO ACTION - within normal range"


def build_submission(
    keys: pd.DataFrame,
    probabilities: pd.DataFrame,
    next_state: pd.DataFrame,
    exceptions: pd.DataFrame,
    anomaly_scores: pd.Series,
    top_drivers: pd.Series,
    model_confidence: pd.Series,
    template_columns: list = None,
) -> pd.DataFrame:
    """Assemble the submission frame in the template's column order."""
    columns = template_columns or load_template_columns()
    idx = keys.index

    out = pd.DataFrame(index=idx)
    out["loan_id"] = keys[LOAN_ID_COLUMN].values
    out["reporting_month"] = pd.to_datetime(
        keys[TIME_COLUMN]
    ).dt.strftime("%Y-%m-%d").values

    mapping = {
        "next_3m_delinquency_prob": "next_3m_delinquency_flag",
        "next_6m_delinquency_prob": "next_6m_delinquency_flag",
        "next_12m_default_prob": "next_12m_default_flag",
        "next_12m_prepayment_prob": "next_12m_prepayment_flag",
    }
    for out_col, source in mapping.items():
        if source in probabilities.columns:
            out[out_col] = probabilities[source].reindex(idx).values
        else:
            # An absent target is written as NaN and caught by the
            # validator, rather than filled with a plausible-looking zero.
            out[out_col] = np.nan
            logger.warning("target %s missing; %s will be blank", source, out_col)

    out["next_state_pred"] = next_state["next_state_pred"].reindex(idx).values
    out["next_state_confidence"] = (
        next_state["next_state_confidence"].reindex(idx).values
    )
    out["exception_required_pred"] = (
        exceptions["exception_required_pred"].reindex(idx).astype(int).values
    )
    out["exception_type_pred"] = (
        exceptions["exception_type_pred"].reindex(idx).values
    )
    out["exception_confidence"] = (
        exceptions["exception_confidence"].reindex(idx).values
    )
    out["anomaly_score"] = anomaly_scores.reindex(idx).values
    out["top_drivers"] = top_drivers.reindex(idx).values
    out["model_confidence"] = model_confidence.reindex(idx).values
    out["recommended_action"] = out.apply(recommend_action, axis=1)

    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise ValueError("assembled submission is missing columns: {}".format(missing))
    return out[columns].reset_index(drop=True)


def validate_submission(
    submission: pd.DataFrame,
    reference_panel: pd.DataFrame,
    template_columns: list = None,
) -> ValidationReport:
    """Every check that could make the file unusable, run before writing."""
    columns = template_columns or load_template_columns()
    errors, warnings, checks = [], [], []

    def check(name, passed, detail=""):
        checks.append({"check": name,
                       "result": "pass" if passed else "FAIL",
                       "detail": detail})
        return passed

    if not check("column order matches template",
                 list(submission.columns) == columns,
                 "expected {} columns".format(len(columns))):
        errors.append(
            "column order differs from the template. expected: {} | got: {}"
            .format(columns, list(submission.columns))
        )

    if not check("one row per input row",
                 len(submission) == len(reference_panel),
                 "{:,} rows vs {:,} input rows".format(
                     len(submission), len(reference_panel))):
        errors.append(
            "row count {:,} does not match the panel being scored ({:,})"
            .format(len(submission), len(reference_panel))
        )

    if len(submission) == len(reference_panel):
        same_ids = (
            submission["loan_id"].to_numpy()
            == reference_panel[LOAN_ID_COLUMN].to_numpy()
        ).all()
        if not check("rows align with the input, in order", bool(same_ids)):
            errors.append(
                "loan_id order does not match the input panel; the "
                "submission cannot be joined back row for row"
            )

    for col in PROBABILITY_COLUMNS:
        if col not in submission.columns:
            continue
        values = pd.to_numeric(submission[col], errors="coerce")
        n_bad = int(((values < 0) | (values > 1)).sum())
        n_null = int(values.isna().sum())
        if not check("{} within [0,1]".format(col), n_bad == 0,
                     "{} out of range".format(n_bad)):
            errors.append("{} has {} values outside [0,1]".format(col, n_bad))
        if not check("{} has no nulls".format(col), n_null == 0,
                     "{} null".format(n_null)):
            errors.append("{} has {} null values".format(col, n_null))

    if "next_state_pred" in submission.columns:
        unknown = set(submission["next_state_pred"].dropna()) - set(STATE_ORDER)
        if not check("next_state_pred uses known states", not unknown,
                     str(sorted(unknown)) if unknown else ""):
            errors.append("next_state_pred contains unknown states: {}"
                          .format(sorted(unknown)))

    if "exception_required_pred" in submission.columns:
        vals = set(pd.to_numeric(
            submission["exception_required_pred"], errors="coerce").dropna().unique())
        if not check("exception_required_pred is 0/1", vals <= {0, 1},
                     str(sorted(vals))):
            errors.append("exception_required_pred must be 0 or 1")

    # A constant confidence column is the classic silent placeholder: it
    # passes every range check and means nothing.
    for col in ("model_confidence", "anomaly_score"):
        if col in submission.columns:
            n_unique = submission[col].nunique()
            if not check("{} varies".format(col), n_unique > 1,
                         "{} distinct values".format(n_unique)):
                errors.append(
                    "{} is constant, which means it is a placeholder rather "
                    "than a computed value".format(col)
                )

    # Excessive ties are the subtler version of the same failure, and the
    # reason this check exists: an isotonic calibrator mapped 8,889 scored
    # rows onto 18 distinct probabilities, with half the portfolio sharing
    # one value. Every range check passed, the column "varied", and the
    # ranking the submission is scored on had been destroyed.
    for col in TARGET_PROBABILITY_COLUMNS:
        if col not in submission.columns:
            continue
        values = pd.to_numeric(submission[col], errors="coerce").dropna()
        if values.empty:
            continue
        tie_share = float(values.value_counts(normalize=True).iloc[0])
        distinct_ratio = values.nunique() / max(len(values), 1)
        passed = tie_share <= MAX_TIE_SHARE and distinct_ratio >= MIN_DISTINCT_RATIO
        if not check("{} preserves ranking".format(col), passed,
                     "largest tie {:.1%}, {} distinct".format(
                         tie_share, values.nunique())):
            errors.append(
                "{} has collapsed: {:.1%} of rows share one value across only "
                "{} distinct values. The probabilities may be individually "
                "calibrated but the ranking is gone, and ranking is what this "
                "column is scored on.".format(col, tie_share, values.nunique())
            )

    for col in ("top_drivers", "recommended_action"):
        if col in submission.columns:
            blank = int(submission[col].isna().sum()
                        + (submission[col].astype(str).str.strip() == "").sum())
            if not check("{} is populated".format(col), blank == 0,
                         "{} blank".format(blank)):
                warnings.append("{} is blank on {} rows".format(col, blank))

    if submission.isna().any().any():
        null_cols = submission.columns[submission.isna().any()].tolist()
        warnings.append("null values present in: {}".format(null_cols))

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        checks=pd.DataFrame(checks),
    )
