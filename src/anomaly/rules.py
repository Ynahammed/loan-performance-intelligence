"""
Deterministic rule engine over validation_rules.json.

WHY RULES FIRST, ML SECOND
--------------------------
Most of the exceptions in this data are not statistical anomalies -- they
are logical impossibilities. A balance that grows on a loan with no
modification, a reporting month before origination, a loan reporting
Current after it already prepaid: none of these need a model, and using
one would make an auditable defect unauditable. So the rule engine runs
first and is measured, and the ML layer is asked only to work on what the
rules cannot reach.

That measurement is the point. `benchmark_against_labels` reports, per
rule, the precision and recall against the shipped exception_type labels,
so the rule/ML division of labour is a number rather than a claim.

Rules are declared in data/validation_rules.json and implemented here by
rule_id. A declared rule with no implementation is reported as
unimplemented rather than silently passing -- a rule engine that quietly
skips checks is worse than no rule engine.

PHASE: 2
STATUS: implemented.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, LOAN_ID_COLUMN, TIME_COLUMN

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}

# Which shipped exception_type each rule is expected to detect. Used only
# for benchmarking -- the rules themselves never look at the labels.
RULE_TO_EXCEPTION_TYPE = {
    "balance_non_increasing": "balance_inconsistency",
    "balance_within_original": "balance_inconsistency",
    "date_order_valid": "date_invalid",
    "document_completeness_at_closure": "document_gap",
    "terminal_state_immutability": "status_reversal",
}

# Rules that flag a risk SIGNAL rather than a labelled defect. They feed
# the anomaly model as features; pooling them with the detector rules when
# scoring against exception_required would understate the detectors.
ADVISORY_RULES = {"servicer_source_agreement", "stale_record"}

TERMINAL_STATES = ("Default", "Prepaid")

# Expected days-past-due band for each reported status. Sourced from the
# observed distribution on clean records, not invented.
DPD_BANDS = {
    "Current": (0, 0),
    "30DPD": (30, 59),
    "60DPD": (60, 89),
    "90DPD": (90, 10_000),
    "Default": (90, 10_000),
    "Prepaid": (0, 0),
}

# A servicer update timestamped more than this many days after the
# reporting month is treated as stale. Clean records land within 9 days.
STALE_UPDATE_DAYS = 45


def load_rules(rules_path: Path | str = None) -> list:
    """Read the rule declarations. The JSON is the source of truth for
    which rules exist and how severe they are; this module only supplies
    the implementations."""
    rules_path = Path(rules_path or (DATA_DIR / "validation_rules.json"))
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    return payload.get("rules", [])


# ---------------------------------------------------------------------------
# Rule implementations. Each returns a boolean Series aligned to `df.index`,
# True where the rule is VIOLATED.
# ---------------------------------------------------------------------------


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    key = "month_index" if "month_index" in df.columns else TIME_COLUMN
    return df.sort_values([LOAN_ID_COLUMN, key])


def rule_balance_non_increasing(df: pd.DataFrame, **kw) -> pd.Series:
    """current_balance must not rise month-over-month without a modification.

    Amortisation only goes one way. The one legitimate exception is a
    modification (capitalised arrears), which is why modification_flag
    gates the check rather than being ignored.
    """
    s = _sorted(df)
    prev = s.groupby(LOAN_ID_COLUMN)["current_balance"].shift(1)
    rose = s["current_balance"] > prev
    modified = s.get("modification_flag", pd.Series(0, index=s.index)).fillna(0) == 1
    return (rose & ~modified).reindex(df.index).fillna(False)



def rule_balance_within_original(df: pd.DataFrame, **kw) -> pd.Series:
    """current_balance must never exceed original_balance without a
    modification.

    Added after the fact, and worth recording why. The declared
    `balance_non_increasing` rule compares each month against the previous
    one, so it cannot fire on a loan's FIRST observed month -- there is no
    previous row. That gap left 33 balance_inconsistency records unflagged,
    and a residual ML model then "discovered" them at ROC-AUC 1.000, which
    is what a too-good-to-be-true number usually means: not a brilliant
    model, a deficient rule.

    The profiler's own cross-column check had already found the better
    detector (`balance_within_original`, 574 violations against the 574
    labelled cases). This promotes that check into the rule engine, where
    the finding is deterministic and auditable instead of being laundered
    through a classifier.
    """
    if not {"current_balance", "original_balance"}.issubset(df.columns):
        return pd.Series(False, index=df.index)
    modified = df.get("modification_flag", pd.Series(0, index=df.index)).fillna(0) == 1
    over = df["current_balance"] > df["original_balance"] * 1.0001
    return (over & ~modified).fillna(False)


def rule_date_order_valid(df: pd.DataFrame, **kw) -> pd.Series:
    """A loan cannot report performance before it was originated."""
    return (df[TIME_COLUMN] < df["origination_month"]).fillna(False)


def rule_dpd_status_consistency(df: pd.DataFrame, **kw) -> pd.Series:
    """days_past_due must sit inside the band its status implies."""
    if "days_past_due" not in df.columns:
        return pd.Series(False, index=df.index)
    lo = df["current_status"].map(lambda s: DPD_BANDS.get(s, (0, 10_000))[0])
    hi = df["current_status"].map(lambda s: DPD_BANDS.get(s, (0, 10_000))[1])
    dpd = df["days_past_due"]
    return ((dpd < lo) | (dpd > hi)).fillna(False)


def rule_terminal_state_immutability(df: pd.DataFrame, **kw) -> pd.Series:
    """Once Default or Prepaid, a loan cannot report a non-terminal state.

    Flags the RESURRECTED rows, not the terminal row itself -- the
    defective record is the one that came back, and that is the row a
    reviewer needs to open.
    """
    s = _sorted(df)
    is_terminal = s["current_status"].isin(TERMINAL_STATES)
    # Both the cumsum AND the shift must be grouped. Shifting after the
    # groupby-cumsum shifts across loan boundaries instead of within each
    # loan, so every loan that ends in a terminal state leaks a false flag
    # onto the next loan's first row -- 476 spurious flags against 6 real
    # ones on this panel, which the precision benchmark caught.
    seen_terminal = (
        is_terminal.groupby(s[LOAN_ID_COLUMN]).cumsum()
        .groupby(s[LOAN_ID_COLUMN]).shift(1).fillna(0)
    )
    resurrected = (seen_terminal > 0) & (~is_terminal)
    return resurrected.reindex(df.index).fillna(False)


def rule_document_completeness_at_closure(df: pd.DataFrame, **kw) -> pd.Series:
    """Closed loans should have a complete document set.

    'Pending' is deliberately NOT treated as a violation. Measured on this
    data, flagging Pending as well drops precision from 83% to 35% -- a
    pending document at closure is a workflow state, a missing one is a
    control gap. That distinction is the difference between a rule a
    reviewer trusts and one they learn to ignore.
    """
    if "document_status" not in df.columns:
        return pd.Series(False, index=df.index)
    terminal = df["current_status"].isin(TERMINAL_STATES)
    missing = df["document_status"].isin(["Missing"]) | df["document_status"].isna()
    return (terminal & missing).fillna(False)


def rule_servicer_source_agreement(df: pd.DataFrame, reconciliation=None, **kw) -> pd.Series:
    """Cross-source disagreement, supplied by src/data/reconciliation.py."""
    if reconciliation is None or "source_conflict" not in getattr(
        reconciliation, "columns", []
    ):
        return pd.Series(False, index=df.index)
    return reconciliation["source_conflict"].reindex(df.index).fillna(False)


def rule_stale_record(df: pd.DataFrame, **kw) -> pd.Series:
    """last_updated_at far from the month it describes.

    Not declared in validation_rules.json, but the data demands it: clean
    records update within 9 days of the reporting month, while defective
    ones run to 2,195 days. Negative lags (an update stamped BEFORE the
    month it reports on) are flagged too -- those are impossible, not
    merely late.
    """
    if "last_updated_at" not in df.columns:
        return pd.Series(False, index=df.index)
    lag = (df["last_updated_at"] - df[TIME_COLUMN]).dt.days
    return ((lag > STALE_UPDATE_DAYS) | (lag < 0)).fillna(False)


RULE_IMPLEMENTATIONS = {
    "balance_non_increasing": rule_balance_non_increasing,
    "balance_within_original": rule_balance_within_original,
    "date_order_valid": rule_date_order_valid,
    "dpd_status_consistency": rule_dpd_status_consistency,
    "terminal_state_immutability": rule_terminal_state_immutability,
    "document_completeness_at_closure": rule_document_completeness_at_closure,
    "servicer_source_agreement": rule_servicer_source_agreement,
    "stale_record": rule_stale_record,
}

# Rules we add beyond the shipped JSON, so the catalog can show which
# checks are organizer-declared and which we introduced.
LOCAL_RULES = [
    {
        "rule_id": "balance_within_original",
        "description": "current_balance must not exceed original_balance "
                       "without a modification (catches the first-month case "
                       "that balance_non_increasing structurally cannot)",
        "severity": "high",
        "applies_to": "loan_monthly_performance",
        "source": "added by us",
    },
    {
        "rule_id": "stale_record",
        "description": "last_updated_at is more than {} days after the reporting "
                       "month, or precedes it".format(STALE_UPDATE_DAYS),
        "severity": "medium",
        "applies_to": "loan_monthly_performance",
        "source": "added by us",
    }
]


@dataclass
class RuleEngineResult:
    flags: pd.DataFrame
    catalog: pd.DataFrame
    unimplemented: list = field(default_factory=list)

    def any_violation(self) -> pd.Series:
        return self.flags.any(axis=1)

    def violation_count(self) -> pd.Series:
        return self.flags.sum(axis=1)

    def severity_score(self) -> pd.Series:
        """Severity-weighted violation score per record (0 = clean)."""
        weights = self.catalog.set_index("rule_id")["weight"]
        score = pd.Series(0.0, index=self.flags.index)
        for rule_id in self.flags.columns:
            score = score + self.flags[rule_id].astype(float) * weights.get(rule_id, 1.0)
        return score

    def benchmark_against_labels(
        self, exception_required: pd.Series, exception_type: pd.Series = None
    ) -> pd.DataFrame:
        """Per-rule precision/recall against the shipped labels.

        This is the number that establishes where rules stop and ML has to
        start. Rules with no corresponding label type are reported with
        their flag count only, not scored against a target they were never
        meant to hit.
        """
        rows = []
        for rule_id in self.flags.columns:
            flagged = self.flags[rule_id]
            target_type = RULE_TO_EXCEPTION_TYPE.get(rule_id)
            row = {"rule_id": rule_id, "n_flagged": int(flagged.sum())}

            if target_type is not None and exception_type is not None:
                truth = exception_type == target_type
                tp = int((flagged & truth).sum())
                row.update(
                    {
                        "target_label": target_type,
                        "n_labelled": int(truth.sum()),
                        "true_positives": tp,
                        "precision": round(tp / max(int(flagged.sum()), 1), 4),
                        "recall": round(tp / max(int(truth.sum()), 1), 4),
                    }
                )
            else:
                row.update({"target_label": "(no labelled type)"})

            row["exception_rate_when_flagged"] = round(
                float(exception_required[flagged].mean()) if flagged.any() else 0.0, 4
            )
            rows.append(row)

        out = pd.DataFrame(rows)
        n_true = int((exception_required == 1).sum())

        def _roll_up(name: str, mask: pd.Series) -> dict:
            tp = int((mask & (exception_required == 1)).sum())
            return {
                "rule_id": name,
                "n_flagged": int(mask.sum()),
                "target_label": "exception_required",
                "n_labelled": n_true,
                "true_positives": tp,
                "precision": round(tp / max(int(mask.sum()), 1), 4),
                "recall": round(tp / max(n_true, 1), 4),
                "exception_rate_when_flagged": round(
                    float(exception_required[mask].mean()) if mask.any() else 0.0, 4
                ),
            }

        # Detector rules target a labelled exception type. Advisory rules
        # (source conflict, staleness) are risk SIGNALS that feed the
        # anomaly model -- scoring them against exception_required would
        # understate the detectors they are pooled with.
        detector_cols = [c for c in self.flags.columns if c not in ADVISORY_RULES]
        if detector_cols:
            out.loc[len(out)] = _roll_up(
                "ANY DETECTOR RULE", self.flags[detector_cols].any(axis=1)
            )
        out.loc[len(out)] = _roll_up("ANY RULE (incl. advisory)", self.any_violation())
        return out


def evaluate_rules(
    df: pd.DataFrame,
    rules: list = None,
    reconciliation: pd.DataFrame = None,
    include_local_rules: bool = True,
) -> RuleEngineResult:
    """Run every implemented rule over `df`."""
    rules = list(rules if rules is not None else load_rules())
    for r in rules:
        r.setdefault("source", "validation_rules.json")
    if include_local_rules:
        rules = rules + LOCAL_RULES

    flags = pd.DataFrame(index=df.index)
    catalog_rows = []
    unimplemented = []

    for rule in rules:
        rule_id = rule["rule_id"]
        impl = RULE_IMPLEMENTATIONS.get(rule_id)
        if impl is None:
            unimplemented.append(rule_id)
            logger.warning("Rule %r declared but not implemented", rule_id)
            catalog_rows.append(
                {
                    "rule_id": rule_id,
                    "severity": rule.get("severity", "medium"),
                    "weight": SEVERITY_WEIGHTS.get(rule.get("severity", "medium"), 1.0),
                    "source": rule.get("source", ""),
                    "implemented": False,
                    "n_flagged": 0,
                    "description": rule.get("description", ""),
                }
            )
            continue

        violated = impl(df, reconciliation=reconciliation).astype(bool)
        flags[rule_id] = violated
        catalog_rows.append(
            {
                "rule_id": rule_id,
                "severity": rule.get("severity", "medium"),
                "weight": SEVERITY_WEIGHTS.get(rule.get("severity", "medium"), 1.0),
                "source": rule.get("source", ""),
                "implemented": True,
                "n_flagged": int(violated.sum()),
                "description": rule.get("description", ""),
            }
        )

    return RuleEngineResult(
        flags=flags,
        catalog=pd.DataFrame(catalog_rows),
        unimplemented=unimplemented,
    )


def combine_rule_and_ml_signals(
    rule_score: pd.Series,
    anomaly_score: pd.Series,
    rule_weight: float = 0.6,
) -> pd.Series:
    """Blend the deterministic rule score with the unsupervised anomaly score.

    Rules are weighted higher by default because a rule violation is a
    fact about the record while an anomaly score is a statement about the
    population. Both are min-max normalised first so the blend is not
    dominated by whichever happens to have the wider raw range.
    """
    def _norm(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = float(s.min()), float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(0.0, index=s.index)
        return (s - lo) / (hi - lo)

    return rule_weight * _norm(rule_score) + (1 - rule_weight) * _norm(anomaly_score)
