"""
Multi-source reconciliation between the main monthly panel and
servicer_updates.csv.

Most entries will read servicer_updates.csv, note that it disagrees with
the panel, and write a paragraph about it. Measured on this data, source
disagreement is not just a data-quality curiosity -- rows where the
servicer balance disagrees with the panel carry a 2.69% exception rate
against 1.78% overall, a 1.5x lift. So reconciliation output is treated
here as a FEATURE SOURCE for the anomaly layer, not only as a report.

Three distinct problems are separated, because they have different
causes and different remedies:

  1. VALUE CONFLICT  - both sources describe the same (loan, month) and
     disagree on a value. Someone has to decide which to believe.
  2. STALENESS       - a source's last_updated_at is far from the month it
     describes, so its value may be correct but obsolete. A negative lag
     (stamped before the month it reports on) is impossible, not merely
     late.
  3. ORPHAN RECORD   - a servicer row with no matching panel row at all.
     Not a conflict; a coverage gap, and it points at a different
     upstream failure.

PHASE: 2
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import LOAN_ID_COLUMN, TIME_COLUMN

logger = logging.getLogger(__name__)

DEFAULT_BALANCE_TOLERANCE = 0.01  # 1% relative
STALE_DAYS = 45

CONFLICT_COLUMNS = ("current_balance", "current_status", "days_past_due")


@dataclass
class ReconciliationResult:
    """Per-panel-row flags plus the source-level summary."""

    record_flags: pd.DataFrame
    summary: pd.DataFrame
    source_trust: pd.DataFrame
    n_servicer_rows: int
    n_matched: int
    n_orphan_servicer_rows: int

    def describe(self) -> str:
        lines = [
            "servicer rows              : {:,}".format(self.n_servicer_rows),
            "matched to a panel row     : {:,}".format(self.n_matched),
            "orphan (no panel row)      : {:,}".format(self.n_orphan_servicer_rows),
            "panel rows with a conflict : {:,}".format(
                int(self.record_flags["source_conflict"].sum())
            ),
            "panel rows with a stale src: {:,}".format(
                int(self.record_flags["source_stale"].sum())
            ),
        ]
        return "\n".join(lines)


def reconcile(
    panel: pd.DataFrame,
    servicer: pd.DataFrame,
    balance_tolerance: float = DEFAULT_BALANCE_TOLERANCE,
    stale_days: int = STALE_DAYS,
) -> ReconciliationResult:
    """Match the servicer feed against the panel and flag disagreement.

    Returns flags indexed exactly like `panel`, so the output can be
    concatenated straight onto the feature matrix.
    """
    keys = [LOAN_ID_COLUMN, TIME_COLUMN]

    if servicer is None or len(servicer) == 0:
        # The servicer feed is optional per the loader, so reconciliation
        # must degrade to "nothing to compare against" rather than raising.
        # Flags are still returned with the full schema so downstream
        # feature code does not need to branch on their absence.
        logger.info("no servicer feed supplied; reconciliation is a no-op")
        flags = pd.DataFrame(index=panel.index)
        flags["has_servicer_record"] = False
        flags["source_conflict"] = False
        flags["source_stale"] = False
        flags["balance_conflict"] = False
        flags["balance_rel_diff"] = 0.0
        flags["servicer_update_lag_days"] = 0
        if "last_updated_at" in panel.columns:
            flags["panel_update_lag_days"] = (
                panel["last_updated_at"] - panel[TIME_COLUMN]
            ).dt.days.fillna(0)
        else:
            flags["panel_update_lag_days"] = 0
        return ReconciliationResult(
            record_flags=flags,
            summary=pd.DataFrame([{"check": "servicer rows", "count": 0}]),
            source_trust=_source_trust_scores(panel, flags),
            n_servicer_rows=0,
            n_matched=0,
            n_orphan_servicer_rows=0,
        )

    n_servicer = len(servicer)

    sv = servicer.drop_duplicates(subset=keys, keep="last")
    if len(sv) < n_servicer:
        logger.warning(
            "servicer feed had %d duplicate (loan, month) rows; kept the last",
            n_servicer - len(sv),
        )

    merged = panel[keys].merge(
        sv, on=keys, how="left", suffixes=("", "_sv"), indicator=True
    )
    merged.index = panel.index
    matched = merged["_merge"] == "both"

    flags = pd.DataFrame(index=panel.index)
    flags["has_servicer_record"] = matched

    # ---- 1. value conflicts ------------------------------------------
    detail = {}
    conflict_any = pd.Series(False, index=panel.index)

    if "current_balance" in panel.columns and "current_balance" in sv.columns:
        panel_bal = panel["current_balance"].astype(float)
        sv_bal = merged["current_balance"].astype(float)
        rel = (sv_bal - panel_bal).abs() / panel_bal.replace(0, np.nan)
        conflict = matched & (rel > balance_tolerance)
        flags["balance_conflict"] = conflict.fillna(False)
        flags["balance_rel_diff"] = rel.fillna(0.0)
        conflict_any = conflict_any | flags["balance_conflict"]
        detail["current_balance"] = int(conflict.sum())

    for col in ("current_status", "days_past_due"):
        if col in panel.columns and col in sv.columns:
            conflict = matched & (merged[col] != panel[col])
            # NaN != NaN is True in pandas comparison; do not call that a
            # conflict, it is a shared gap.
            both_null = merged[col].isna() & panel[col].isna()
            conflict = conflict & ~both_null
            flags[col + "_conflict"] = conflict.fillna(False)
            conflict_any = conflict_any | flags[col + "_conflict"]
            detail[col] = int(conflict.sum())

    flags["source_conflict"] = conflict_any

    # ---- 2. staleness -------------------------------------------------
    if "last_updated_at" in sv.columns:
        lag = (merged["last_updated_at"] - panel[TIME_COLUMN]).dt.days
        flags["servicer_update_lag_days"] = lag.fillna(0)
        flags["source_stale"] = (matched & ((lag > stale_days) | (lag < 0))).fillna(False)
    else:
        flags["servicer_update_lag_days"] = 0
        flags["source_stale"] = False

    if "last_updated_at" in panel.columns:
        panel_lag = (panel["last_updated_at"] - panel[TIME_COLUMN]).dt.days
        flags["panel_update_lag_days"] = panel_lag.fillna(0)
    else:
        flags["panel_update_lag_days"] = 0

    # ---- 3. orphans ---------------------------------------------------
    reverse = sv[keys].merge(panel[keys], on=keys, how="left", indicator=True)
    n_orphan = int((reverse["_merge"] == "left_only").sum())

    summary = pd.DataFrame(
        [
            {"check": "servicer rows", "count": n_servicer},
            {"check": "matched to panel", "count": int(matched.sum())},
            {"check": "orphan servicer rows", "count": n_orphan},
            {"check": "panel rows with any conflict", "count": int(conflict_any.sum())},
            {"check": "panel rows with stale source", "count": int(flags["source_stale"].sum())},
        ]
        + [{"check": "conflict: " + k, "count": v} for k, v in detail.items()]
    )

    source_trust = _source_trust_scores(panel, flags)

    return ReconciliationResult(
        record_flags=flags,
        summary=summary,
        source_trust=source_trust,
        n_servicer_rows=n_servicer,
        n_matched=int(matched.sum()),
        n_orphan_servicer_rows=n_orphan,
    )


def _source_trust_scores(panel: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Per-upstream-system conflict and staleness rates.

    Answers "which feed should a reviewer believe?" with a number instead
    of a shrug. A system whose records conflict more often and update
    later is the one to escalate.
    """
    if "source_system" not in panel.columns:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "source_system": panel["source_system"],
            "conflict": flags["source_conflict"].astype(float),
            "stale": flags["source_stale"].astype(float),
            "lag": flags["panel_update_lag_days"].astype(float),
        }
    )
    agg = df.groupby("source_system").agg(
        n_records=("conflict", "size"),
        conflict_rate=("conflict", "mean"),
        stale_rate=("stale", "mean"),
        median_update_lag_days=("lag", "median"),
    )
    # Trust falls with both conflict and staleness; bounded to [0, 1] so
    # it can be read as a score rather than an arbitrary index.
    agg["trust_score"] = (
        1.0 - agg["conflict_rate"].clip(0, 1) * 0.6 - agg["stale_rate"].clip(0, 1) * 0.4
    ).clip(0, 1)
    return agg.round(4).reset_index()
