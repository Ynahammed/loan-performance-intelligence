"""
Time-aware validation strategy.

This is a *panel* dataset: one row per loan per month, with the same
loan_id appearing in many rows. That makes two failure modes possible,
and this module exists to prevent both.

1. ROW-LEVEL RANDOM SPLIT LEAKAGE
   A plain train_test_split puts LN100000's March row in train and its
   April row in validation. The two rows are near-identical, so the model
   memorises rather than generalises and the metrics are meaningless. The
   challenge rules list this as a disqualification condition. Every split
   produced here is therefore chronological.

2. LABEL-HORIZON LEAKAGE (the subtle one)
   Targets look forward: next_12m_default_flag on a 2023-06 row is
   determined by data through 2024-06. If we cut train/test at 2023-12,
   the training rows from 2023-01..2023-12 carry labels that already
   "know" what happens inside the test period. The fix is a purge gap of
   H months between the end of train and the start of test, where H is
   the label's forward horizon. Standard practice in financial ML, and
   skipped by almost every naive time split.

Note on loan overlap: a loan legitimately appearing in BOTH train (early
months) and test (later months) is NOT leakage under a temporal split --
it is exactly the deployment setting, and it is what the organizer's own
train/test files do (1,455 loans overlap). We additionally expose a
group-disjoint variant as a stricter robustness diagnostic, so both
numbers can be reported rather than argued about.

PHASE: 4
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import LOAN_ID_COLUMN, RANDOM_SEED, TIME_COLUMN

logger = logging.getLogger(__name__)

# Forward horizon in months for each target, used to size the purge gap.
TARGET_HORIZON_MONTHS = {
    "next_3m_delinquency_flag": 3,
    "next_6m_delinquency_flag": 6,
    "next_12m_default_flag": 12,
    "next_12m_prepayment_flag": 12,
    "next_state": 1,
    "exception_required": 0,
    "exception_type": 0,
}


@dataclass
class Split:
    """One train/test split, carrying the reasoning that produced it."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    purge_months: int
    n_purged: int
    strategy: str
    warnings: list

    def describe(self) -> str:
        overlap = len(set(self.train[LOAN_ID_COLUMN]) & set(self.test[LOAN_ID_COLUMN]))
        lines = [
            "strategy      : {}".format(self.strategy),
            "train         : {:,} rows, {:,} loans, through {}".format(
                len(self.train),
                self.train[LOAN_ID_COLUMN].nunique(),
                self.train_end.date(),
            ),
            "purge gap     : {} month(s), {:,} rows withheld".format(
                self.purge_months, self.n_purged
            ),
            "test          : {:,} rows, {:,} loans, from {}".format(
                len(self.test),
                self.test[LOAN_ID_COLUMN].nunique(),
                self.test_start.date(),
            ),
            "loan overlap  : {:,} (expected under a temporal split)".format(overlap),
        ]
        for w in self.warnings:
            lines.append("WARNING       : {}".format(w))
        return "\n".join(lines)


def _month_floor(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def time_aware_split(
    df,
    time_column=TIME_COLUMN,
    test_fraction=0.2,
    purge_months=0,
    train_end=None,
):
    """Chronological split with an optional purge gap.

    `purge_months` should be set to the forward horizon of the target being
    modelled (see TARGET_HORIZON_MONTHS / `split_for_target`). Rows falling
    inside the gap are dropped from BOTH sides -- they are precisely the
    rows whose labels straddle the boundary.
    """
    if time_column not in df.columns:
        raise ValueError("time column {!r} not in dataframe".format(time_column))

    warnings = []
    t = _month_floor(df[time_column])
    months = np.sort(t.dropna().unique())
    if len(months) < 3:
        raise ValueError(
            "only {} distinct month(s) in {}; a temporal split is not "
            "meaningful".format(len(months), time_column)
        )

    if train_end is None:
        # The cut must reserve BOTH the purge gap and the test window before
        # it is chosen. Taking 80% of the months first and then subtracting a
        # 12-month gap pushes test_start past the end of the panel and yields
        # an empty test set -- which is how both 12-month targets silently
        # produced no evaluation at all on the first run.
        n_test = max(1, int(round(len(months) * test_fraction)))
        cut_idx = len(months) - n_test - purge_months - 1
        if cut_idx < 1:
            cut_idx = max(0, len(months) - purge_months - 2)
            warnings.append(
                "panel spans {} months; a {}-month purge leaves too little "
                "room for a {:.0%} test window, so the test set is the "
                "largest that fits".format(len(months), purge_months, test_fraction)
            )
        train_end = pd.Timestamp(months[cut_idx])
    else:
        train_end = pd.Timestamp(train_end)

    test_start = train_end + pd.DateOffset(months=purge_months + 1)

    train = df[t <= train_end]
    test = df[t >= test_start]
    n_purged = len(df) - len(train) - len(test)

    if len(test) == 0:
        warnings.append(
            "purge gap of {} months left an empty test set; the panel is too "
            "short for this horizon".format(purge_months)
        )
    if purge_months == 0:
        warnings.append(
            "no purge gap applied -- only valid for targets with no forward "
            "horizon (e.g. exception flags computed from the row itself)"
        )

    return Split(
        train=train.copy(),
        test=test.copy(),
        train_end=train_end,
        test_start=test_start,
        purge_months=purge_months,
        n_purged=n_purged,
        strategy="purged temporal split",
        warnings=warnings,
    )


def split_for_target(df, target, time_column=TIME_COLUMN, test_fraction=0.2):
    """Temporal split sized to the target's own forward horizon."""
    horizon = TARGET_HORIZON_MONTHS.get(target)
    if horizon is None:
        horizon = 0
        logger.warning(
            "Unknown target %r -- assuming horizon 0, no purge gap applied.", target
        )
    return time_aware_split(
        df, time_column=time_column, test_fraction=test_fraction, purge_months=horizon
    )


def group_disjoint_temporal_split(
    df,
    time_column=TIME_COLUMN,
    group_column=LOAN_ID_COLUMN,
    test_fraction=0.2,
    purge_months=0,
    holdout_loan_fraction=0.3,
    seed=RANDOM_SEED,
):
    """Stricter diagnostic: cut by time AND hold out unseen loans.

    Answers "how much of the score comes from having seen this loan
    before?". Reported alongside the primary temporal split rather than
    instead of it -- the temporal split is the realistic one.
    """
    base = time_aware_split(df, time_column, test_fraction, purge_months)
    rng = np.random.default_rng(seed)
    loans = np.sort(base.train[group_column].unique())
    n_hold = int(len(loans) * holdout_loan_fraction)
    holdout = set(rng.choice(loans, size=n_hold, replace=False))

    train = base.train[~base.train[group_column].isin(holdout)]
    test = base.test[base.test[group_column].isin(holdout)]

    warnings = list(base.warnings)
    if len(test) == 0:
        warnings.append("no held-out loans survive into the test window")

    return Split(
        train=train,
        test=test,
        train_end=base.train_end,
        test_start=base.test_start,
        purge_months=purge_months,
        n_purged=base.n_purged,
        strategy="group-disjoint temporal split (strict diagnostic)",
        warnings=warnings,
    )


def rolling_time_series_splits(
    df,
    time_column=TIME_COLUMN,
    n_splits=4,
    purge_months=0,
    min_train_months=12,
):
    """Expanding-window walk-forward CV, each fold purged.

    Fold k trains on everything up to month m_k and tests on the window
    after the purge gap. Guards against tuning to one lucky test slice.
    """
    t = _month_floor(df[time_column])
    months = np.sort(t.dropna().unique())
    if len(months) < min_train_months + n_splits + purge_months:
        raise ValueError(
            "panel spans {} months; not enough for {} folds with a {}-month "
            "purge and {}-month minimum train".format(
                len(months), n_splits, purge_months, min_train_months
            )
        )

    usable = len(months) - min_train_months - purge_months
    step = max(1, usable // n_splits)
    splits = []
    for k in range(n_splits):
        end_idx = min_train_months + k * step
        if end_idx >= len(months) - purge_months - 1:
            break
        train_end = pd.Timestamp(months[end_idx])
        test_start = train_end + pd.DateOffset(months=purge_months + 1)
        test_end = train_end + pd.DateOffset(months=purge_months + step)

        train = df[t <= train_end]
        test = df[(t >= test_start) & (t <= test_end)]
        if len(test) == 0:
            continue
        splits.append(
            Split(
                train=train.copy(),
                test=test.copy(),
                train_end=train_end,
                test_start=test_start,
                purge_months=purge_months,
                n_purged=0,
                strategy="walk-forward fold {}/{}".format(k + 1, n_splits),
                warnings=[],
            )
        )
    return splits
