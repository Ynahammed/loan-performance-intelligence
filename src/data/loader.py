"""
Loads and merges the raw data pack: loan_static_attributes.csv,
loan_monthly_performance_train/test.csv, servicer_updates.csv.

Responsibilities:
- Read CSVs with explicit dtypes where known (avoid pandas guessing wrong).
- Parse date columns (origination_month, reporting_month, last_updated_at).
- Join static attributes onto the monthly panel by loan_id.
- Validate expected columns are present; raise a clear error if not
  (rather than failing deep inside the pipeline later).
- NEVER silently drop rows -- log counts at every step.

PHASE: 2
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, LOAN_ID_COLUMN, TIME_COLUMN

logger = logging.getLogger(__name__)

DATE_COLUMNS = ("reporting_month", "origination_month", "last_updated_at")

# Columns the panel must have for anything downstream to work. Anything
# beyond this is optional and handled defensively, so the organizer's real
# data pack can carry extra fields without breaking the pipeline.
REQUIRED_PANEL_COLUMNS = (
    "loan_id",
    "month_index",
    "reporting_month",
    "current_status",
)


@dataclass
class DataPack:
    """Everything read off disk, plus the counts we logged getting there."""

    train: pd.DataFrame
    test: pd.DataFrame | None = None
    static: pd.DataFrame | None = None
    servicer_updates: pd.DataFrame | None = None
    load_log: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "test", "static", "servicer_updates"):
            df = getattr(self, name)
            if df is None:
                rows.append({"table": name, "rows": 0, "columns": 0, "loans": 0})
                continue
            loans = df[LOAN_ID_COLUMN].nunique() if LOAN_ID_COLUMN in df.columns else 0
            rows.append(
                {"table": name, "rows": len(df), "columns": df.shape[1], "loans": loans}
            )
        return pd.DataFrame(rows)


def _read_csv(path: Path, log: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        log.append(f"MISSING  {path.name}")
        logger.warning("Optional file not found: %s", path)
        return None
    df = pd.read_csv(path)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    log.append(f"loaded   {path.name}: {len(df):,} rows x {df.shape[1]} cols")
    logger.info("Loaded %s: %s rows", path.name, f"{len(df):,}")
    return df


def _validate_panel(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s): {missing}. "
            f"Present columns: {sorted(df.columns)}"
        )


def load_data_pack(data_dir: Path | str = DATA_DIR) -> DataPack:
    """Read the full data pack from `data_dir`.

    Only the training panel is mandatory. Everything else is optional so
    the pipeline still runs against a partial pack (and so the organizer's
    real pack can drop in without code changes).
    """
    data_dir = Path(data_dir)
    log: list[str] = []

    train = _read_csv(data_dir / "loan_monthly_performance_train.csv", log)
    if train is None:
        raise FileNotFoundError(
            f"loan_monthly_performance_train.csv not found in {data_dir}"
        )
    _validate_panel(train, "train panel")

    test = _read_csv(data_dir / "loan_monthly_performance_test.csv", log)
    if test is not None:
        _validate_panel(test, "test panel")

    static = _read_csv(data_dir / "loan_static_attributes.csv", log)
    servicer = _read_csv(data_dir / "servicer_updates.csv", log)

    pack = DataPack(
        train=train, test=test, static=static, servicer_updates=servicer, load_log=log
    )
    return pack


def attach_static_attributes(
    panel: pd.DataFrame, static: pd.DataFrame | None, suffix: str = "_static"
) -> pd.DataFrame:
    """Left-join origination-level attributes onto a monthly panel.

    Row count is asserted unchanged -- a many-to-one join that silently
    fans out rows is the classic way to corrupt a panel dataset.
    """
    if static is None:
        return panel

    dupes = static[LOAN_ID_COLUMN].duplicated().sum()
    if dupes:
        raise ValueError(
            f"loan_static_attributes has {dupes} duplicate loan_id rows; "
            "joining would fan out the panel."
        )

    # Only bring across columns the panel does not already carry, plus
    # anything genuinely origination-only (e.g. vintage, original_term_months).
    new_cols = [
        c for c in static.columns if c == LOAN_ID_COLUMN or c not in panel.columns
    ]
    before = len(panel)
    merged = panel.merge(static[new_cols], on=LOAN_ID_COLUMN, how="left", validate="m:1")
    if len(merged) != before:
        raise AssertionError(
            f"static join changed row count: {before} -> {len(merged)}"
        )
    logger.info("Attached %d static columns", len(new_cols) - 1)
    return merged


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical panel ordering: by loan, then chronologically."""
    order = [LOAN_ID_COLUMN]
    if "month_index" in df.columns:
        order.append("month_index")
    elif TIME_COLUMN in df.columns:
        order.append(TIME_COLUMN)
    return df.sort_values(order).reset_index(drop=True)
