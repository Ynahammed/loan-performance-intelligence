"""
Single-loan what-if simulation.

THE CORRECTNESS PROBLEM
-----------------------
A what-if is only meaningful if the changed input flows through exactly
the pipeline that produced the original score. The tempting shortcut is
to patch the feature vector directly -- set `balance_ratio` and re-score.
That silently breaks the relationships between features: raising
`current_balance` should also move `balance_ratio` AND
`amortisation_gap`, and changing `interest_rate` should move
`rate_incentive`. Patching one and not the others feeds the model a
combination that cannot occur in reality, and the answer it gives back is
about nothing.

So overrides are applied to the RAW row and the real feature engineering
is re-run over the loan's own history. Every derived feature updates
together because it is the same code that computed them the first time.

WHAT DOES NOT MOVE, AND WHY
---------------------------
Data-quality features -- rule violations, source conflict, the record
quality score -- are carried across unchanged. They describe the RECORD,
not the borrower. Asking "what if this borrower's DTI were lower" should
not alter whether the servicer feed disagreed with the panel about last
month's balance. Recomputing them would let a user improve a loan's score
by pretending its data defects away.

History features expand over the loan's own past AND include the row
being scored, so an override does move them -- flipping the scored month
to 90DPD adds one to `months_delinquent_to_date`, correctly, because that
month now is delinquent. What an override cannot do is reach backwards: a
loan that spent six months delinquent still has, because it did. The
regression test asserts exactly that split.

PHASE: 9
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    LOAN_ID_COLUMN, MODELS_DIR, RISK_THRESHOLDS, STATE_COLUMN, STATE_ORDER,
    TIME_COLUMN,
)
from src.features.engineering import engineer_features

logger = logging.getLogger(__name__)

# Fields a reviewer may plausibly ask "what if" about, with the type of
# control each needs. Deliberately not every column: overriding
# `loan_id` or `reporting_month` is meaningless, and overriding a target
# would be nonsense.
OVERRIDABLE_FIELDS = {
    "current_balance": {"kind": "number", "label": "Current balance", "step": 1000.0},
    "interest_rate": {"kind": "number", "label": "Interest rate (%)", "step": 0.125},
    "days_past_due": {"kind": "number", "label": "Days past due", "step": 1.0},
    "remaining_term_months": {"kind": "number", "label": "Remaining term (months)", "step": 1.0},
    "modification_flag": {"kind": "binary", "label": "Modified"},
    "current_status": {"kind": "category", "label": "Payment status",
                       "options": STATE_ORDER},
    "credit_score_band": {"kind": "category", "label": "Credit score band"},
    "ltv_band": {"kind": "category", "label": "LTV band"},
    "dti_band": {"kind": "category", "label": "DTI band"},
}

# Carried across untouched: these describe the record, not the loan.
# Neutral defaults are used when the caller cannot supply the real values
# -- a standalone simulation should still run, and the alternative is
# silently skipping every model that was trained with them. The defaults
# say "a clean record", which is stated in the result rather than assumed.
DATA_QUALITY_FEATURES = {
    "rule_violation_count": 0.0,
    "source_conflict": 0.0,
    "source_stale": 0.0,
    "servicer_update_lag_days": 0.0,
    "data_quality_score": 100.0,
}


@dataclass
class ScenarioResult:
    loan_id: str
    reporting_month: str
    overrides: dict
    baseline: dict
    scenario: dict
    baseline_state: tuple = None
    scenario_state: tuple = None
    changed_features: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def deltas(self) -> pd.DataFrame:
        rows = []
        for target, before in self.baseline.items():
            after = self.scenario.get(target)
            if after is None:
                continue
            rows.append({
                "target": target,
                "baseline": round(before, 5),
                "scenario": round(after, 5),
                "change_pp": round((after - before) * 100, 3),
                "relative_change": (
                    round((after - before) / before, 4) if before else None
                ),
                "baseline_band": risk_band(before),
                "scenario_band": risk_band(after),
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = ["Loan {} as of {}".format(self.loan_id, self.reporting_month)]
        for name, before, after in self.changed_features:
            lines.append("  {}: {} -> {}".format(name, before, after))
        return "\n".join(lines)


def risk_band(probability: float) -> str:
    """Configurable bands, read from config rather than hardcoded here."""
    if probability is None or not np.isfinite(probability):
        return "unknown"
    if probability < RISK_THRESHOLDS["low_max"]:
        return "Low"
    if probability < RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


def load_scoring_models(models_dir: Path = None) -> dict:
    """Load the persisted champions. Missing models are skipped, not fatal."""
    models_dir = Path(models_dir or MODELS_DIR)
    out = {"binary": {}, "transition": None}

    supervised = models_dir / "supervised"
    if supervised.exists():
        for path in sorted(supervised.glob("*.joblib")):
            try:
                bundle = joblib.load(path)
                out["binary"][bundle["target"]] = bundle
            except Exception as exc:
                logger.warning("could not load %s: %s", path.name, exc)

    transition_path = models_dir / "transition_model.joblib"
    if transition_path.exists():
        try:
            out["transition"] = joblib.load(transition_path)
        except Exception as exc:
            logger.warning("could not load transition model: %s", exc)

    return out


def apply_overrides(history: pd.DataFrame, row_index, overrides: dict) -> pd.DataFrame:
    """Patch the RAW row, then re-derive everything from it."""
    patched = history.copy()
    for field_name, value in overrides.items():
        if field_name not in patched.columns:
            logger.warning("override %r is not a column; ignored", field_name)
            continue
        patched.loc[row_index, field_name] = value
    return patched


def _score_binary(models: dict, row: pd.DataFrame) -> dict:
    out = {}
    for target, bundle in models["binary"].items():
        cols = bundle["columns"]
        missing = [c for c in cols if c not in row.columns]
        if missing:
            logger.warning("target %s missing %d columns; skipped",
                           target, len(missing))
            continue
        p = bundle["model"].predict_proba(row[cols])[:, 1]
        calibration = bundle.get("calibration")
        if calibration is not None:
            p = calibration.apply(p)
        out[target] = float(np.clip(p[0], 0.0, 1.0))
    return out


def _score_next_state(models: dict, row: pd.DataFrame) -> tuple:
    bundle = models.get("transition")
    if bundle is None:
        return None
    model = bundle["model"]
    states = bundle.get("states", STATE_ORDER)
    origin = str(row[STATE_COLUMN].iloc[0])
    if origin in model.absorbing:
        return (origin, 1.0)
    if origin not in states:
        return None
    proba = model.row_probabilities(row, origin)[0]
    proba = np.clip(proba, 1e-9, None)
    proba = proba / proba.sum()
    best = int(proba.argmax())
    return (states[best], float(proba[best]))


def simulate(
    history: pd.DataFrame,
    row_index,
    overrides: dict,
    models: dict = None,
    engineered_original: pd.DataFrame = None,
) -> ScenarioResult:
    """Re-score one loan-month under changed inputs.

    `history` is the loan's raw rows, ordered. `row_index` selects the row
    being asked about. `overrides` maps raw column names to new values.
    """
    models = models or load_scoring_models()
    notes = []
    if not models["binary"]:
        notes.append("no persisted supervised models found; run "
                     "`python -m scripts.run_supervised_models` first")

    baseline_history, _ = engineer_features(history)
    patched_history = apply_overrides(history, row_index, overrides)
    scenario_history, _ = engineer_features(patched_history)

    baseline_row = baseline_history.loc[[row_index]]
    scenario_row = scenario_history.loc[[row_index]]

    # Carry the data-quality features across untouched: they describe the
    # record, not the borrower, and recomputing them would let a user
    # improve a score by pretending the record's defects away.
    source = engineered_original
    defaulted = []
    for column, neutral in DATA_QUALITY_FEATURES.items():
        if source is not None and column in source.columns:
            value = source.loc[row_index, column]
        else:
            value = neutral
            defaulted.append(column)
        baseline_row = baseline_row.assign(**{column: value})
        scenario_row = scenario_row.assign(**{column: value})
    if defaulted:
        notes.append(
            "Data-quality features were not supplied and default to a clean "
            "record ({}). Both sides of the comparison use the same values, "
            "so the delta is unaffected; the absolute probabilities assume "
            "the record has no known defects.".format(", ".join(defaulted))
        )

    baseline = _score_binary(models, baseline_row)
    scenario = _score_binary(models, scenario_row)

    changed = []
    for field_name, new_value in overrides.items():
        if field_name not in history.columns:
            continue
        old_value = history.loc[row_index, field_name]
        if pd.isna(old_value) and pd.isna(new_value):
            continue
        if old_value != new_value:
            changed.append((field_name, old_value, new_value))

    # Report which DERIVED features moved as a consequence, so a reviewer
    # can see that the override propagated rather than trusting that it did.
    derived = []
    for column in ("balance_ratio", "amortisation_gap", "term_progress",
                   "rate_incentive", "rate_incentive_pctile",
                   "credit_score_band_ordinal", "ltv_band_ordinal",
                   "dti_band_ordinal"):
        if column not in baseline_row.columns:
            continue
        before = baseline_row[column].iloc[0]
        after = scenario_row[column].iloc[0]
        if pd.isna(before) and pd.isna(after):
            continue
        if not np.isclose(float(before or 0), float(after or 0), equal_nan=True):
            derived.append((column + " (derived)", round(float(before), 5),
                            round(float(after), 5)))
    changed.extend(derived)

    if not changed:
        notes.append("no input changed; baseline and scenario are identical")

    return ScenarioResult(
        loan_id=str(history.loc[row_index, LOAN_ID_COLUMN]),
        reporting_month=str(pd.Timestamp(history.loc[row_index, TIME_COLUMN]).date()),
        overrides=overrides,
        baseline=baseline,
        scenario=scenario,
        baseline_state=_score_next_state(models, baseline_row),
        scenario_state=_score_next_state(models, scenario_row),
        changed_features=changed,
        notes=notes,
    )


def field_options(panel: pd.DataFrame, field_name: str) -> list:
    """Observed values for a categorical override, so the UI cannot offer a
    level the encoder has never seen."""
    spec = OVERRIDABLE_FIELDS.get(field_name, {})
    if "options" in spec:
        return list(spec["options"])
    if field_name in panel.columns:
        return sorted(str(v) for v in panel[field_name].dropna().unique())
    return []
