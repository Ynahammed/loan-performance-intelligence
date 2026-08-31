"""
Tests for single-loan what-if simulation.

The one that matters is `test_override_propagates_to_derived_features`:
a what-if that patches one feature and leaves its dependents stale feeds
the model a combination that cannot occur, and returns a confident answer
about nothing. Nothing about the output would look wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import engineer_features
from src.scenarios.simulator import (
    DATA_QUALITY_FEATURES,
    OVERRIDABLE_FIELDS,
    ScenarioResult,
    apply_overrides,
    field_options,
    risk_band,
    simulate,
)


def make_history(months=18, loan="LN0001", seed=0):
    rng = np.random.default_rng(seed)
    rate = 5.25
    orig = 300_000.0
    rows = []
    for m in range(months):
        rows.append({
            "loan_id": loan,
            "month_index": m,
            "reporting_month": pd.Timestamp("2022-01-01") + pd.DateOffset(months=m),
            "origination_month": pd.Timestamp("2022-01-01"),
            "loan_age_months": m,
            "remaining_term_months": 360 - m,
            "interest_rate": rate,
            "original_interest_rate": rate,
            "original_balance": orig,
            "current_balance": orig * (1 - 0.002 * m),
            "current_status": "Current" if m % 6 else "30DPD",
            "days_past_due": 0 if m % 6 else 30,
            "modification_flag": 0,
            "document_status": "Complete",
            "last_updated_at": pd.Timestamp("2022-01-05") + pd.DateOffset(months=m),
            "source_system": "LOS_A",
            "state": "CA",
            "credit_score_band": "700-739",
            "ltv_band": "71-80",
            "dti_band": "37-43",
            "servicer_name": "Acme",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------- propagation


def test_override_propagates_to_derived_features():
    """Raising the balance must move balance_ratio AND amortisation_gap.

    Patching a feature vector directly would move neither, or only one,
    and the model would be scored on a state that cannot exist.
    """
    history = make_history()
    idx = history.index[-1]

    before, _ = engineer_features(history)
    patched = apply_overrides(history, idx, {"current_balance": 500_000.0})
    after, _ = engineer_features(patched)

    assert after.loc[idx, "balance_ratio"] > before.loc[idx, "balance_ratio"]
    assert after.loc[idx, "amortisation_gap"] > before.loc[idx, "amortisation_gap"]


def test_rate_override_moves_the_rate_incentive():
    history = make_history()
    idx = history.index[-1]
    before, _ = engineer_features(history)
    after, _ = engineer_features(
        apply_overrides(history, idx, {"interest_rate": 9.5}))
    assert after.loc[idx, "rate_incentive"] > before.loc[idx, "rate_incentive"]


def test_override_does_not_touch_other_rows():
    """A what-if about one month must not rewrite the loan's history."""
    history = make_history()
    idx = history.index[5]
    patched = apply_overrides(history, idx, {"current_balance": 1.0})
    untouched = history.drop(index=idx)
    pd.testing.assert_frame_equal(patched.drop(index=idx), untouched)


def test_override_cannot_rewrite_prior_months():
    """History features are expanding sums that INCLUDE the scored row, so
    flipping that row to 90DPD legitimately adds one to the delinquency
    count. What must not happen is the override reaching backwards: every
    earlier row's history must be identical.

    The first version of this test asserted the scored row was unchanged
    too, which was simply wrong about the feature's definition.
    """
    history = make_history()
    idx = history.index[-1]
    before, _ = engineer_features(history)
    after, _ = engineer_features(
        apply_overrides(history, idx, {"current_status": "90DPD",
                                       "days_past_due": 95}))

    earlier = history.index[:-1]
    for column in ("months_delinquent_to_date", "n_status_changes_to_date",
                   "max_dpd_to_date", "delinquency_streak"):
        pd.testing.assert_series_equal(
            before.loc[earlier, column], after.loc[earlier, column],
            check_names=False,
        )

    # The scored row was Current and is now 90DPD, so it now counts once.
    assert (after.loc[idx, "months_delinquent_to_date"]
            == before.loc[idx, "months_delinquent_to_date"] + 1)


def test_unknown_override_field_is_ignored_not_fatal():
    history = make_history()
    idx = history.index[-1]
    patched = apply_overrides(history, idx, {"not_a_column": 1})
    pd.testing.assert_frame_equal(patched, history)


# ------------------------------------------------------------- bands


@pytest.mark.parametrize("probability,expected", [
    (0.0, "Low"), (0.29, "Low"), (0.31, "Medium"),
    (0.59, "Medium"), (0.61, "High"), (1.0, "High"),
])
def test_risk_bands_follow_the_configured_thresholds(probability, expected):
    assert risk_band(probability) == expected


def test_risk_band_handles_missing_values():
    assert risk_band(None) == "unknown"
    assert risk_band(float("nan")) == "unknown"


# ----------------------------------------------------------- results


def test_deltas_report_direction_and_band_change():
    result = ScenarioResult(
        loan_id="L1", reporting_month="2023-01-01", overrides={},
        baseline={"next_3m_delinquency_flag": 0.20},
        scenario={"next_3m_delinquency_flag": 0.65},
    )
    row = result.deltas().iloc[0]
    assert row["change_pp"] == pytest.approx(45.0)
    assert row["baseline_band"] == "Low"
    assert row["scenario_band"] == "High"


def test_deltas_survive_a_zero_baseline():
    """Relative change is undefined against zero; it must not raise."""
    result = ScenarioResult(
        loan_id="L1", reporting_month="2023-01-01", overrides={},
        baseline={"t": 0.0}, scenario={"t": 0.1},
    )
    row = result.deltas().iloc[0]
    assert row["relative_change"] is None


# --------------------------------------------------------- integration


def test_simulate_runs_without_persisted_models_and_says_so():
    """A missing model directory must produce a clear note rather than a
    stack trace, so the dashboard can tell the user what to run."""
    history = make_history()
    idx = history.index[-1]
    result = simulate(history, idx, {"days_past_due": 45},
                      models={"binary": {}, "transition": None})
    assert result.baseline == {}
    assert any("no persisted supervised models" in n for n in result.notes)


def test_simulate_records_what_changed():
    history = make_history()
    idx = history.index[-1]
    result = simulate(history, idx, {"current_balance": 250_000.0},
                      models={"binary": {}, "transition": None})
    names = [c[0] for c in result.changed_features]
    assert "current_balance" in names
    # And the derived consequences are reported, not just the raw edit.
    assert any("derived" in n for n in names)


def test_simulate_notes_an_empty_override():
    history = make_history()
    idx = history.index[-1]
    current = history.loc[idx, "days_past_due"]
    result = simulate(history, idx, {"days_past_due": current},
                      models={"binary": {}, "transition": None})
    assert any("no input changed" in n for n in result.notes)


def test_data_quality_features_default_to_a_clean_record_with_a_note():
    """They describe the record, not the borrower. Defaulting silently
    would let a user improve a score by pretending defects away."""
    history = make_history()
    idx = history.index[-1]
    result = simulate(history, idx, {"days_past_due": 60},
                      models={"binary": {}, "transition": None})
    assert any("default to a clean record" in n for n in result.notes)
    assert set(DATA_QUALITY_FEATURES) >= {"rule_violation_count",
                                          "data_quality_score"}


def test_field_options_come_from_observed_values():
    panel = make_history()
    options = field_options(panel, "credit_score_band")
    assert options == ["700-739"]
    # Declared options win where the field has a fixed vocabulary.
    assert "Default" in field_options(panel, "current_status")


def test_every_overridable_field_declares_a_control_kind():
    for name, spec in OVERRIDABLE_FIELDS.items():
        assert spec["kind"] in ("number", "binary", "category"), name
        assert spec.get("label"), name
