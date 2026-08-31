"""
Tests for the transition / survival engine.

These are regression tests for failure modes that actually occurred while
building it, not coverage theatre. Each one corresponds to a bug that was
either hit or is silent enough to be worth pinning:

  - log_loss column ordering (hit: produced a metric ~60x too large that
    still looked like a plausible "bad model" number)
  - lambda estimator factory breaking joblib persistence (hit)
  - shipped next_state being trusted on rows with no successor (avoided by
    design; pinned here so a refactor cannot reintroduce it)
  - scenario stress silently renormalising itself away to nothing
  - absorbing states leaking probability mass back out
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.survival import (
    DiscreteTimeTransitionModel,
    EmpiricalTransitionBaseline,
    apply_scenario_to_matrix,
    build_transition_frame,
    multiclass_log_loss,
    project_balance_ratio,
    simulate_paths,
)
from src.models.validation import time_aware_split

STATES = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid"]
ABSORBING = ["Default", "Prepaid"]


def make_panel(n_loans=60, n_months=24, seed=0):
    """Small synthetic panel with the same shape as the real one."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_loans):
        loan = "L{:04d}".format(i)
        state = "Current"
        rate = float(rng.uniform(3.0, 7.0))
        orig = float(rng.uniform(100_000, 500_000))
        for m in range(n_months):
            rows.append(
                {
                    "loan_id": loan,
                    "month_index": m,
                    "reporting_month": pd.Timestamp("2020-01-01")
                    + pd.DateOffset(months=m),
                    "current_status": state,
                    "loan_age_months": m,
                    "remaining_term_months": 360 - m,
                    "interest_rate": rate,
                    "original_balance": orig,
                    "current_balance": orig * (1 - 0.002 * m),
                    "balance_ratio": 1 - 0.002 * m,
                    "modification_flag": 0,
                    "credit_score_band": rng.choice(["<620", "660-699", "740-779"]),
                    "next_state": "Current",  # deliberately wrong on purpose
                }
            )
            if state == "Current":
                state = rng.choice(["Current", "30DPD"], p=[0.96, 0.04])
            elif state == "30DPD":
                state = rng.choice(["Current", "30DPD", "60DPD"], p=[0.5, 0.3, 0.2])
            elif state == "60DPD":
                state = rng.choice(["60DPD", "90DPD"], p=[0.6, 0.4])
            elif state == "90DPD":
                state = rng.choice(["90DPD", "Default"], p=[0.7, 0.3])
    return pd.DataFrame(rows)


# --------------------------------------------------------------- frame


def test_transition_frame_drops_unobservable_last_rows():
    panel = make_panel(n_loans=10, n_months=6)
    tf = build_transition_frame(panel)
    # One row per loan has no successor observation.
    assert tf.n_terminal_rows_dropped == 10
    assert len(tf.data) == tf.n_input_rows - 10


def test_transition_frame_ignores_shipped_next_state():
    """The fixture ships next_state='Current' everywhere, which is wrong.

    Transitions must come from the observed lead of current_status, so the
    derived to_state must disagree with the shipped column wherever the
    loan actually moved.
    """
    panel = make_panel(n_loans=40, n_months=24, seed=3)
    tf = build_transition_frame(panel)
    assert (tf.data["to_state"] != "Current").any(), "fixture produced no movement"
    assert tf.label_disagreements > 0
    # And the derived column, not the shipped one, is what got used.
    assert "to_state" in tf.data.columns


def test_transition_frame_drops_month_gaps():
    panel = make_panel(n_loans=5, n_months=8)
    # Punch a hole: remove month 3 for one loan, so 2 -> 4 is a two-month jump.
    holed = panel[~((panel.loan_id == "L0000") & (panel.month_index == 3))]
    tf = build_transition_frame(holed)
    assert tf.n_gap_rows_dropped == 1


# --------------------------------------------------------------- matrices


@pytest.mark.parametrize("model_cls", [EmpiricalTransitionBaseline, DiscreteTimeTransitionModel])
def test_transition_matrices_are_stochastic(model_cls):
    panel = make_panel(seed=1)
    tf = build_transition_frame(panel)
    model = model_cls().fit(tf.data)
    M = model.full_matrices(tf.data.head(50))
    assert M.shape == (50, len(STATES), len(STATES))
    assert (M >= 0).all()
    np.testing.assert_allclose(M.sum(axis=2), 1.0, atol=1e-9)


def test_absorbing_states_are_identity_rows():
    panel = make_panel(seed=2)
    tf = build_transition_frame(panel)
    model = DiscreteTimeTransitionModel().fit(tf.data)
    M = model.full_matrices(tf.data.head(20))
    for s in ABSORBING:
        i = STATES.index(s)
        assert (M[:, i, i] == 1.0).all()
        off = np.delete(M[:, i, :], i, axis=1)
        assert (off == 0.0).all(), "{} leaked probability mass".format(s)


# --------------------------------------------------------------- metrics


def test_multiclass_log_loss_handles_non_lexicographic_states():
    """The bug this pins: sklearn orders `labels` lexicographically, so
    columns in ladder order get compared against the wrong classes. A
    perfectly confident, perfectly correct prediction must score ~0."""
    y = np.array(["Current", "30DPD", "Prepaid", "Current"])
    proba = np.zeros((4, len(STATES)))
    for row, label in enumerate(y):
        proba[row, STATES.index(label)] = 1.0
    assert multiclass_log_loss(y, proba, STATES) < 1e-6


def test_multiclass_log_loss_penalises_wrong_confident_predictions():
    y = np.array(["Current", "Current"])
    proba = np.zeros((2, len(STATES)))
    proba[:, STATES.index("Default")] = 1.0
    assert multiclass_log_loss(y, proba, STATES) > 10


# --------------------------------------------------------------- scenarios


def _uniform_matrices(n=5):
    M = np.zeros((n, len(STATES), len(STATES)))
    for i, s in enumerate(STATES):
        if s in ABSORBING:
            M[:, i, i] = 1.0
        else:
            M[:, i, :] = 1.0 / len(STATES)
    return M


def test_neutral_scenario_is_identity():
    M = _uniform_matrices()
    out = apply_scenario_to_matrix(
        M, STATES, ABSORBING,
        {"default_multiplier": 1.0, "delinquency_multiplier": 1.0,
         "prepayment_multiplier": 1.0},
    )
    np.testing.assert_array_equal(M, out)


def test_adverse_scenario_actually_moves_mass():
    """Guards the subtle failure where a stress is applied and then
    renormalised straight back out, leaving the matrix unchanged."""
    M = _uniform_matrices()
    out = apply_scenario_to_matrix(
        M, STATES, ABSORBING,
        {"default_multiplier": 1.85, "delinquency_multiplier": 1.55,
         "prepayment_multiplier": 0.55},
    )
    i90 = STATES.index("90DPD")
    idef = STATES.index("Default")
    assert out[0, i90, idef] > M[0, i90, idef]

    icur = STATES.index("Current")
    iprep = STATES.index("Prepaid")
    assert out[0, icur, iprep] < M[0, icur, iprep]

    np.testing.assert_allclose(out.sum(axis=2), 1.0, atol=1e-9)


def test_scenario_output_stays_a_valid_matrix_under_extreme_multipliers():
    M = _uniform_matrices()
    out = apply_scenario_to_matrix(
        M, STATES, ABSORBING,
        {"default_multiplier": 50.0, "delinquency_multiplier": 50.0,
         "prepayment_multiplier": 0.0},
    )
    assert (out >= 0).all()
    np.testing.assert_allclose(out.sum(axis=2), 1.0, atol=1e-9)


# --------------------------------------------------------------- simulation


def test_cumulative_incidence_is_monotone_and_bounded():
    panel = make_panel(seed=4)
    tf = build_transition_frame(panel)
    model = DiscreteTimeTransitionModel().fit(tf.data)
    start = panel.groupby("loan_id").tail(1).reset_index(drop=True)
    active = start[~start.current_status.isin(ABSORBING)].reset_index(drop=True)

    sim = simulate_paths(model, active, active.current_status, horizon=12)
    for state in ABSORBING:
        cif = sim.cumulative_incidence(state)
        assert cif.shape == (len(active), 13)
        assert (np.diff(cif, axis=1) >= -1e-12).all(), "CIF must not decrease"
        assert (cif >= 0).all() and (cif <= 1).all()

    dist = sim.distribution
    np.testing.assert_allclose(dist.sum(axis=2), 1.0, atol=1e-9)


def test_competing_risks_cifs_sum_to_at_most_one():
    """The whole point of one shared chain: default and prepayment compete,
    so their cumulative incidences cannot jointly exceed 1."""
    panel = make_panel(seed=5)
    tf = build_transition_frame(panel)
    model = DiscreteTimeTransitionModel().fit(tf.data)
    start = panel.groupby("loan_id").tail(1).reset_index(drop=True)
    active = start[~start.current_status.isin(ABSORBING)].reset_index(drop=True)
    sim = simulate_paths(model, active, active.current_status, horizon=24)
    total = sim.cumulative_incidence("Default") + sim.cumulative_incidence("Prepaid")
    assert (total <= 1.0 + 1e-9).all()


def test_adverse_scenario_raises_projected_defaults():
    panel = make_panel(seed=6)
    tf = build_transition_frame(panel)
    model = DiscreteTimeTransitionModel().fit(tf.data)
    start = panel.groupby("loan_id").tail(1).reset_index(drop=True)
    active = start[~start.current_status.isin(ABSORBING)].reset_index(drop=True)

    base = simulate_paths(model, active, active.current_status, horizon=12)
    adverse = simulate_paths(
        model, active, active.current_status, horizon=12,
        scenario_multipliers={"default_multiplier": 1.85,
                              "delinquency_multiplier": 1.55,
                              "prepayment_multiplier": 0.55},
        scenario_name="adverse_credit",
    )
    assert adverse.portfolio_curve("Default")[-1] > base.portfolio_curve("Default")[-1]
    assert adverse.portfolio_curve("Prepaid")[-1] < base.portfolio_curve("Prepaid")[-1]


# --------------------------------------------------------------- covariates


def test_balance_ratio_projection_amortises_downward():
    X = pd.DataFrame(
        {
            "balance_ratio": [1.0, 0.8],
            "interest_rate": [5.0, 6.0],
            "loan_age_months": [12, 60],
            "remaining_term_months": [348, 300],
        }
    )
    projected = project_balance_ratio(X, 12)
    assert (projected < X["balance_ratio"]).all()
    assert (projected > 0).all()


def test_model_is_picklable():
    """Regression: a lambda estimator factory made the fitted model
    unpicklable, which only surfaced at the persistence step."""
    import pickle

    panel = make_panel(seed=7)
    tf = build_transition_frame(panel)
    model = DiscreteTimeTransitionModel().fit(tf.data)
    restored = pickle.loads(pickle.dumps(model))
    a = model.row_probabilities(tf.data.head(5), "Current")
    b = restored.row_probabilities(tf.data.head(5), "Current")
    np.testing.assert_allclose(a, b)


# --------------------------------------------------------------- validation


def test_purge_gap_removes_boundary_months():
    panel = make_panel(n_loans=20, n_months=36)
    no_purge = time_aware_split(panel, purge_months=0)
    purged = time_aware_split(panel, purge_months=6)
    assert purged.n_purged > no_purge.n_purged
    assert len(purged.train) + len(purged.test) < len(panel)
    gap = (purged.test_start.to_period("M") - purged.train_end.to_period("M")).n
    assert gap == 7  # 6 purged months plus the step to the next month


def test_split_never_puts_a_later_train_row_after_a_test_row():
    panel = make_panel(n_loans=20, n_months=36)
    split = time_aware_split(panel, purge_months=3)
    assert split.train.reporting_month.max() < split.test.reporting_month.min()
