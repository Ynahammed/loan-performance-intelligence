"""
Tests for scoring and submission assembly.

The submission is the one deliverable where a silent defect is fatal: a
judge who cannot load the file cannot score anything else. So the
validator is tested against each way it can go wrong, and the scoring
frame is tested for the leakage its design is meant to prevent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import STATE_ORDER
from src.models.predict import build_scoring_frame
from src.submission import (
    ACTION_THRESHOLDS,
    build_submission,
    recommend_action,
    validate_submission,
)


def panel(loans, months, start="2022-01-01", seed=0, **over):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(loans):
        loan = "L{:04d}".format(i)
        rate = float(rng.uniform(3.5, 7.0))
        orig = float(rng.uniform(150_000, 400_000))
        for m in range(months):
            rows.append({
                "loan_id": loan,
                "month_index": m,
                "reporting_month": pd.Timestamp(start) + pd.DateOffset(months=m),
                "origination_month": pd.Timestamp(start),
                "loan_age_months": m,
                "remaining_term_months": 360 - m,
                "interest_rate": rate,
                "original_interest_rate": rate,
                "original_balance": orig,
                "current_balance": orig * (1 - 0.002 * m),
                "current_status": "Current" if m % 5 else "30DPD",
                "days_past_due": 0 if m % 5 else 35,
                "modification_flag": 0,
                "document_status": "Complete",
                "last_updated_at": pd.Timestamp(start) + pd.DateOffset(months=m, days=4),
                "source_system": "LOS_A",
                "state": "CA",
                "credit_score_band": "700-739",
                "ltv_band": "71-80",
                "dti_band": "37-43",
                "servicer_name": "Acme",
                **over,
            })
    return pd.DataFrame(rows)


TEMPLATE = [
    "loan_id", "reporting_month", "next_3m_delinquency_prob",
    "next_6m_delinquency_prob", "next_12m_default_prob",
    "next_12m_prepayment_prob", "next_state_pred", "next_state_confidence",
    "exception_required_pred", "exception_type_pred", "exception_confidence",
    "anomaly_score", "top_drivers", "recommended_action", "model_confidence",
]


# --------------------------------------------------------- scoring frame


def test_scoring_frame_carries_history_across_the_panel_boundary():
    """A loan delinquent through the training window must not be scored as
    though its record were spotless when the test window starts."""
    train = panel(4, 12, start="2022-01-01")
    train["current_status"] = "30DPD"
    train["days_past_due"] = 35
    test = panel(4, 3, start="2023-01-01", seed=1)

    frame = build_scoring_frame(train, test)
    scored = frame.score
    assert (scored["months_delinquent_to_date"] >= 12).all(), (
        "history reset at the panel boundary"
    )


def test_scoring_frame_preserves_input_row_order_and_count():
    train = panel(3, 6)
    test = panel(3, 4, start="2023-01-01", seed=2)
    frame = build_scoring_frame(train, test)
    assert len(frame.score_index) == len(test)
    np.testing.assert_array_equal(
        frame.score["loan_id"].to_numpy(), test["loan_id"].to_numpy()
    )


def test_scoring_frame_keeps_duplicate_rows():
    """The test file ships duplicate (loan_id, month) pairs. Deduplicating
    would change the submission's shape against the input."""
    train = panel(2, 6)
    test = panel(2, 3, start="2023-01-01", seed=3)
    test = pd.concat([test, test.iloc[[0]]], ignore_index=True)
    frame = build_scoring_frame(train, test)
    assert len(frame.score_index) == len(test)
    assert any("duplicate" in n for n in frame.notes)


def test_scoring_frame_reports_loans_without_history():
    train = panel(2, 6)
    test = panel(5, 3, start="2023-01-01", seed=4)
    frame = build_scoring_frame(train, test)
    assert len(frame.loans_without_history) == 3
    assert any("no training history" in n for n in frame.notes)


def test_history_features_do_not_see_the_future_within_the_score_window():
    """Truncating later score months must not change earlier ones."""
    train = panel(3, 8)
    full = panel(3, 6, start="2023-01-01", seed=5)
    early = full[full.month_index < 3].copy()

    f_full = build_scoring_frame(train, full).score
    f_early = build_scoring_frame(train, early).score

    f_full = (f_full[f_full.month_index < 3]
              .sort_values(["loan_id", "month_index"]))
    f_early = f_early.sort_values(["loan_id", "month_index"])
    for col in ("months_delinquent_to_date", "delinquency_streak",
                "n_status_changes_to_date"):
        np.testing.assert_allclose(
            f_full[col].to_numpy(dtype=float),
            f_early[col].to_numpy(dtype=float),
            err_msg="{} changed when later months were removed".format(col),
        )


# ------------------------------------------------------------- actions


def test_data_exception_outranks_every_risk_score():
    """A record that may be wrong should be fixed before it is acted on."""
    row = pd.Series({
        "exception_required_pred": 1,
        "next_12m_default_prob": 0.99,
        "next_3m_delinquency_prob": 0.99,
        "anomaly_score": 0.99,
    })
    assert "data exception" in recommend_action(row)


def test_action_ladder_is_ordered_by_severity():
    base = {"exception_required_pred": 0, "anomaly_score": 0.0,
            "next_12m_default_prob": 0.0, "next_3m_delinquency_prob": 0.0,
            "next_12m_prepayment_prob": 0.0}
    t = ACTION_THRESHOLDS

    assert "NO ACTION" in recommend_action(pd.Series(base))
    assert "default" in recommend_action(pd.Series(
        {**base, "next_12m_default_prob": t["default_review"] + 0.01}))
    assert "delinquency" in recommend_action(pd.Series(
        {**base, "next_3m_delinquency_prob": t["delinquency_review"] + 0.01}))
    assert "MONITOR" in recommend_action(pd.Series(
        {**base, "next_3m_delinquency_prob": t["delinquency_monitor"] + 0.01}))


def test_every_action_is_advisory_never_a_decision():
    """No action value may instruct anything other than human review."""
    cases = [
        {"exception_required_pred": 1},
        {"exception_required_pred": 0, "anomaly_score": 0.95},
        {"exception_required_pred": 0, "next_12m_default_prob": 0.9},
        {"exception_required_pred": 0, "next_3m_delinquency_prob": 0.5},
        {"exception_required_pred": 0, "next_12m_prepayment_prob": 0.9},
        {"exception_required_pred": 0},
    ]
    forbidden = ("deny", "foreclose", "approve", "reject", "terminate",
                 "default the", "close the")
    for case in cases:
        action = recommend_action(pd.Series(case)).lower()
        assert action.startswith(("review", "monitor", "no action"))
        assert not any(word in action for word in forbidden)


# ---------------------------------------------------------- validation


def _good_submission(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "loan_id": ["L{:04d}".format(i) for i in range(n)],
        "reporting_month": ["2024-01-01"] * n,
        "next_3m_delinquency_prob": rng.uniform(0, 1, n),
        "next_6m_delinquency_prob": rng.uniform(0, 1, n),
        "next_12m_default_prob": rng.uniform(0, 1, n),
        "next_12m_prepayment_prob": rng.uniform(0, 1, n),
        "next_state_pred": rng.choice(STATE_ORDER, n),
        "next_state_confidence": rng.uniform(0, 1, n),
        "exception_required_pred": rng.integers(0, 2, n),
        "exception_type_pred": "none",
        "exception_confidence": rng.uniform(0, 1, n),
        "anomaly_score": rng.uniform(0, 1, n),
        "top_drivers": "credit score (+)",
        "recommended_action": "NO ACTION - within normal range",
        "model_confidence": rng.uniform(0, 1, n),
    })[TEMPLATE]


def _reference(n=20):
    return pd.DataFrame({
        "loan_id": ["L{:04d}".format(i) for i in range(n)],
        "reporting_month": pd.Timestamp("2024-01-01"),
    })


def test_valid_submission_passes():
    report = validate_submission(_good_submission(), _reference(), TEMPLATE)
    assert report.ok, report.describe()


def test_column_reordering_is_caught():
    sub = _good_submission()
    sub = sub[[sub.columns[1], sub.columns[0]] + list(sub.columns[2:])]
    report = validate_submission(sub, _reference(), TEMPLATE)
    assert not report.ok
    assert any("column order" in e for e in report.errors)


def test_row_count_mismatch_is_caught():
    report = validate_submission(_good_submission(20), _reference(25), TEMPLATE)
    assert not report.ok
    assert any("row count" in e for e in report.errors)


def test_row_misalignment_is_caught():
    sub = _good_submission()
    sub["loan_id"] = sub["loan_id"].iloc[::-1].values
    report = validate_submission(sub, _reference(), TEMPLATE)
    assert not report.ok
    assert any("order does not match" in e for e in report.errors)


@pytest.mark.parametrize("bad", [1.5, -0.2, np.nan])
def test_out_of_range_probabilities_are_caught(bad):
    sub = _good_submission()
    sub.loc[0, "next_12m_default_prob"] = bad
    report = validate_submission(sub, _reference(), TEMPLATE)
    assert not report.ok


def test_unknown_next_state_is_caught():
    sub = _good_submission()
    sub.loc[0, "next_state_pred"] = "Foreclosed"
    report = validate_submission(sub, _reference(), TEMPLATE)
    assert not report.ok
    assert any("unknown states" in e for e in report.errors)


def test_constant_model_confidence_is_caught():
    """The classic silent placeholder: passes every range check, means
    nothing, and survives to submission."""
    sub = _good_submission()
    sub["model_confidence"] = 0.75
    report = validate_submission(sub, _reference(), TEMPLATE)
    assert not report.ok
    assert any("constant" in e for e in report.errors)


def test_non_binary_exception_flag_is_caught():
    sub = _good_submission()
    sub.loc[0, "exception_required_pred"] = 2
    report = validate_submission(sub, _reference(), TEMPLATE)
    assert not report.ok


# ------------------------------------------------------------ assembly


def test_build_submission_emits_the_template_order():
    n = 12
    idx = pd.RangeIndex(n)
    rng = np.random.default_rng(1)
    keys = pd.DataFrame({
        "loan_id": ["L{}".format(i) for i in range(n)],
        "reporting_month": pd.Timestamp("2024-02-01"),
    }, index=idx)
    probs = pd.DataFrame({
        t: rng.uniform(0, 1, n) for t in
        ("next_3m_delinquency_flag", "next_6m_delinquency_flag",
         "next_12m_default_flag", "next_12m_prepayment_flag")
    }, index=idx)
    next_state = pd.DataFrame({
        "next_state_pred": "Current",
        "next_state_confidence": rng.uniform(0.5, 1, n),
    }, index=idx)
    exceptions = pd.DataFrame({
        "exception_required_pred": 0,
        "exception_type_pred": "none",
        "exception_confidence": rng.uniform(0, 0.5, n),
    }, index=idx)

    out = build_submission(
        keys, probs, next_state, exceptions,
        pd.Series(rng.uniform(0, 1, n), index=idx),
        pd.Series("credit score (+)", index=idx),
        pd.Series(rng.uniform(0, 1, n), index=idx),
        template_columns=TEMPLATE,
    )
    assert list(out.columns) == TEMPLATE
    assert len(out) == n
    assert out["reporting_month"].iloc[0] == "2024-02-01"
    assert out["recommended_action"].notna().all()


def test_missing_target_becomes_null_not_a_plausible_zero():
    """A blank the validator will catch beats a zero that looks real."""
    n = 5
    idx = pd.RangeIndex(n)
    keys = pd.DataFrame({
        "loan_id": ["L{}".format(i) for i in range(n)],
        "reporting_month": pd.Timestamp("2024-02-01"),
    }, index=idx)
    probs = pd.DataFrame({"next_3m_delinquency_flag": 0.1}, index=idx)
    next_state = pd.DataFrame({
        "next_state_pred": "Current", "next_state_confidence": 0.9}, index=idx)
    exceptions = pd.DataFrame({
        "exception_required_pred": 0, "exception_type_pred": "none",
        "exception_confidence": 0.1}, index=idx)

    out = build_submission(
        keys, probs, next_state, exceptions,
        pd.Series(0.5, index=idx), pd.Series("x", index=idx),
        pd.Series(0.5, index=idx), template_columns=TEMPLATE,
    )
    assert out["next_12m_default_prob"].isna().all()
    report = validate_submission(out, keys, TEMPLATE)
    assert not report.ok
