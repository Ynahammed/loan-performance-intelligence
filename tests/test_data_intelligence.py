"""
Tests for the data intelligence layer: rules, reconciliation, profiling,
drift.

As with the survival tests, these pin failure modes that actually
occurred. The most important is `test_terminal_rule_does_not_leak_across_loans`:
the terminal-state rule originally shifted across loan boundaries instead
of within each loan, giving 482 flags for 6 real reversals. Precision
against the labels caught it; this test keeps it caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly.rules import (
    RULE_IMPLEMENTATIONS,
    evaluate_rules,
    load_rules,
    rule_balance_non_increasing,
    rule_date_order_valid,
    rule_document_completeness_at_closure,
    rule_dpd_status_consistency,
    rule_stale_record,
    rule_terminal_state_immutability,
)
from src.data.drift import adversarial_validation, compute_psi, drift_report
from src.data.profiler import (
    cramers_v,
    cross_column_relationship_breaks,
    find_inconsistent_categories,
    record_quality_scores,
)
from src.data.reconciliation import reconcile


def panel_row(loan, month, **over):
    row = {
        "loan_id": loan,
        "month_index": month,
        "reporting_month": pd.Timestamp("2022-01-01") + pd.DateOffset(months=month),
        "origination_month": pd.Timestamp("2022-01-01"),
        "current_status": "Current",
        "days_past_due": 0,
        "current_balance": 100_000.0 - month * 100,
        "original_balance": 100_000.0,
        "modification_flag": 0,
        "document_status": "Complete",
        "last_updated_at": pd.Timestamp("2022-01-05") + pd.DateOffset(months=month),
        "source_system": "LOS_A",
    }
    row.update(over)
    return row


def make_panel(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------- rules


def test_balance_rule_flags_increase_without_modification():
    df = make_panel([
        panel_row("A", 0),
        panel_row("A", 1, current_balance=150_000.0),   # rose, no mod -> flag
        panel_row("A", 2, current_balance=160_000.0, modification_flag=1),  # ok
    ])
    flags = rule_balance_non_increasing(df)
    assert list(flags) == [False, True, False]


def test_balance_rule_does_not_leak_across_loans():
    """Loan B's first row must not be compared against loan A's last."""
    df = make_panel([
        panel_row("A", 0, current_balance=10_000.0),
        panel_row("B", 0, current_balance=900_000.0),
    ])
    assert not rule_balance_non_increasing(df).any()


def test_terminal_rule_does_not_leak_across_loans():
    """Regression for the 482-flags-for-6-reversals bug.

    Loan A ends Prepaid. Loan B starts Current. Without grouping the shift
    as well as the cumsum, B's first row inherits A's terminal state and is
    flagged as a resurrection.
    """
    df = make_panel([
        panel_row("A", 0),
        panel_row("A", 1, current_status="Prepaid"),
        panel_row("B", 0),
        panel_row("B", 1),
    ])
    flags = rule_terminal_state_immutability(df)
    assert not flags.any(), "flagged a clean loan following a terminal one"


def test_terminal_rule_flags_a_real_resurrection():
    df = make_panel([
        panel_row("A", 0),
        panel_row("A", 1, current_status="Prepaid"),
        panel_row("A", 2, current_status="Current"),   # impossible
    ])
    flags = rule_terminal_state_immutability(df)
    assert list(flags) == [False, False, True]


def test_date_rule_flags_reporting_before_origination():
    df = make_panel([
        panel_row("A", 0),
        panel_row("A", 1, reporting_month=pd.Timestamp("2021-06-01")),
    ])
    assert list(rule_date_order_valid(df)) == [False, True]


def test_dpd_status_consistency():
    df = make_panel([
        panel_row("A", 0),                                        # Current, 0
        panel_row("A", 1, current_status="Current", days_past_due=45),
        panel_row("A", 2, current_status="30DPD", days_past_due=35),
        panel_row("A", 3, current_status="30DPD", days_past_due=120),
    ])
    assert list(rule_dpd_status_consistency(df)) == [False, True, False, True]


def test_document_rule_treats_pending_as_acceptable():
    """Pending is a workflow state; Missing is a control gap. Conflating
    them drops precision from 83% to 35% on the real data."""
    df = make_panel([
        panel_row("A", 0, current_status="Prepaid", document_status="Pending"),
        panel_row("B", 0, current_status="Prepaid", document_status="Missing"),
        panel_row("C", 0, current_status="Current", document_status="Missing"),
    ])
    assert list(rule_document_completeness_at_closure(df)) == [False, True, False]


def test_stale_rule_flags_negative_lag():
    """An update stamped before the month it reports on is impossible,
    not merely late."""
    df = make_panel([
        panel_row("A", 0),
        panel_row("A", 1, last_updated_at=pd.Timestamp("2021-01-01")),
    ])
    assert list(rule_stale_record(df)) == [False, True]


def test_declared_rules_all_have_implementations():
    """A declared rule with no implementation must be reported, never
    silently treated as passing."""
    declared = {r["rule_id"] for r in load_rules()}
    missing = declared - set(RULE_IMPLEMENTATIONS)
    assert not missing, "unimplemented rules: {}".format(missing)


def test_engine_reports_unimplemented_rules_rather_than_passing_them():
    df = make_panel([panel_row("A", 0)])
    result = evaluate_rules(
        df,
        rules=[{"rule_id": "not_a_real_rule", "severity": "high", "description": ""}],
        include_local_rules=False,
    )
    assert result.unimplemented == ["not_a_real_rule"]
    assert not result.catalog.iloc[0]["implemented"]


def test_benchmark_precision_and_recall_math():
    df = make_panel([
        panel_row("A", 0),
        panel_row("A", 1, reporting_month=pd.Timestamp("2020-01-01")),  # date_invalid
        panel_row("B", 0),
    ])
    result = evaluate_rules(df, include_local_rules=False)
    bench = result.benchmark_against_labels(
        pd.Series([0, 1, 0], index=df.index),
        pd.Series(["none", "date_invalid", "none"], index=df.index),
    )
    row = bench[bench.rule_id == "date_order_valid"].iloc[0]
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0


def test_severity_score_weights_high_above_medium():
    df = make_panel([
        panel_row("A", 0, reporting_month=pd.Timestamp("2020-01-01")),  # high
        panel_row("B", 0, current_status="Prepaid", document_status="Missing"),  # medium
    ])
    result = evaluate_rules(df, include_local_rules=False)
    score = result.severity_score()
    assert score.iloc[0] > score.iloc[1]


# ------------------------------------------------------- reconciliation


def test_reconcile_detects_conflict_staleness_and_orphans():
    panel = make_panel([panel_row("A", 0), panel_row("B", 0)])
    servicer = pd.DataFrame([
        {"loan_id": "A", "reporting_month": panel.reporting_month.iloc[0],
         "current_status": "Current", "current_balance": 500_000.0,
         "days_past_due": 0, "last_updated_at": pd.Timestamp("2022-01-05")},
        {"loan_id": "ZZ", "reporting_month": pd.Timestamp("2022-01-01"),
         "current_status": "Current", "current_balance": 1.0,
         "days_past_due": 0, "last_updated_at": pd.Timestamp("2030-01-01")},
    ])
    result = reconcile(panel, servicer)
    assert result.n_orphan_servicer_rows == 1
    assert bool(result.record_flags["balance_conflict"].iloc[0])
    assert not bool(result.record_flags["balance_conflict"].iloc[1])
    assert not bool(result.record_flags["has_servicer_record"].iloc[1])


def test_shared_nulls_are_not_a_conflict():
    """NaN != NaN in pandas; a gap present in both sources is agreement
    about ignorance, not disagreement."""
    panel = make_panel([panel_row("A", 0, days_past_due=np.nan)])
    servicer = pd.DataFrame([{
        "loan_id": "A", "reporting_month": panel.reporting_month.iloc[0],
        "current_status": "Current", "current_balance": panel.current_balance.iloc[0],
        "days_past_due": np.nan, "last_updated_at": pd.Timestamp("2022-01-05"),
    }])
    result = reconcile(panel, servicer)
    assert not bool(result.record_flags["days_past_due_conflict"].iloc[0])


# ------------------------------------------------------------ profiling


def test_cramers_v_bounds():
    a = pd.Series(list("aabbcc") * 20)
    assert cramers_v(a, a) == pytest.approx(1.0, abs=0.05)
    rng = np.random.default_rng(0)
    b = pd.Series(rng.choice(list("xyz"), size=len(a)))
    assert 0.0 <= cramers_v(a, b) < 0.4


def test_inconsistent_categories_merges_case_and_known_aliases():
    df = pd.DataFrame({"state": ["CA", "ca", "California", "NY", "TX"]})
    out = find_inconsistent_categories(df)
    row = out[out.canonical == "ca"].iloc[0]
    assert row["n_variants"] == 3
    assert row["n_rows"] == 3


def test_inconsistent_categories_does_not_merge_distinct_values():
    df = pd.DataFrame({"servicer_name": ["Acme Servicing", "Apex Servicing"]})
    assert len(find_inconsistent_categories(df)) == 0


def test_cross_column_checks_catch_balance_exceeding_original():
    df = make_panel([
        panel_row("A", 0),
        panel_row("B", 0, current_balance=200_000.0),
    ])
    checks = cross_column_relationship_breaks(df)
    row = checks[checks.check == "balance_within_original"].iloc[0]
    assert row["n_violations"] == 1


def test_quality_score_is_bounded_and_penalises_violations():
    df = make_panel([panel_row("A", 0), panel_row("B", 0)])
    severity = pd.Series([0.0, 3.0], index=df.index)
    out = record_quality_scores(df, severity)
    assert (out.data_quality_score >= 0).all() and (out.data_quality_score <= 100).all()
    assert out.data_quality_score.iloc[0] > out.data_quality_score.iloc[1]


# ---------------------------------------------------------------- drift


def test_psi_is_zero_for_identical_samples():
    s = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert compute_psi(s, s.copy()) == pytest.approx(0.0, abs=1e-6)


def test_psi_rises_with_a_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    small = pd.Series(rng.normal(0.2, 1, 5000))
    large = pd.Series(rng.normal(2.0, 1, 5000))
    assert compute_psi(a, small) < compute_psi(a, large)
    assert compute_psi(a, large) > 0.25


def test_psi_handles_categoricals_and_unseen_levels():
    a = pd.Series(["x"] * 90 + ["y"] * 10)
    b = pd.Series(["x"] * 50 + ["z"] * 50)
    assert compute_psi(a, b) > 0.25


def test_drift_report_skips_datetime_columns():
    df = make_panel([panel_row("A", i) for i in range(20)])
    out = drift_report(df, df.copy())
    assert "reporting_month" not in set(out.column)


def test_adversarial_validation_excludes_time_index_columns():
    """The function must not report near-perfect separation just because
    the split was chronological -- that is a restatement, not a finding."""
    a = make_panel([panel_row("A", i) for i in range(60)])
    b = make_panel([panel_row("B", i) for i in range(60, 120)])
    out = adversarial_validation(a, b, exclude_time_like=True)
    assert "month_index" not in set(out["top_features"].column)
    assert "loan_age_months" not in set(out["top_features"].column)
