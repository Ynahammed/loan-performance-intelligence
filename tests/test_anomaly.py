"""
Tests for the anomaly detector and the exception layer.

The one that matters most is `test_detector_never_sees_the_target`: the
whole claim that agreement between the unsupervised score and the rules
is evidence rests on the detector being target-blind. If a refactor lets
a target column into the feature set, the component keeps working and
quietly stops meaning anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly.detector import (
    DETECTOR_EXCLUDE,
    anomaly_reasons_frame,
    detect_anomalies,
    detector_features,
    explain_anomaly,
    phrase_reason,
)
from src.anomaly.exception_model import (
    combined_exception_output,
    rule_assigned_type,
    rule_covered_mask,
)
from src.anomaly.rules import (
    evaluate_rules,
    rule_balance_within_original,
)
from src.explainability.labels import describe_exception, describe_rule, label


def make_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "loan_id": ["L{:04d}".format(i) for i in range(n)],
        "reporting_month": pd.Timestamp("2022-01-01"),
        "origination_month": pd.Timestamp("2021-01-01"),
        "current_status": "Current",
        "days_past_due": 0,
        "modification_flag": 0,
        "loan_age_months": rng.integers(1, 60, n),
        "interest_rate": rng.normal(5.0, 0.5, n),
        "original_balance": rng.normal(250_000, 30_000, n),
        "current_balance": rng.normal(240_000, 30_000, n),
        "balance_ratio": rng.normal(0.95, 0.03, n),
        "document_status": "Complete",
        "exception_required": 0,
        "exception_type": "none",
    })


# ------------------------------------------------------- target blindness


def test_detector_never_sees_the_target():
    df = make_frame()
    df["next_12m_default_flag"] = 0
    df["rule_violation_count"] = 0
    df["data_quality_score"] = 100.0
    cols = detector_features(df)
    for banned in DETECTOR_EXCLUDE:
        assert banned not in cols, "{} leaked into the detector".format(banned)
    assert "exception_required" not in cols
    assert "exception_type" not in cols


def test_detector_excludes_rule_outputs():
    """Rule outputs must stay out, or the detector becomes a laundered
    copy of the rule engine rather than an independent signal."""
    df = make_frame()
    df["rule_violation_count"] = 1
    assert "rule_violation_count" not in detector_features(df)


def test_detector_drops_constant_and_non_numeric_columns():
    df = make_frame()
    df["always_seven"] = 7
    cols = detector_features(df)
    assert "always_seven" not in cols
    assert "current_status" not in cols


# ------------------------------------------------------------- scoring


def test_anomaly_scores_are_bounded_and_rank_the_planted_outlier():
    df = make_frame(n=300)
    df.loc[0, "original_balance"] = 50_000_000.0
    df.loc[0, "current_balance"] = 50_000_000.0
    result = detect_anomalies(df)
    assert (result.scores >= 0).all() and (result.scores <= 1).all()
    assert result.scores.loc[0] > result.scores.median()


def test_detector_scores_new_records_against_the_fitted_population():
    """`fit_on` is the deployment shape: the population is the training
    window, later records are judged against it."""
    train = make_frame(n=300, seed=1)
    score = make_frame(n=50, seed=2)
    score.loc[score.index[0], "current_balance"] = 9_000_000.0
    result = detect_anomalies(score, fit_on=train)
    assert len(result.scores) == len(score)
    assert result.scores.iloc[0] == result.scores.max()


# -------------------------------------------------------- explanations


def test_explain_anomaly_ranks_by_deviation_and_respects_top_n():
    df = make_frame(n=500)
    df.loc[0, "original_balance"] = 5_000_000.0
    df.loc[0, "interest_rate"] = 25.0
    result = detect_anomalies(df)
    drivers = explain_anomaly(
        df.loc[0], result.reference, df[result.feature_columns], top_n=2
    )
    assert len(drivers) <= 2
    if len(drivers) == 2:
        assert drivers[0]["robust_deviations"] >= drivers[1]["robust_deviations"]
    assert all(d["direction"] in ("above", "below") for d in drivers)


def test_phrase_reason_returns_a_formatted_sentence():
    """Regression: the percentile branch was split across a return with an
    implicit string concatenation, so it returned an unformatted template
    and the second line was unreachable."""
    entry = {
        "label": "current balance", "direction": "above", "value": 900000.0,
        "population_median": 240000.0, "robust_deviations": 12.3,
        "percentile": 99.8,
    }
    text = phrase_reason(entry)
    assert "{" not in text and "}" not in text
    assert "current balance" in text and "99.8" in text

    del entry["percentile"]
    text2 = phrase_reason(entry)
    assert "{" not in text2 and "12.3" in text2


def test_reasons_frame_handles_records_with_no_extreme_feature():
    df = make_frame(n=300)
    result = detect_anomalies(df)
    frame = anomaly_reasons_frame(df, result, df.index[:5])
    assert len(frame) == 5
    assert frame["drivers"].str.len().gt(0).all()


# ------------------------------------------------------------- labels


def test_label_falls_back_without_raising():
    assert label("balance_ratio") == "share of the original balance still outstanding"
    assert label("a_brand_new_feature") == "a brand new feature"


def test_exception_and_rule_descriptions_are_human():
    assert "modification" in describe_exception("balance_inconsistency")
    assert "_" not in describe_rule("terminal_state_immutability")


# ---------------------------------------------------------- exceptions


def test_balance_within_original_catches_the_first_month_case():
    """The gap that `balance_non_increasing` structurally cannot cover:
    a loan whose very first observed row already exceeds its original
    balance has no previous row to compare against."""
    df = pd.DataFrame({
        "loan_id": ["A", "B"],
        "current_balance": [300_000.0, 200_000.0],
        "original_balance": [250_000.0, 250_000.0],
        "modification_flag": [0, 0],
    })
    assert list(rule_balance_within_original(df)) == [True, False]


def test_balance_within_original_respects_modifications():
    df = pd.DataFrame({
        "loan_id": ["A"],
        "current_balance": [300_000.0],
        "original_balance": [250_000.0],
        "modification_flag": [1],
    })
    assert not rule_balance_within_original(df).any()


def test_rule_covered_mask_ignores_advisory_rules():
    """Advisory rules raise suspicion; they do not resolve a record, so
    they must not count as coverage."""
    flags = pd.DataFrame({
        "balance_non_increasing": [False, False],
        "servicer_source_agreement": [True, False],
        "stale_record": [True, False],
    })
    assert not rule_covered_mask(flags).any()


def test_rule_assigned_type_maps_the_firing_rule():
    flags = pd.DataFrame({
        "balance_non_increasing": [True, False, False],
        "date_order_valid": [False, True, False],
    })
    assert list(rule_assigned_type(flags)) == [
        "balance_inconsistency", "date_invalid", "none"
    ]


def test_combined_output_lets_rules_outrank_model_scores():
    df = make_frame(n=250)
    df.loc[0, "current_balance"] = df.loc[0, "original_balance"] * 2
    result = detect_anomalies(df)
    engine = evaluate_rules(df, rules=[], include_local_rules=True)

    class _Stub:
        residual_model = None
        type_model = None
        feature_cols = []

    out = combined_exception_output(df, engine.flags, result, _Stub())
    covered = rule_covered_mask(engine.flags)
    assert out.loc[covered, "decided_by"].eq("rule").all()
    assert (out.loc[covered, "exception_probability"] > 0.9).all()
    assert out["exception_probability"].between(0, 1).all()


def test_combined_output_does_not_claim_a_model_decided_everything():
    """Labelling every unflagged row 'residual model' reads as though a
    model adjudicated the whole portfolio."""
    df = make_frame(n=200)
    result = detect_anomalies(df)
    engine = evaluate_rules(df, rules=[], include_local_rules=True)

    class _Stub:
        residual_model = None
        type_model = None
        feature_cols = []

    out = combined_exception_output(df, engine.flags, result, _Stub())
    assert "residual model" not in set(out["decided_by"])
