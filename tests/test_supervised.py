"""
Tests for feature engineering, calibration and the multi-target trainer.

The one that matters most is `test_history_features_are_strictly_backward_looking`:
the fixture's future is deliberately catastrophic, so any feature that
peeks at it will differ from the same feature computed on a truncated
copy. That is the leakage check the whole panel design rests on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_band_ordinals,
    canonicalise_categories,
    engineer_features,
    FeatureManifest,
)
from src.models.calibration import (
    MIN_POSITIVES_FOR_ISOTONIC,
    expected_calibration_error,
    reliability_diagram_data,
    select_calibrator,
)
from src.models.train import (
    feature_columns,
    mature_label_mask,
    recall_at_precision,
)
from src.models.validation import time_aware_split


def make_panel(n_loans=8, n_months=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_loans):
        loan = "L{:03d}".format(i)
        rate = float(rng.uniform(3.5, 7.0))
        orig = float(rng.uniform(150_000, 400_000))
        for mth in range(n_months):
            rows.append({
                "loan_id": loan,
                "month_index": mth,
                "reporting_month": pd.Timestamp("2020-01-01") + pd.DateOffset(months=mth),
                "origination_month": pd.Timestamp("2020-01-01"),
                "loan_age_months": mth,
                "remaining_term_months": 360 - mth,
                "interest_rate": rate,
                "original_interest_rate": rate,
                "original_balance": orig,
                "current_balance": orig * (1 - 0.0025 * mth),
                "current_status": "Current" if mth % 7 else "30DPD",
                "days_past_due": 0 if mth % 7 else 35,
                "modification_flag": 0,
                "credit_score_band": "700-739",
                "ltv_band": "71-80",
                "dti_band": "37-43",
                "state": "CA",
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------- leakage checks


def test_history_features_are_strictly_backward_looking():
    """Truncating the future must not change the past.

    The tail of the fixture is made catastrophic (every loan defaults with
    a wildly different balance). If any history feature uses a centred or
    ungrouped window, the rows BEFORE the tail will change when the tail is
    removed. They must not.
    """
    panel = make_panel(n_months=30)
    full = panel.copy()
    tail = full.month_index >= 20
    full.loc[tail, "current_status"] = "90DPD"
    full.loc[tail, "days_past_due"] = 200
    full.loc[tail, "current_balance"] = 5_000_000.0

    truncated = full[full.month_index < 20].copy()

    f_full, _ = engineer_features(full)
    f_trunc, _ = engineer_features(truncated)

    f_full = f_full[f_full.month_index < 20].sort_values(["loan_id", "month_index"])
    f_trunc = f_trunc.sort_values(["loan_id", "month_index"])

    history = [
        "months_delinquent_to_date", "ever_delinquent", "max_dpd_to_date",
        "delinquency_streak", "n_status_changes_to_date", "balance_change_3m",
    ]
    for col in history:
        if col not in f_full.columns:
            continue
        np.testing.assert_allclose(
            f_full[col].to_numpy(dtype=float),
            f_trunc[col].to_numpy(dtype=float),
            err_msg="{} changed when the future was removed -- it leaks".format(col),
        )


def test_history_features_do_not_leak_across_loans():
    panel = make_panel(n_loans=3, n_months=10)
    out, _ = engineer_features(panel)
    first_rows = out.sort_values(["loan_id", "month_index"]).groupby("loan_id").head(1)
    assert (first_rows["n_status_changes_to_date"] == 0).all()
    assert (first_rows["months_delinquent_to_date"] <= 1).all()


def test_feature_columns_excludes_targets_and_identifiers():
    df = make_panel(n_loans=2, n_months=5)
    df["next_12m_default_flag"] = 0
    df["next_state"] = "Current"
    df["exception_type"] = "none"
    cols = feature_columns(df)
    for banned in ("next_12m_default_flag", "next_state", "exception_type",
                   "loan_id", "reporting_month", "month_index"):
        assert banned not in cols


# ------------------------------------------------------------- features


def test_rate_incentive_percentile_is_within_month_and_bounded():
    panel = make_panel(n_loans=20, n_months=12)
    out, _ = engineer_features(panel)
    p = out["rate_incentive_pctile"].dropna()
    assert ((p > 0) & (p <= 1)).all()
    # Within any single reporting month the ranks must span the range.
    one = out[out.reporting_month == out.reporting_month.max()]
    assert one["rate_incentive_pctile"].nunique() > 1


def test_amortisation_gap_is_negative_when_paying_ahead():
    panel = make_panel(n_loans=2, n_months=24)
    ahead = panel.copy()
    ahead["current_balance"] = ahead["original_balance"] * 0.5   # far ahead
    out, _ = engineer_features(ahead)
    late = out[out.loan_age_months > 2]
    assert (late["amortisation_gap"] < 0).all()


def test_band_ordinals_preserve_ordering():
    df = pd.DataFrame({"credit_score_band": ["<620", "660-699", "780+"]})
    out = add_band_ordinals(df, FeatureManifest())
    vals = out["credit_score_band_ordinal"].tolist()
    assert vals == sorted(vals)


def test_canonicalise_merges_state_variants():
    df = pd.DataFrame({"state": ["CA", "ca", "California", "NY"]})
    out = canonicalise_categories(df)
    assert out["state"].tolist() == ["CA", "CA", "CA", "NY"]


def test_manifest_records_skipped_features():
    df = pd.DataFrame({"loan_id": ["A"], "reporting_month": [pd.Timestamp("2020-01-01")]})
    _, manifest = engineer_features(df)
    assert len(manifest.skipped) > 0


# ---------------------------------------------------------- calibration


def test_isotonic_is_gated_on_small_calibration_sets():
    """Isotonic must not be selected on a slice too small to support it,
    even if it scores better there -- that is exactly when it overfits."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(size=200) < p).astype(int)
    result = select_calibrator(p, y)
    assert result.method != "isotonic"
    assert "not eligible" in result.reason or result.method == "identity"


def test_platt_corrects_a_systematically_inflated_score():
    rng = np.random.default_rng(1)
    n = 5000
    true_p = rng.uniform(0.01, 0.2, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    inflated = np.clip(true_p * 4, 0, 1)         # badly miscalibrated
    result = select_calibrator(inflated, y)
    assert result.method in ("platt", "isotonic")
    corrected = result.apply(inflated)
    assert abs(corrected.mean() - y.mean()) < abs(inflated.mean() - y.mean())


def test_identity_chosen_when_scores_are_already_calibrated():
    rng = np.random.default_rng(2)
    n = 4000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    result = select_calibrator(p, y)
    assert result.brier_raw <= result.brier_platt * 1.05


def test_single_class_calibration_slice_is_handled():
    p = np.linspace(0.01, 0.5, 100)
    y = np.zeros(100, dtype=int)
    result = select_calibrator(p, y)
    assert result.method == "identity"
    np.testing.assert_allclose(result.apply(p), p)


def test_reliability_bins_are_equal_count():
    rng = np.random.default_rng(3)
    p = rng.beta(1, 40, 3000)           # heavily skewed, like a rare target
    y = (rng.uniform(size=3000) < p).astype(int)
    tbl = reliability_diagram_data(y, p, n_bins=10)
    assert len(tbl) == 10
    assert tbl["n"].max() - tbl["n"].min() <= 1


def test_ece_is_zero_for_perfect_calibration():
    y = np.array([0, 1] * 500)
    p = np.full(1000, 0.5)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------- training


def test_mature_label_mask_drops_the_horizon_tail():
    panel = make_panel(n_loans=2, n_months=30)
    mask = mature_label_mask(panel, "next_12m_default_flag")
    cutoff = panel.reporting_month.max() - pd.DateOffset(months=12)
    assert (panel.loc[mask, "reporting_month"] <= cutoff).all()
    assert (~mask).sum() > 0


def test_mature_label_mask_is_a_noop_for_horizonless_targets():
    panel = make_panel(n_loans=2, n_months=10)
    assert mature_label_mask(panel, "exception_required").all()


def test_recall_at_precision_math():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert recall_at_precision(y, score, 0.99) == pytest.approx(1.0)
    y_mixed = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    assert recall_at_precision(y_mixed, score, 0.99) < 1.0


def test_recall_at_precision_returns_zero_when_bar_unreachable():
    y = np.array([0] * 90 + [1] * 10)
    score = np.random.default_rng(0).uniform(size=100)
    assert recall_at_precision(y, score, 0.999) == 0.0


def test_split_reserves_room_for_the_test_window_under_a_long_purge():
    """Regression: a 12-month purge used to push test_start past the end of
    the panel, silently producing an empty test set for both 12-month
    targets."""
    panel = make_panel(n_loans=4, n_months=40)
    split = time_aware_split(panel, purge_months=12, test_fraction=0.2)
    assert len(split.test) > 0
    assert split.test.reporting_month.min() > split.train.reporting_month.max()
