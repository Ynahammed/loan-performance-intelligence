"""
Tests for SHAP aggregation, error analysis and uncertainty.

Two pin things that fail silently rather than loudly:

  - `_positive_class_values`: explainers return (n,f), (n,f,2) or a list
    depending on version and estimator. Picking the wrong slice explains
    the NEGATIVE class, and every number downstream is sign-flipped
    without anything raising.
  - `source_column_map`: if one-hot columns do not aggregate back to
    their source, global importance shows forty near-zero state
    indicators and no row for "state".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.explainability.error_analysis import (
    _standardised_gap,
    analyse_errors,
    choose_threshold,
    confidence_band,
    temporal_ensemble_uncertainty,
)
from src.explainability.shap_explainer import (
    _positive_class_values,
    explain_model,
    explain_one,
    phrase_local_explanation,
    source_column_map,
)
from src.models.train import build_preprocessor


def make_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "loan_age_months": rng.integers(1, 60, n),
        "interest_rate": rng.normal(5.0, 0.8, n),
        "balance_ratio": rng.normal(0.9, 0.08, n),
        "credit_score_band": rng.choice(["<620", "700-739", "780+"], n),
        "state": rng.choice(["CA", "TX", "NY"], n),
        "reporting_month": [
            pd.Timestamp("2020-01-01") + pd.DateOffset(months=int(i % 40))
            for i in range(n)
        ],
    })
    logit = (-2 + 0.05 * df.interest_rate * 4
             - 3 * df.balance_ratio
             + (df.credit_score_band == "<620") * 1.5)
    df["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


FEATURES = ["loan_age_months", "interest_rate", "balance_ratio",
            "credit_score_band", "state"]


def fit_pipeline(df, cols=FEATURES):
    model = Pipeline([
        ("pre", build_preprocessor(df, cols, scale=False)),
        ("clf", HistGradientBoostingClassifier(max_iter=40, random_state=0)),
    ])
    model.fit(df[cols], df["y"])
    return model


# ------------------------------------------------------- shape handling


def test_positive_class_values_handles_three_dimensional_output():
    """(n, f, 2) must yield the POSITIVE class slice. Taking the first
    slice explains the negative class and flips every sign downstream."""
    v = np.zeros((5, 3, 2))
    v[:, :, 0] = -1.0
    v[:, :, 1] = 2.0
    out = _positive_class_values(v)
    assert out.shape == (5, 3)
    assert (out == 2.0).all()


def test_positive_class_values_passes_two_dimensional_through():
    v = np.ones((7, 4))
    np.testing.assert_array_equal(_positive_class_values(v), v)


# ------------------------------------------------------- name mapping


def test_source_column_map_aggregates_one_hot_back_to_the_source():
    df = make_data(200)
    pre = build_preprocessor(df, FEATURES, scale=False)
    pre.fit(df[FEATURES])
    names = list(pre.get_feature_names_out())
    sources = source_column_map(pre, names)

    assert len(sources) == len(names)
    assert "state" in sources
    assert "credit_score_band" in sources
    # Every one-hot piece of `state` must resolve to `state`.
    for name, src in zip(names, sources):
        if "state_" in name:
            assert src == "state"


def test_source_column_map_prefers_the_longest_matching_prefix():
    """`credit_score_band_740-779` must map to `credit_score_band`, not to
    a shorter column that happens to be a prefix."""
    class _Stub:
        transformers_ = [
            ("num", "passthrough", ["credit"]),
            ("cat", "passthrough", ["credit_score_band"]),
        ]

    names = ["num__credit", "cat__credit_score_band_740-779"]
    assert source_column_map(_Stub(), names) == ["credit", "credit_score_band"]


# --------------------------------------------------------- attributions


def test_global_importance_covers_every_source_column_and_sums_to_100():
    df = make_data()
    model = fit_pipeline(df)
    result = explain_model(model, df, FEATURES, sample=300)
    assert set(result.global_importance.source_column) == set(FEATURES)
    assert result.global_importance.share_pct.sum() == pytest.approx(100.0, abs=0.5)
    assert (result.global_importance.mean_abs_shap >= 0).all()


def test_global_importance_ranks_the_planted_driver_highly():
    df = make_data(1200)
    model = fit_pipeline(df)
    result = explain_model(model, df, FEATURES, sample=400)
    top3 = set(result.global_importance.head(3).source_column)
    assert "balance_ratio" in top3 or "interest_rate" in top3


def test_local_explanation_is_signed_and_limited():
    df = make_data()
    model = fit_pipeline(df)
    result = explain_model(model, df, FEATURES, sample=200)
    local = explain_one(model, df.iloc[0], FEATURES, result, top_n=3)
    assert len(local) <= 3
    assert set(local.direction) <= {"increases risk", "reduces risk"}
    assert local.contribution.abs().is_monotonic_decreasing


def test_reviewer_phrasing_never_claims_causation():
    local = pd.DataFrame([
        {"label": "credit score", "contribution": 0.4, "direction": "increases risk"},
        {"label": "interest rate", "contribution": -0.2, "direction": "reduces risk"},
    ])
    text = phrase_local_explanation(local, 0.31).lower()
    # Check for causal CLAIMS, not the substring "cause" -- the required
    # disclaimer ("not established causes") legitimately contains it.
    assert "because" not in text
    assert "caused by" not in text
    assert "leads to" not in text
    assert "not established causes" in text
    assert "the model" in text
    assert "31" in text


def test_reviewer_phrasing_handles_an_empty_attribution():
    assert phrase_local_explanation(pd.DataFrame(), 0.1)


# -------------------------------------------------------- error analysis


def test_choose_threshold_reports_when_precision_is_unreachable():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=500) < 0.02).astype(int)
    score = rng.uniform(size=500)          # pure noise
    threshold, reason = choose_threshold(y, score, target_precision=0.9)
    assert threshold == 0.5
    assert "unreachable" in reason


def test_choose_threshold_finds_a_reachable_operating_point():
    y = np.array([0] * 90 + [1] * 10)
    score = np.r_[np.linspace(0.0, 0.4, 90), np.linspace(0.8, 0.99, 10)]
    threshold, reason = choose_threshold(y, score, target_precision=0.9)
    # 0.4 is a valid answer: it admits the single highest-scoring negative
    # alongside all ten positives, for 10/11 = 90.9% precision.
    assert 0.4 <= threshold < 0.99
    assert "unreachable" not in reason
    precision = (score[score >= threshold] > 0.4).mean()
    assert precision >= 0.9


def test_confusion_counts_are_internally_consistent():
    df = make_data(600)
    model = fit_pipeline(df)
    p = model.predict_proba(df[FEATURES])[:, 1]
    errors = analyse_errors(df, df["y"], p)
    assert errors.confusion.n.sum() == len(df)
    counts = errors.confusion.set_index("outcome").n
    assert counts["true positive"] + counts["false negative"] == int(df.y.sum())


def test_standardised_gap_is_signed_and_ordered_by_magnitude():
    a = pd.DataFrame({"x": np.r_[np.ones(50) * 5, np.ones(50) * 5.2],
                      "y": np.random.default_rng(0).normal(0, 1, 100)})
    b = pd.DataFrame({"x": np.random.default_rng(1).normal(0, 1, 100),
                      "y": np.random.default_rng(2).normal(0, 1, 100)})
    out = _standardised_gap(a, b, ["x", "y"])
    assert out.iloc[0]["feature"] == "x"
    assert out.iloc[0]["std_gap"] > 0
    assert out["std_gap"].abs().is_monotonic_decreasing


def test_segment_errors_skip_thin_segments():
    df = make_data(600)
    df.loc[df.index[:5], "state"] = "ZZ"       # a segment far below the floor
    model = fit_pipeline(df)
    p = model.predict_proba(df[FEATURES])[:, 1]
    errors = analyse_errors(df, df["y"], p)
    if "state" in errors.segment_errors:
        assert "ZZ" not in set(errors.segment_errors["state"].segment)


# ----------------------------------------------------------- uncertainty


def test_confidence_band_is_relative_not_absolute():
    """A swing of 0.02 is noise around 0.40 and enormous around 0.01."""
    spread = pd.Series([0.02, 0.02])
    mean = pd.Series([0.40, 0.01])
    bands = confidence_band(spread, mean, method="absolute")
    assert bands.iloc[0] == "high confidence"
    assert bands.iloc[1] == "low confidence"


def test_quantile_bands_actually_split_the_portfolio():
    """Regression: fixed cutoffs put 91% of rows in one band on a
    low-prevalence target, which is a label that discriminates nothing."""
    rng = np.random.default_rng(0)
    spread = pd.Series(rng.uniform(0.001, 0.05, 900))
    mean = pd.Series(rng.uniform(0.002, 0.02, 900))
    bands = confidence_band(spread, mean)
    shares = bands.value_counts(normalize=True)
    assert len(shares) == 3
    assert shares.max() < 0.5


def test_uncertainty_degrades_gracefully_on_a_short_panel():
    df = make_data(200)
    df["reporting_month"] = pd.Timestamp("2020-01-01")   # one month only
    result = temporal_ensemble_uncertainty(df, "y", FEATURES, df, n_splits=4)
    assert result.per_row.empty
    assert result.n_folds == 0
    assert any("failed" in n or "fewer than two" in n for n in result.notes)


def test_uncertainty_produces_a_spread_across_folds():
    df = make_data(2000)
    eval_df = df.tail(200)
    result = temporal_ensemble_uncertainty(
        df, "y", FEATURES, eval_df, n_splits=3, max_iter=30
    )
    if result.n_folds >= 2:
        assert len(result.per_row) == len(eval_df)
        assert (result.per_row["spread"] >= 0).all()
        assert (result.per_row["max_prediction"]
                >= result.per_row["min_prediction"]).all()
