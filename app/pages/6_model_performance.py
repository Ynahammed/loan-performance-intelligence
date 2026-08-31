"""Model Performance page: metrics, calibration, error analysis. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import (  # noqa: E402
    load_csv, load_json, load_markdown, page_header, require,
)

page_header(
    "Model Performance",
    "Held-out, purged and label-mature results for every target, with the "
    "results that did not go our way included.",
    inputs=["supervised_metrics.json", "uncertainty.csv"],
)

require("supervised_metrics.json")
supervised = load_json("supervised_metrics.json")

LABELS = {
    "next_3m_delinquency_flag": "3-month delinquency",
    "next_6m_delinquency_flag": "6-month delinquency",
    "next_12m_default_flag": "12-month default",
    "next_12m_prepayment_flag": "12-month prepayment",
}

st.info(
    "**PR-AUC is the headline, not ROC-AUC.** At prevalences between 0.9% "
    "and 9%, ROC-AUC is dominated by the majority class and flatters "
    "everything. `pr_auc_lift` is PR-AUC over the base rate, which is what "
    "makes the four targets comparable to each other.",
    icon="📐",
)

# ------------------------------------------------------------ summary
rows = []
for target, label in LABELS.items():
    entry = supervised.get(target)
    if not entry:
        continue
    metrics = pd.DataFrame(entry["metrics"])
    base = metrics[metrics.model.str.startswith("baseline")]
    best = (metrics[metrics.calibrated] if metrics.calibrated.any()
            else metrics).sort_values("pr_auc", ascending=False).iloc[0]
    rows.append({
        "Target": label,
        "Prevalence": best["prevalence"],
        "Baseline PR-AUC": base.iloc[0]["pr_auc"] if len(base) else None,
        "Final PR-AUC": best["pr_auc"],
        "Lift": best.get("pr_auc_lift"),
        "ROC-AUC": best["roc_auc"],
        "Recall@P50": best.get("recall_at_p50"),
        "Brier": best["brier"],
        "ECE": best["ece"],
        "Champion": entry["champion"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.warning(
    "**Two results that did not go our way, reported rather than buried.** "
    "Prepayment reaches recall of 0.000 at 50% precision — a "
    "precision-controlled prepayment queue cannot be run at all on this "
    "data. And on 12-month default the engineered features add nothing: "
    "the logistic baseline ties the gradient-boosted model and is champion, "
    "because roughly 66 events cannot pay for the extra capacity.",
    icon="⚠️",
)

# --------------------------------------------------------- per target
st.subheader("Baseline versus improved")
target = st.selectbox("Target", list(LABELS), format_func=lambda k: LABELS[k])
entry = supervised.get(target)
if entry:
    st.dataframe(pd.DataFrame(entry["metrics"]),
                 use_container_width=True, hide_index=True)
    cal = entry.get("calibration")
    if cal:
        st.subheader("Calibration")
        c1, c2, c3 = st.columns(3)
        c1.metric("Method", cal.get("method", "-"))
        c2.metric("Brier, raw", "{}".format(cal.get("brier_raw", "-")))
        c3.metric("Improvement", "{}".format(cal.get("brier_improvement", "-")))
        st.caption("Selection reason: " + str(cal.get("reason", "-")))
        st.caption(
            "Isotonic is gated on more than Brier score. It once collapsed "
            "8,889 scored rows onto 18 distinct probabilities with half the "
            "portfolio sharing one value — individually well calibrated, "
            "and the ranking destroyed. It is now rejected whenever it "
            "collapses the ordering, regardless of how good its Brier looks."
        )

# ------------------------------------------------------- next_state
ns = supervised.get("next_state")
if ns:
    st.subheader("Next-state prediction")
    st.dataframe(pd.DataFrame(ns["metrics"]),
                 use_container_width=True, hide_index=True)
    st.caption(
        "The persistence baseline wins macro-F1 by refusing to predict "
        "change at all, while being roughly 3.8x worse as a probability. It "
        "emits hard 0/1 labels, cannot rank, and cannot fill "
        "`next_state_confidence` — so it is reported honestly and excluded "
        "from champion selection."
    )

# ------------------------------------------------------- uncertainty
try:
    unc = load_csv("uncertainty.csv")
    st.subheader("Prediction stability")
    over = (unc.spread > unc.mean_prediction).mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", "{:,}".format(len(unc)))
    c2.metric("Median spread", "{:.5f}".format(unc.spread.median()))
    c3.metric("Spread exceeds prediction", "{:.1%}".format(over))
    fig = px.scatter(unc.sample(min(3000, len(unc)), random_state=0),
                     x="mean_prediction", y="spread", opacity=0.35,
                     color="confidence" if "confidence" in unc.columns else None)
    fig.update_layout(height=360, xaxis_title="mean prediction across folds",
                      yaxis_title="spread across folds")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Uncertainty is dispersion across models refitted on different "
        "expanding time windows — not a bootstrap. Sampling noise is not "
        "what breaks a loan model in deployment; the world moving is."
    )
except Exception:
    pass

with st.expander("Full model performance report"):
    st.markdown(load_markdown("model_performance_report.md") or "Not generated yet.")
with st.expander("Model card"):
    st.markdown(load_markdown("model_card.md") or "Not generated yet.")
