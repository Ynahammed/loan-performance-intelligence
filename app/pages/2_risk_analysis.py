"""Risk Analysis page: score distributions and segment risk. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import load_csv, page_header, require  # noqa: E402

page_header(
    "Risk Analysis",
    "Scored probabilities across the unlabelled test panel, with "
    "configurable risk bands.",
    inputs=["submission.csv"],
)

require("submission.csv")
sub = load_csv("submission.csv")

TARGETS = {
    "next_3m_delinquency_prob": "3-month delinquency",
    "next_6m_delinquency_prob": "6-month delinquency",
    "next_12m_default_prob": "12-month default",
    "next_12m_prepayment_prob": "12-month prepayment",
}

target = st.selectbox(
    "Outcome", list(TARGETS), format_func=lambda k: TARGETS[k],
)

# --------------------------------------------------------- thresholds
st.sidebar.subheader("Risk bands")
st.sidebar.caption(
    "Thresholds are configurable, not hardcoded. Moving them changes the "
    "banding live; it does not retrain anything."
)
low = st.sidebar.slider("Low / Medium boundary", 0.0, 1.0, 0.30, 0.01)
high = st.sidebar.slider("Medium / High boundary", 0.0, 1.0, 0.60, 0.01)
if high <= low:
    st.sidebar.error("The upper boundary must exceed the lower one.")
    st.stop()

scores = sub[target]
band = pd.cut(scores, [-0.001, low, high, 1.001],
              labels=["Low", "Medium", "High"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records scored", "{:,}".format(len(scores)))
c2.metric("Mean probability", "{:.2%}".format(scores.mean()))
c3.metric("High risk", "{:,}".format(int((band == "High").sum())))
c4.metric("Max probability", "{:.2%}".format(scores.max()))

# ------------------------------------------------------------ dist
st.subheader("Probability distribution")
fig = px.histogram(sub, x=target, nbins=60)
fig.add_vline(x=low, line_dash="dash", annotation_text="Low/Med")
fig.add_vline(x=high, line_dash="dash", annotation_text="Med/High")
fig.update_layout(height=340, xaxis_title=TARGETS[target], yaxis_title="records")
st.plotly_chart(fig, use_container_width=True)

if scores.max() < high:
    st.info(
        "No record reaches the High band at the current threshold. That is "
        "a property of a calibrated model on a low-prevalence outcome, not "
        "a bug: the highest score here is {:.2%}. Calibrated probabilities "
        "for a rare event are legitimately small, and a model that "
        "routinely emitted 80% for a 1%-prevalence outcome would be "
        "badly calibrated rather than confident.".format(scores.max()),
        icon="ℹ️",
    )

# --------------------------------------------------------- by segment
st.subheader("Risk by segment")
panel_cols = [c for c in ("next_state_pred", "exception_type_pred",
                          "recommended_action") if c in sub.columns]
segment = st.selectbox("Segment by", panel_cols)
agg = (sub.groupby(segment)[target]
       .agg(records="size", mean_probability="mean")
       .reset_index().sort_values("mean_probability", ascending=False))
agg["mean_probability"] = agg["mean_probability"].round(5)
st.dataframe(agg, use_container_width=True, hide_index=True)

# ------------------------------------------------------- action queue
st.subheader("Review queue")
st.caption(
    "Produced by a deterministic policy over model outputs with published "
    "thresholds — not by a model, and never a lending decision."
)
actions = sub.recommended_action.value_counts().rename("records").to_frame()
actions["share"] = (100 * actions.records / len(sub)).round(2)
st.dataframe(actions, use_container_width=True)

st.subheader("Highest-scoring records")
show = [c for c in ("loan_id", "reporting_month", target, "next_state_pred",
                    "exception_type_pred", "anomaly_score",
                    "model_confidence", "recommended_action") if c in sub.columns]
st.dataframe(sub.nlargest(25, target)[show],
             use_container_width=True, hide_index=True)
