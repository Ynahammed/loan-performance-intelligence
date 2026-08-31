"""Loan Explorer page: one loan, its history and its scores. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import (  # noqa: E402
    load_csv, load_panel, page_header, require,
)

page_header(
    "Loan Explorer",
    "One loan at a time: its servicing history, its scores, and what the "
    "model weighted.",
    inputs=["submission.csv"],
)

require("submission.csv")
sub = load_csv("submission.csv")
panel = load_panel()

# ---------------------------------------------------------- selection
scored = sorted(sub.loan_id.unique())
default_idx = 0
if "next_3m_delinquency_prob" in sub.columns:
    riskiest = sub.nlargest(1, "next_3m_delinquency_prob").loan_id.iloc[0]
    if riskiest in scored:
        default_idx = scored.index(riskiest)

loan_id = st.selectbox("Loan", scored, index=default_idx)
history = panel[panel.loan_id == loan_id].sort_values("reporting_month")
rows = sub[sub.loan_id == loan_id].sort_values("reporting_month")

if history.empty:
    st.warning(
        "This loan has no rows in the training panel — it appears only in "
        "the scored period. Its history features start from zero, which "
        "makes them structurally less informative for this record."
    )

# ------------------------------------------------------------ profile
st.subheader("Origination profile")
if len(history):
    first = history.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original balance", "${:,.0f}".format(first.get("original_balance", 0)))
    c2.metric("Rate", "{:.3f}%".format(first.get("interest_rate", 0)))
    c3.metric("Credit band", str(first.get("credit_score_band", "-")))
    c4.metric("LTV band", str(first.get("ltv_band", "-")))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State", str(first.get("state", "-")))
    c2.metric("Purpose", str(first.get("loan_purpose", "-")))
    c3.metric("Servicer", str(first.get("servicer_name", "-")))
    c4.metric("Months observed", "{:,}".format(len(history)))

# ------------------------------------------------------------ history
if len(history):
    st.subheader("Servicing history")
    fig = px.line(history, x="reporting_month", y="current_balance", markers=False)
    fig.update_layout(height=280, xaxis_title="", yaxis_title="current balance")
    st.plotly_chart(fig, use_container_width=True)

    status = history[["reporting_month", "current_status", "days_past_due"]].copy()
    status["reporting_month"] = status.reporting_month.dt.strftime("%Y-%m")
    delinquent = status[status.current_status != "Current"]
    if len(delinquent):
        st.caption(
            "Delinquent in {} of {} observed months. Worst status reached: "
            "{}.".format(len(delinquent), len(status),
                         delinquent.current_status.iloc[-1])
        )
    else:
        st.caption("Current in every observed month.")
    with st.expander("Month-by-month status"):
        st.dataframe(status, use_container_width=True, hide_index=True)

# ------------------------------------------------------------ scores
st.subheader("Scored months")
if rows.empty:
    st.info("This loan does not appear in the scored test panel.")
else:
    prob_cols = [c for c in rows.columns if c.endswith("_prob")]
    show = ["reporting_month"] + prob_cols + [
        c for c in ("next_state_pred", "next_state_confidence",
                    "exception_type_pred", "anomaly_score",
                    "model_confidence") if c in rows.columns]
    st.dataframe(rows[show], use_container_width=True, hide_index=True)

    st.subheader("What the model weighted")
    st.caption(
        "SHAP attributions describe what moved this model's output. They "
        "are not statements about what causes a loan to default."
    )
    for r in rows.itertuples():
        with st.expander("{} — {}".format(r.reporting_month, r.recommended_action)):
            st.write("**Top drivers:** ", r.top_drivers)
            st.write("**Anomaly score:** {:.4f}".format(r.anomaly_score))
            st.write("**Model confidence:** {:.4f}".format(r.model_confidence))
            st.caption(
                "Model confidence is the complement of the percentile rank "
                "of prediction spread across models trained on different "
                "time periods — how much this score depends on when the "
                "model was fitted."
            )
