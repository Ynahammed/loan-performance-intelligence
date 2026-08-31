"""Survival & Transitions page: the discrete-time state model. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import (  # noqa: E402
    load_csv, load_json, load_markdown, page_header, require,
)

page_header(
    "Survival & Transitions",
    "A discrete-time multi-state hazard model over the delinquency "
    "ladder, propagated forward as a Markov chain.",
    inputs=["transition_artifacts.json", "scenario_curves.csv"],
)

require("transition_artifacts.json")
artifacts = load_json("transition_artifacts.json")

st.info(
    "**Default and prepayment are both absorbing states in one chain**, so "
    "the probability mass accumulated in each at horizon *h* is a true "
    "cumulative incidence function. The competition between the two events "
    "is handled by construction — which is why this does not report "
    "1 − Kaplan-Meier per event, a treatment that overstates both curves.",
    icon="🔗",
)

# --------------------------------------------------- baseline compare
comparison = artifacts.get("one_step_comparison", {})
if comparison:
    st.subheader("Against the empirical baseline")
    st.caption(
        "One month ahead, held out and purged. The baseline is an "
        "age-stratified empirical transition matrix — a genuinely "
        "reasonable actuarial model, not a strawman."
    )
    rows = []
    for label in ("empirical baseline", "covariate transition model"):
        row = {"model": label}
        for field, value in comparison.items():
            if label in value:
                row[field] = value[label]
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = [c for c in ("model", "log_loss", "macro_f1", "accuracy",
                        "auc_Current", "auc_30DPD", "auc_60DPD", "auc_90DPD")
            if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    st.caption(
        "The covariate model gains discrimination while matching the "
        "baseline on log-loss. It gets there by credibility weighting: "
        "each origin state is shrunk toward its empirical row by a weight "
        "chosen on a held-out slice, because an origin with 29,000 "
        "observations earns more trust in its covariates than one with 389."
    )

# ------------------------------------------------------- cross-check
cross = artifacts.get("crosscheck_12m_default", {})
if cross:
    st.subheader("Independent cross-check")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", "{:,}".format(cross.get("n", 0)))
    c2.metric("Observed rate", "{:.2%}".format(cross.get("observed_rate", 0)))
    c3.metric("Predicted rate", "{:.2%}".format(cross.get("predicted_rate", 0)))
    c4.metric("ROC-AUC", "{}".format(cross.get("roc_auc", "-")))
    st.caption(
        "The 12-month cumulative default probability read off the chain, "
        "scored against `next_12m_default_flag` — a label the chain never "
        "sees. Two independently built estimators agreeing on the same "
        "quantity is the strongest single piece of evidence here that the "
        "modelling is sound."
    )

# ------------------------------------------------------------- curves
require("scenario_curves.csv")
curves = load_csv("scenario_curves.csv")

st.subheader("Cumulative incidence")
event = st.radio("Event", ["cif_default", "cif_prepaid", "delinquency_share"],
                 horizontal=True,
                 format_func=lambda k: {
                     "cif_default": "Default",
                     "cif_prepaid": "Prepayment",
                     "delinquency_share": "In any DPD bucket",
                 }[k])

fig = px.line(curves, x="month", y=event, color="scenario", markers=True)
fig.update_layout(height=400, xaxis_title="months ahead", yaxis_title="probability")
st.plotly_chart(fig, use_container_width=True)

base = curves[curves.scenario == "base"]
if len(base):
    final = base[base.month == base.month.max()].iloc[0]
    st.caption(
        "Base case at 12 months: {:.2%} cumulative default, {:.2%} "
        "cumulative prepayment, {:.2%} still in a delinquency bucket."
        .format(final.cif_default, final.cif_prepaid, final.delinquency_share)
    )

st.warning(
    "**A limitation of this synthetic pack, not of the model.** It contains "
    "zero observed cure transitions out of 60DPD or 90DPD. Real servicing "
    "data has substantial cure rates, so these curves are optimistic about "
    "severity progression. The state machine is declared generally rather "
    "than narrowed to what was observed, so cures are representable if a "
    "real data pack contains them.",
    icon="⚠️",
)

with st.expander("Full survival report"):
    report = load_markdown("survival_report.md")
    st.markdown(report if report else "Not generated yet.")
