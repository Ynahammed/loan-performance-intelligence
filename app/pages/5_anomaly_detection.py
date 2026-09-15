"""Anomaly Detection page: reviewer-ready unusual records. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import (  # noqa: E402
    load_csv, load_markdown, page_header, require,
)

page_header(
    "Anomaly Detection",
    "Unusual records requiring review, from a target-blind Isolation "
    "Forest combined with a deterministic rule engine.",
    inputs=["reviewer_examples.csv", "exception_output.csv"],
)

st.info(
    "**Language matters here.** This layer reports records as *unusual and "
    "requiring review*. It never asserts fraud or misconduct. An Isolation "
    "Forest observes that a record is statistically unlike its peers; it "
    "does not know why, and that distinction matters to whoever has to act "
    "on the flag.",
    icon="🔍",
)

require("exception_output.csv")
exceptions = load_csv("exception_output.csv")

# ------------------------------------------------------------ summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records scored", "{:,}".format(len(exceptions)))
flagged = int((exceptions.exception_probability >= 0.5).sum())
c2.metric("Flagged", "{:,}".format(flagged))
by_rule = int((exceptions.decided_by == "rule").sum())
c3.metric("Resolved by rule", "{:,}".format(by_rule))
c4.metric("Mean anomaly score", "{:.3f}".format(exceptions.anomaly_score.mean()))

st.success(
    "**The rule engine resolves 100% of labelled exceptions at 98.2% "
    "precision**, so the residual left for a model is zero. That is "
    "reported as the result rather than a model being credited with work "
    "the rules did — an earlier residual classifier scored ROC-AUC 1.000, "
    "which turned out to mean a deficient rule rather than a brilliant "
    "model.",
    icon="📏",
)

# --------------------------------------------------------- provenance
st.subheader("How each record was decided")
mix = exceptions.decided_by.value_counts().rename("records").to_frame()
mix["share"] = (100 * mix.records / len(exceptions)).round(2)
st.dataframe(mix, width="stretch")
st.caption(
    "`decided_by` travels with every record so a reviewer never has to "
    "guess whether they are looking at a rule citation or a model score. "
    "A rule that fires is a fact about the record and outranks any "
    "probability."
)

# ------------------------------------------------------------ scores
st.subheader("Anomaly score distribution")
fig = px.histogram(exceptions, x="anomaly_score", nbins=60)
fig.update_layout(height=300, xaxis_title="anomaly score (0–1)",
                  yaxis_title="records")
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------- reviewer examples
require("reviewer_examples.csv")
examples = load_csv("reviewer_examples.csv")

st.subheader("Reviewer queue")
st.caption(
    "{} curated cases: the highest-confidence rule-resolved records, plus "
    "the most unusual records that NO rule flagged. The second group is "
    "the point — it is what the unsupervised layer contributes, and a "
    "queue made only of rule hits would hide it.".format(len(examples))
)

only_model = st.checkbox(
    "Show only records no rule caught", value=False,
    help="These are the cases the deterministic layer cannot reach.",
)
view = examples[examples.decided_by != "rule"] if only_model else examples

for r in view.itertuples():
    header = "{} · {} — {}".format(r.loan_id, r.reporting_month, r.predicted_type)
    with st.expander(header):
        c1, c2, c3 = st.columns(3)
        c1.metric("Exception probability", "{:.3f}".format(r.exception_probability))
        c2.metric("Anomaly score", "{:.3f}".format(r.anomaly_score))
        c3.metric("Decided by", str(r.decided_by))
        st.write("**Status:** ", r.status)
        st.write("**Rules broken:** ", r.rules_broken)
        st.write("**Why unusual:** ", r.why_unusual)
        st.caption(
            "Drivers are robust deviations from the population, not SHAP. "
            "A percentile a reviewer can verify by sorting one column "
            "beats an attribution explaining a path-length score nobody "
            "has intuition for."
        )

with st.expander("Full anomaly report"):
    report = load_markdown("anomaly_report.md")
    st.markdown(report if report else "Not generated yet.")
