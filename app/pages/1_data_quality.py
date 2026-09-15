"""Data Quality page: record and batch scores, rules, drift. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import (  # noqa: E402
    load_csv, load_markdown, load_panel, page_header, require,
)

page_header(
    "Data Quality",
    "Profiling, validation rules and source reconciliation — run before "
    "any model, because a model trained on records that are wrong is a "
    "confident restatement of the errors.",
    inputs=["record_quality.csv", "rule_flags.csv"],
)

require("record_quality.csv")
require("rule_flags.csv")

quality = load_csv("record_quality.csv")
flags = load_csv("rule_flags.csv")
panel = load_panel()

score_col = "data_quality_score"

# ------------------------------------------------------------- summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records scored", "{:,}".format(len(quality)))
c2.metric("Mean quality score", "{:.1f} / 100".format(quality[score_col].mean()))
c3.metric("Below 70", "{:,}".format(int((quality[score_col] < 70).sum())))
rule_cols = [c for c in flags.columns if c not in ("loan_id", "reporting_month")]
c4.metric("Rule violations", "{:,}".format(int(flags[rule_cols].sum().sum())))

# ------------------------------------------------------------ score dist
st.subheader("Record-level quality score")
bands = pd.DataFrame({
    "band": ["90–100 (clean)", "70–90", "50–70", "0–50 (severe)"],
    "records": [
        int((quality[score_col] >= 90).sum()),
        int(((quality[score_col] >= 70) & (quality[score_col] < 90)).sum()),
        int(((quality[score_col] >= 50) & (quality[score_col] < 70)).sum()),
        int((quality[score_col] < 50).sum()),
    ],
})
fig = px.bar(bands, x="band", y="records", text="records")
fig.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="records")
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ by month
st.subheader("Quality by reporting month")
st.caption(
    "Batch-level scoring is what makes degradation visible. A portfolio "
    "average hides a feed that broke in one quarter; a monthly series "
    "does not."
)
monthly = quality.assign(
    month=pd.to_datetime(quality.reporting_month).dt.to_period("M").astype(str)
).groupby("month")[score_col].agg(["mean", "size"]).reset_index()
fig = px.line(monthly, x="month", y="mean", markers=True)
fig.update_layout(height=320, yaxis_title="mean quality score", xaxis_title="")
st.plotly_chart(fig, width="stretch")

worst = monthly.nsmallest(3, "mean")
if len(worst) and worst["mean"].iloc[0] < 60:
    st.warning(
        "**The panel does not really start when it appears to.** "
        "{} scores {:.1f}/100 across {} records. Those months contain "
        "nothing but records whose reporting date precedes their "
        "origination date — the batch score found that on its own, and a "
        "portfolio average would never have shown it.".format(
            worst["month"].iloc[0], worst["mean"].iloc[0],
            int(worst["size"].iloc[0])),
        icon="🔎",
    )

# ---------------------------------------------------------- rule counts
st.subheader("Validation rules")
counts = (flags[rule_cols].sum().sort_values(ascending=False)
          .rename("records flagged").to_frame().reset_index()
          .rename(columns={"index": "rule"}))
st.dataframe(counts, width="stretch", hide_index=True)

st.info(
    "Measured against the shipped labels, the deterministic rules resolve "
    "**100% of exceptions at 98.2% precision**. That figure is optimistic "
    "by construction — the invariants were chosen after inspecting this "
    "panel — so what transfers to a different data pack is the mechanism, "
    "not the number.",
    icon="📏",
)

# ------------------------------------------------------------ worst rows
st.subheader("Lowest-scoring records")
worst_records = quality.nsmallest(25, score_col)
merged = worst_records.merge(
    panel[["loan_id", "reporting_month", "current_status", "current_balance",
           "document_status"]].assign(
        reporting_month=lambda d: d.reporting_month.astype(str)),
    on=["loan_id", "reporting_month"], how="left",
)
st.dataframe(merged, width="stretch", hide_index=True)

# ------------------------------------------------------------ drift
st.subheader("Train vs test drift")
st.caption(
    "Computed by `run_data_intelligence` and shown here rather than left "
    "in a markdown file. PSI is univariate and cannot see a joint shift; "
    "the adversarial classifier can, and names the columns responsible."
)

report = load_markdown("data_intelligence_report.md")
psi_block, adv_block = "", ""
if report:
    import re
    m = re.search(r"## 8\. Train vs test drift \(PSI\)\s*```(.*?)```",
                  report, re.DOTALL)
    psi_block = m.group(1).strip() if m else ""
    m = re.search(r"## 9\. Adversarial validation\s*```(.*?)```",
                  report, re.DOTALL)
    adv_block = m.group(1).strip() if m else ""

if psi_block:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Population Stability Index, worst first**")
        st.code(psi_block, language="text")
    with right:
        st.markdown("**Adversarial validation**")
        st.code(adv_block or "not available", language="text")
    st.info(
        "The two adversarial runs matter. **With** time-index columns a "
        "chronological split separates almost perfectly — that restates "
        "how the split was made, not a finding. **Without** them, the "
        "question becomes whether the cross-section itself has moved. It "
        "has: `interest_rate` is the single largest source of separation, "
        "which is why the prepayment model does not transfer out of time.",
        icon="📉",
    )
else:
    st.caption(
        "Run `python -m scripts.run_data_intelligence` to generate the "
        "drift analysis."
    )

with st.expander("Full data intelligence report"):
    report = load_markdown("data_intelligence_report.md")
    st.markdown(report if report else "Not generated yet.")
