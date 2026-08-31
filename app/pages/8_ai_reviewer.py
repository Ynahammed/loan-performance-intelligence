"""AI Reviewer page: the copilot and its governance record. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import (  # noqa: E402
    load_jsonl, load_markdown, page_header,
)

page_header(
    "AI Reviewer",
    "Grounded reviewer notes, the guardrails that validate them, and the "
    "audit trail of every call.",
    inputs=["llm_calls.jsonl"],
)

st.info(
    "**The LLM never produces a number.** It receives a validated JSON "
    "context object built from computed results — never a dataframe — and "
    "every generation is checked against that context before display. A "
    "rejected generation is replaced by a deterministic rendering of the "
    "same context, so the worst case for a reviewer is a plainer note, "
    "never a missing or unvalidated one.",
    icon="🔒",
)

log = load_jsonl("llm_calls.jsonl")
if log.empty:
    st.warning(
        "No calls logged yet. Run `python -m scripts.run_llm_review` to "
        "generate reviewer notes and populate the audit trail."
    )
    st.stop()

# ------------------------------------------------------------ summary
errored = log[log.error.notna()] if "error" in log.columns else log.iloc[0:0]
reached = log[~log.index.isin(errored.index)]
rejected = reached[~reached.accepted] if "accepted" in reached.columns else reached.iloc[0:0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Calls logged", "{:,}".format(len(log)))
c2.metric("Reached the guardrail", "{:,}".format(len(reached)))
c3.metric("Rejected", "{:,}".format(len(rejected)))
c4.metric("Provider errors", "{:,}".format(len(errored)))

st.caption(
    "Provider errors and guardrail rejections are counted separately. "
    "Pooling them once produced a 59% \"rejection rate\" that was mostly "
    "HTTP 404s from a retired model — a number that reads as \"the model "
    "hallucinates constantly\" when it means \"the endpoint was down\"."
)

if "provider" in log.columns:
    st.subheader("By provider")
    mix = log.groupby(["provider", "model"]).size().rename("calls").reset_index()
    st.dataframe(mix, use_container_width=True, hide_index=True)
    if (log.provider == "fault-injection").any():
        st.caption(
            "`fault-injection` rows are text we wrote deliberately to "
            "exercise the guardrails, labelled as such here and in the "
            "report. They are never presented as model output."
        )

# ------------------------------------------------------- rejections
if len(rejected):
    st.subheader("Why generations were rejected")
    counts = {}
    for v in rejected.get("validation", pd.Series(dtype=object)).dropna():
        for check in (v or {}).get("failed_checks", []):
            counts[check] = counts.get(check, 0) + 1
    if counts:
        st.dataframe(
            pd.Series(counts).rename("rejections").sort_values(ascending=False)
            .to_frame(), use_container_width=True,
        )
    st.caption(
        "Across live runs the dominant genuine failure was "
        "`carries_disclaimer`: the model writes accurate, well-grounded "
        "notes and simply omits the sentence marking the output as a "
        "recommendation, despite the prompt requiring it."
    )

# ------------------------------------------------------------- notes
st.subheader("Reviewer notes")
accepted = reached[reached.accepted] if "accepted" in reached.columns else reached
notes = accepted[accepted.task.isin(["reviewer_note", "reviewer_note_fallback"])] \
    if "task" in accepted.columns else accepted
for r in notes.tail(8).itertuples():
    with st.expander("{} · {} · {}".format(
            str(r.timestamp)[:19], r.provider, r.model)):
        st.write(r.output)
        st.caption("Validated against its own context before display.")

# -------------------------------------------------------- inspect log
with st.expander("Inspect a logged call"):
    idx = st.number_input("Record", 0, max(len(log) - 1, 0), 0)
    record = log.iloc[int(idx)].to_dict()
    st.json({k: v for k, v in record.items() if k != "prompt"})
    st.text_area("Prompt as sent", record.get("prompt", ""), height=220)

st.divider()
with st.expander("Rejected and corrected LLM output (required evidence)"):
    st.markdown(load_markdown("rejected_llm_outputs.md") or "Not generated yet.")
with st.expander("LLM copilot report"):
    st.markdown(load_markdown("llm_copilot_report.md") or "Not generated yet.")
