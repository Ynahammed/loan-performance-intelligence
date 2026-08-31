"""
Streamlit entry point.

    streamlit run app/dashboard.py

Reads the artifacts the pipeline runners produce. It does not train
anything: see `app/components/data_access.py` for why that separation
matters beyond speed.

PHASE: 11
STATUS: implemented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.data_access import (  # noqa: E402
    SYNTHETIC_BADGE, artifact_status, load_csv, load_json, load_panel,
)

st.set_page_config(
    page_title="Loan Performance Intelligence Engine",
    page_icon="📊",
    layout="wide",
)

st.title("Loan Performance Intelligence Engine")
st.caption(SYNTHETIC_BADGE)

st.markdown(
    "An ML-first engine for loan-level data: profiling, multi-horizon "
    "performance prediction, a discrete-time transition model, rule and "
    "anomaly-based exception detection, portfolio stress scenarios, "
    "explainability, and a governed LLM reviewer copilot."
)

st.info(
    "**The LLM never produces a number.** Every probability, score and "
    "driver on these pages is computed by a non-LLM model. The copilot "
    "narrates those results and its output is validated against them "
    "before display.",
    icon="🔒",
)

# ---------------------------------------------------------------- KPIs
panel = load_panel()
st.subheader("Portfolio")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Loans", "{:,}".format(panel.loan_id.nunique()))
c2.metric("Monthly records", "{:,}".format(len(panel)))
c3.metric(
    "Observation window",
    "{} – {}".format(panel.reporting_month.min().strftime("%b %Y"),
                     panel.reporting_month.max().strftime("%b %Y")),
)
delinquent = panel.current_status.isin(["30DPD", "60DPD", "90DPD"]).mean()
c4.metric("Records delinquent", "{:.2%}".format(delinquent))

# ----------------------------------------------------------- model row
try:
    supervised = load_json("supervised_metrics.json")
except Exception:
    supervised = {}

if supervised:
    st.subheader("Model performance")
    st.caption(
        "PR-AUC is the headline rather than ROC-AUC: at these prevalences "
        "ROC-AUC is dominated by the majority class and flatters everything."
    )
    rows = []
    labels = {
        "next_3m_delinquency_flag": "3-month delinquency",
        "next_6m_delinquency_flag": "6-month delinquency",
        "next_12m_default_flag": "12-month default",
        "next_12m_prepayment_flag": "12-month prepayment",
    }
    for target, label in labels.items():
        entry = supervised.get(target)
        if not entry:
            continue
        metrics = pd.DataFrame(entry["metrics"])
        best = (metrics[metrics.calibrated] if metrics.calibrated.any()
                else metrics).sort_values("pr_auc", ascending=False).iloc[0]
        rows.append({
            "Target": label,
            "Prevalence": "{:.2%}".format(best["prevalence"]),
            "PR-AUC": best["pr_auc"],
            "Lift vs base rate": "{}x".format(best.get("pr_auc_lift", "-")),
            "ROC-AUC": best["roc_auc"],
            "Recall @ 50% precision": best.get("recall_at_p50", "-"),
            "Champion": entry["champion"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.warning(
        "**Known limitation.** Prepayment is close to unpredictable out of "
        "time on this data: recall at 50% precision is 0.000, meaning a "
        "precision-controlled prepayment queue cannot be operated at all. "
        "The driver feature works within a period and does not transfer "
        "across the rate-environment shift. See the Model Performance page.",
        icon="⚠️",
    )

# ---------------------------------------------------- exception summary
try:
    exceptions = load_csv("exception_output.csv")
    st.subheader("Exception detection")
    c1, c2, c3 = st.columns(3)
    flagged = int((exceptions.exception_probability >= 0.5).sum())
    c1.metric("Records flagged", "{:,}".format(flagged))
    c2.metric("Share of portfolio", "{:.2%}".format(flagged / max(len(exceptions), 1)))
    by_rule = int((exceptions.decided_by == "rule").sum())
    c3.metric("Resolved deterministically", "{:,}".format(by_rule),
              help="A fired validation rule is a fact about the record and "
                   "outranks any model score.")
except Exception:
    pass

# -------------------------------------------------------- workflow map
st.divider()
st.subheader("Pipeline")
st.caption(
    "Each phase has one runner that regenerates its artifacts from the "
    "data pack. This app reads what they write and trains nothing."
)

st.code(
    "python -m scripts.run_data_intelligence    # profiling, rules, drift\n"
    "python -m scripts.run_transition_engine    # survival + scenarios\n"
    "python -m scripts.run_supervised_models    # 5 prediction targets\n"
    "python -m scripts.run_anomaly_detection    # anomaly + exceptions\n"
    "python -m scripts.run_explainability       # SHAP, errors, uncertainty\n"
    "python -m scripts.run_llm_review           # copilot + governance\n"
    "python -m scripts.build_submission         # submission.csv\n"
    "python -m scripts.build_model_card         # model card\n",
    language="bash",
)

with st.expander("Artifact freshness"):
    st.caption(
        "Pages show results from these files. If one is stale, the page "
        "showing it is stale — this table is how you find out rather than "
        "being misled."
    )
    st.dataframe(artifact_status(), use_container_width=True, hide_index=True)

st.sidebar.title("Navigation")
st.sidebar.caption(
    "Data Quality → Risk Analysis → Loan Explorer → Survival & Transitions "
    "→ Anomaly Detection → Model Performance → Scenario Simulator → "
    "AI Reviewer"
)
st.sidebar.divider()
st.sidebar.caption(SYNTHETIC_BADGE)
