"""
Cached artifact access for the dashboard.

WHY THE DASHBOARD READS ARTIFACTS RATHER THAN RUNNING THE PIPELINE
------------------------------------------------------------------
Training the four supervised models, the transition engine and the
anomaly detector takes minutes. A dashboard that does that on page load
is unusable, and one that does it on every interaction is worse. So the
pipelines write their results to `models/*.json` and `data/derived/*.csv`,
and the app reads those.

That split has a second benefit worth stating: the dashboard cannot
produce a number the pipeline did not. Anything on screen exists in a
file that a judge can open and check, which is the same property the
LLM layer gets from its grounded context objects.

The cost is that the app can show stale results if the pipelines have not
been rerun. `artifact_status()` surfaces that explicitly -- every page
header shows when its inputs were generated, rather than silently
displaying last week's model.

PHASE: 11
STATUS: implemented.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Which runner produces what, so a missing artifact tells the user how to
# make it rather than just failing.
PRODUCED_BY = {
    "record_quality.csv": "run_data_intelligence",
    "rule_flags.csv": "run_data_intelligence",
    "top_anomalies.csv": "run_data_intelligence",
    "scenario_curves.csv": "run_transition_engine",
    "segment_impacts.csv": "run_transition_engine",
    "transition_artifacts.json": "run_transition_engine",
    "supervised_metrics.json": "run_supervised_models",
    "reviewer_examples.csv": "run_anomaly_detection",
    "exception_output.csv": "run_anomaly_detection",
    "global_importance.csv": "run_explainability",
    "local_examples.csv": "run_explainability",
    "uncertainty.csv": "run_explainability",
    "llm_calls.jsonl": "run_llm_review",
    "submission.csv": "build_submission",
}

SYNTHETIC_BADGE = (
    ":orange-badge[🔧 Synthetic demo data] — this pack was generated to "
    "match the challenge specification. Nothing shown describes real "
    "borrowers or real portfolio behaviour."
)


def _resolve(name: str) -> Path:
    for folder in (DERIVED_DIR, MODELS_DIR, DATA_DIR, LOGS_DIR, PROJECT_ROOT):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return DERIVED_DIR / name


def artifact_age(name: str) -> str:
    path = _resolve(name)
    if not path.exists():
        return "missing"
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )
    hours = age.total_seconds() / 3600
    if hours < 1:
        return "{:.0f} min ago".format(age.total_seconds() / 60)
    if hours < 48:
        return "{:.0f} h ago".format(hours)
    return "{:.0f} days ago".format(hours / 24)


def require(name: str):
    """Stop the page with a runnable instruction if an input is missing."""
    path = _resolve(name)
    if path.exists():
        return path
    runner = PRODUCED_BY.get(name)
    st.warning(
        "`{}` has not been generated yet.\n\n"
        "Run `python -m scripts.{}` to produce it.".format(
            name, runner or "<the relevant pipeline>")
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_resolve(name))


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads(_resolve(name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_jsonl(name: str) -> pd.DataFrame:
    path = _resolve(name)
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_markdown(name: str) -> str:
    path = DOCS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


@st.cache_data(show_spinner=False)
def load_panel(sample: int = None) -> pd.DataFrame:
    """The training panel WITH origination attributes joined on.

    The join is not optional. Every pipeline calls
    `attach_static_attributes` before engineering features, so the models
    were fitted on a frame that includes `vintage` and
    `original_term_months`. Loading the raw panel without them makes the
    what-if simulator silently skip three of its four targets: the models
    that need those columns are dropped, and only the one champion that
    happens to be a logistic baseline on raw columns still scores. Nothing
    errors -- the page just quietly shows one row instead of four.
    """
    df = pd.read_csv(
        DATA_DIR / "loan_monthly_performance_train.csv",
        parse_dates=["reporting_month", "origination_month"],
    )
    static_path = DATA_DIR / "loan_static_attributes.csv"
    if static_path.exists():
        static = pd.read_csv(static_path, parse_dates=["origination_month"])
        new_cols = [c for c in static.columns
                    if c == "loan_id" or c not in df.columns]
        before = len(df)
        df = df.merge(static[new_cols], on="loan_id", how="left", validate="m:1")
        assert len(df) == before, "static join changed the row count"
    if sample and len(df) > sample:
        return df.sample(sample, random_state=42)
    return df


def artifact_status() -> pd.DataFrame:
    """What exists, when it was made, and what makes it."""
    rows = []
    for name, runner in PRODUCED_BY.items():
        path = _resolve(name)
        rows.append({
            "artifact": name,
            "status": "ok" if path.exists() else "missing",
            "generated": artifact_age(name),
            "produced by": "scripts." + runner,
        })
    return pd.DataFrame(rows)


def page_header(title: str, description: str, inputs: list = None) -> None:
    """Standard page header, with provenance for what is displayed."""
    st.title(title)
    st.caption(description)
    if inputs:
        stamps = ["`{}` {}".format(name, artifact_age(name)) for name in inputs]
        st.caption("Showing results generated: " + " · ".join(stamps))
    st.caption(SYNTHETIC_BADGE)
