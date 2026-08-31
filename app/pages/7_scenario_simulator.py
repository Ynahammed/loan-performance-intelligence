"""Scenario Simulator page: portfolio stress and segment impact. PHASE: 11"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.data_access import (  # noqa: E402
    DATA_DIR, load_csv, page_header, require,
)

page_header(
    "Scenario Simulator",
    "Base, adverse-credit and high-prepayment stress applied to the same "
    "transition engine that produces the survival curves.",
    inputs=["scenario_curves.csv", "segment_impacts.csv"],
)

st.info(
    "**These are simulations under stated assumptions, not forecasts.** A "
    "scenario is a multiplicative stress on the transition probabilities; "
    "the engine propagates the assumptions faithfully and does not judge "
    "whether they are plausible.",
    icon="🧪",
)

require("scenario_curves.csv")
curves = load_csv("scenario_curves.csv")

# ------------------------------------------------------- assumptions
macro_path = DATA_DIR / "macro_scenarios.csv"
if macro_path.exists():
    st.subheader("Scenario assumptions")
    st.dataframe(pd.read_csv(macro_path), use_container_width=True,
                 hide_index=True)
    st.caption(
        "Multipliers come from `macro_scenarios.csv`. Stressed transitions "
        "are scaled and the change is absorbed into the stay-in-state cell "
        "so each row still sums to one — renormalising the whole row "
        "instead would dilute the stress back out, which is a subtle way "
        "to make a scenario engine look like it works while doing nothing."
    )

# --------------------------------------------------------- outcomes
final = curves[curves.month == curves.month.max()]
st.subheader("12-month portfolio outcomes")
cols = st.columns(len(final))
base_row = final[final.scenario == "base"]
base_default = float(base_row.cif_default.iloc[0]) if len(base_row) else None
for col, r in zip(cols, final.itertuples()):
    delta = None
    if base_default is not None and r.scenario != "base":
        delta = "{:+.2f} pp".format((r.cif_default - base_default) * 100)
    col.metric(r.scenario.replace("_", " ").title(),
               "{:.2%}".format(r.cif_default), delta,
               help="Cumulative 12-month default incidence")

# ---------------------------------------------------------- curves
metric = st.radio("Curve", ["cif_default", "cif_prepaid", "delinquency_share"],
                  horizontal=True,
                  format_func=lambda k: {
                      "cif_default": "Cumulative default",
                      "cif_prepaid": "Cumulative prepayment",
                      "delinquency_share": "Share delinquent",
                  }[k])
fig = px.line(curves, x="month", y=metric, color="scenario", markers=True)
fig.update_layout(height=380, xaxis_title="months ahead", yaxis_title="probability")
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------- segments
require("segment_impacts.csv")
segments = load_csv("segment_impacts.csv")

st.subheader("Segment exposure")
seg_type = st.selectbox("Segment by", sorted(segments.segment_type.unique()))
sub = segments[segments.segment_type == seg_type]

wide = sub.pivot_table(index=["segment", "n"], columns="scenario",
                       values="default_12m").reset_index()
if "adverse_credit" in wide.columns and "base" in wide.columns:
    wide["uplift_pp"] = ((wide.adverse_credit - wide.base) * 100).round(3)
    wide = wide.sort_values("uplift_pp", ascending=False)

min_n = st.slider("Minimum records in segment", 1, 400, 50)
wide = wide[wide.n >= min_n]
st.dataframe(wide, use_container_width=True, hide_index=True)

if "uplift_pp" in wide.columns and len(wide):
    fig = px.bar(wide.head(15), x="segment", y="uplift_pp")
    fig.update_layout(height=340, xaxis_title="",
                      yaxis_title="adverse-scenario default uplift (pp)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Segments are ranked by how much the adverse-credit scenario "
        "raises their projected default rate. Thin segments are noisy — "
        "the minimum-records filter is there because a 7-loan segment "
        "topping this chart is an artifact, not a finding."
    )


# ---------------------------------------------------------------------------
# Single-loan what-if
# ---------------------------------------------------------------------------

from app.components.data_access import load_panel  # noqa: E402
from src.scenarios.simulator import (  # noqa: E402
    OVERRIDABLE_FIELDS, field_options, load_scoring_models, simulate,
)

st.divider()
st.header("Single-loan what-if")
st.caption(
    "Change an input and re-score the loan through the same fitted "
    "pipeline that produced its original score. This is a model "
    "simulation, not a guarantee about the loan."
)

models = load_scoring_models()
if not models["binary"]:
    st.warning(
        "No persisted models found. Run "
        "`python -m scripts.run_supervised_models` to train and save them."
    )
    st.stop()

panel = load_panel()
loans = sorted(panel.loan_id.unique())
loan_id = st.selectbox("Loan", loans, key="whatif_loan")
history = panel[panel.loan_id == loan_id].sort_values("reporting_month")

month = st.selectbox(
    "Month to score",
    history.reporting_month.dt.strftime("%Y-%m-%d").tolist(),
    index=len(history) - 1,
)
row_index = history.index[
    history.reporting_month.dt.strftime("%Y-%m-%d") == month][0]
current = history.loc[row_index]

st.subheader("Adjust inputs")
overrides = {}
columns = st.columns(3)
for i, (name, spec) in enumerate(OVERRIDABLE_FIELDS.items()):
    if name not in history.columns:
        continue
    col = columns[i % 3]
    value = current[name]
    if spec["kind"] == "number":
        new = col.number_input(
            spec["label"], value=float(value) if pd.notna(value) else 0.0,
            step=spec.get("step", 1.0), key="wf_" + name,
        )
    elif spec["kind"] == "binary":
        new = int(col.checkbox(
            spec["label"], value=bool(value), key="wf_" + name))
    else:
        options = field_options(panel, name)
        current_value = str(value) if pd.notna(value) else options[0]
        idx = options.index(current_value) if current_value in options else 0
        new = col.selectbox(spec["label"], options, index=idx, key="wf_" + name)

    original = float(value) if spec["kind"] in ("number", "binary") and pd.notna(value) else value
    if spec["kind"] in ("number", "binary"):
        if not pd.isna(value) and abs(float(new) - float(value)) > 1e-9:
            overrides[name] = new
    elif str(new) != str(value):
        overrides[name] = new

if not overrides:
    st.info("Change at least one input above to run a simulation.")
    st.stop()

result = simulate(history, row_index, overrides, models)

st.subheader("Effect on the model's estimates")
deltas = result.deltas()
if deltas.empty:
    st.warning("No target could be scored for this record.")
else:
    cols = st.columns(len(deltas))
    for col, r in zip(cols, deltas.itertuples()):
        col.metric(
            r.target.replace("_flag", "").replace("_", " "),
            "{:.2%}".format(r.scenario),
            "{:+.2f} pp".format(r.change_pp),
            delta_color="inverse" if "prepayment" not in r.target else "normal",
        )
    st.dataframe(deltas, use_container_width=True, hide_index=True)

if result.baseline_state and result.scenario_state:
    c1, c2 = st.columns(2)
    c1.metric("Next state, as-is", result.baseline_state[0],
              "{:.1%} confidence".format(result.baseline_state[1]))
    c2.metric("Next state, simulated", result.scenario_state[0],
              "{:.1%} confidence".format(result.scenario_state[1]))

st.subheader("What changed")
st.caption(
    "Derived features are recomputed, not patched. Raising the balance "
    "moves `balance_ratio` and `amortisation_gap` together, because the "
    "override is applied to the raw row and the real feature engineering "
    "is re-run over it."
)
st.dataframe(
    pd.DataFrame(result.changed_features,
                 columns=["field", "before", "after"]),
    use_container_width=True, hide_index=True,
)

for note in result.notes:
    st.caption("Note: " + note)

st.warning(
    "**A model simulation, not a guarantee.** This shows how the model's "
    "estimate responds to a changed input. It is not a prediction about "
    "what will happen to this loan if that change occurs in reality, and "
    "it is a recommendation for review rather than a decision.",
    icon="⚠️",
)
