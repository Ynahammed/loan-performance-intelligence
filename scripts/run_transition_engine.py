"""
End-to-end run of the discrete-time transition / survival engine.

    python -m scripts.run_transition_engine

Produces:
    models/transition_model.joblib      fitted covariate model + baseline
    models/transition_artifacts.json    metrics snapshot for the dashboard
    docs/survival_report.md             the human-readable writeup
    data/derived/scenario_curves.csv    per-scenario portfolio curves
    data/derived/segment_impacts.csv    segment-level scenario deltas

Everything printed here is computed, not asserted. If a number in the
report looks wrong, it is wrong -- rerun and read the traceback.

PHASE: 6
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    configure_console,
    ABSORBING_STATES,
    DATA_DIR,
    DOCS_DIR,
    LOAN_ID_COLUMN,
    MODELS_DIR,
    STATE_COLUMN,
    STATE_ORDER,
    TIME_COLUMN,
)
from src.data.loader import (  # noqa: E402
    attach_static_attributes,
    load_data_pack,
    sort_panel,
)
from src.models.survival import (  # noqa: E402
    DiscreteTimeTransitionModel,
    EmpiricalTransitionBaseline,
    aalen_johansen_cif,
    add_balance_ratio,
    apply_scenario_to_matrix,
    build_time_to_event,
    build_transition_frame,
    compare_to_baseline,
    empirical_transition_counts,
    evaluate_horizon_against_label,
    evaluate_one_step,
    simulate_paths,
)
from src.models.validation import time_aware_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("transition_engine")

DERIVED_DIR = DATA_DIR / "derived"
HORIZON = 12

# The 12-month cross-check needs an earlier training window than the
# one-step evaluation. See the note written into the report.
CROSSCHECK_TRAIN_END = "2021-06-01"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    configure_console()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report: list = []

    def emit(text: str = "") -> None:
        print(text)
        report.append(text)

    # ---------------------------------------------------------------- load
    section("1. Data")
    pack = load_data_pack(DATA_DIR)
    print(pack.summary().to_string(index=False))

    panel = attach_static_attributes(pack.train, pack.static)
    panel = add_balance_ratio(sort_panel(panel))
    emit("Panel: {:,} rows, {:,} loans, {} to {}".format(
        len(panel),
        panel[LOAN_ID_COLUMN].nunique(),
        panel[TIME_COLUMN].min().date(),
        panel[TIME_COLUMN].max().date(),
    ))

    # ------------------------------------------------- transition frame
    section("2. Transition frame (censoring handled here)")
    tf = build_transition_frame(panel)
    print(tf.describe())
    emit()
    emit("### Transition frame construction")
    emit("```")
    emit(tf.describe())
    emit("```")

    counts = empirical_transition_counts(tf.data)
    rates = (counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(2)
    print("\nObserved one-month transition rates (%):")
    print(rates.to_string())
    emit()
    emit("### Observed one-month transition rates (%)")
    emit("```")
    emit(rates.to_string())
    emit("```")

    unobserved = [s for s in STATE_ORDER if counts.loc[s].sum() == 0]
    if unobserved:
        emit()
        emit("Origin states with no observed outgoing transitions in the "
             "training panel: {}. The state machine still represents them; "
             "they fall back to the smoothed empirical row.".format(unobserved))

    # ------------------------------------------------------------- split
    section("3. Purged temporal split (one-step transition target)")
    split = time_aware_split(tf.data, purge_months=1, test_fraction=0.2)
    print(split.describe())
    emit()
    emit("### Validation split")
    emit("```")
    emit(split.describe())
    emit("```")

    # ------------------------------------------------------------- fit
    section("4. Fit baseline and covariate model")
    # Inner temporal split of the training window: fit on the earlier part,
    # choose the per-origin credibility weights on the later part. The outer
    # test set stays untouched by both.
    inner = time_aware_split(split.train, purge_months=1, test_fraction=0.25)
    # The probe sees less data than the final model, so its origin-state
    # threshold is scaled down by the same ratio. Otherwise an origin that
    # WILL get a covariate model after the refit never has an alpha measured
    # for it, and silently inherits the default.
    probe_threshold = max(
        30,
        int(DiscreteTimeTransitionModel().min_rows_per_origin
            * len(inner.train) / max(len(split.train), 1)),
    )
    probe = DiscreteTimeTransitionModel(
        min_rows_per_origin=probe_threshold
    ).fit(inner.train)
    blend = probe.calibrate_blend(inner.test)

    # Alpha is a hyperparameter, so it is chosen on the inner calibration
    # slice and the models are then refit on the FULL training window --
    # otherwise a quarter of the training data is spent on model selection
    # and never recovered, which on a panel this small pushes thin origin
    # states below the threshold for a covariate model at all.
    baseline = EmpiricalTransitionBaseline().fit(split.train)
    model = DiscreteTimeTransitionModel().fit(split.train)
    model.blend_weights_.update(probe.blend_weights_)

    print("Inner fit/calibration split: {:,} fit rows, {:,} calibration rows".format(
        len(inner.train), len(inner.test)))
    print(model.fit_summary().to_string(index=False))
    print("\nCredibility blend selection (log-loss on the calibration slice):")
    print(blend.to_string(index=False))
    emit()
    emit("### Per-origin model selection")
    emit("```")
    emit(model.fit_summary().to_string(index=False))
    emit("```")
    emit("The covariate model is shrunk toward the empirical row per origin "
         "state, with the weight `alpha` chosen on a held-out calibration "
         "slice by log-loss. Origins with thin data earn less trust in their "
         "covariates -- credibility weighting, chosen by measurement rather "
         "than by assertion.")
    emit("```")
    emit(blend.to_string(index=False))
    emit("```")

    # -------------------------------------------------------- evaluate
    section("5. One-step evaluation vs baseline")
    comparison = compare_to_baseline(model, baseline, split.test)
    print(comparison.round(4).to_string())
    emit()
    emit("### One-step-ahead performance (held-out, purged)")
    emit("```")
    emit(comparison.round(4).to_string())
    emit("```")

    # ------------------------------------------------- forward simulation
    section("6. 12-month forward simulation from the panel edge")
    as_of = panel.groupby(LOAN_ID_COLUMN).tail(1).reset_index(drop=True)
    active = as_of[~as_of[STATE_COLUMN].isin(ABSORBING_STATES)].reset_index(drop=True)
    emit()
    emit("### Forward projection")
    emit("Projected {:,} loans still active at the panel edge "
         "({:,} already absorbed and excluded).".format(len(active), len(as_of) - len(active)))

    sim = simulate_paths(model, active, active[STATE_COLUMN], horizon=HORIZON)
    base_default = sim.portfolio_curve("Default")
    base_prepay = sim.portfolio_curve("Prepaid")
    base_delinq = sim.delinquency_curve()

    curve_tbl = pd.DataFrame(
        {
            "month": np.arange(HORIZON + 1),
            "cif_default": base_default.round(5),
            "cif_prepaid": base_prepay.round(5),
            "delinquency_share": base_delinq.round(5),
        }
    )
    print(curve_tbl.to_string(index=False))
    emit("```")
    emit(curve_tbl.to_string(index=False))
    emit("```")
    emit("`cif_default` and `cif_prepaid` are true cumulative incidence "
         "functions: both events are absorbing in one chain, so the "
         "competition between them is accounted for by construction.")

    # -------------------------------------------- competing-risk baseline
    section("7. Aalen-Johansen baseline vs naive 1 - KM")
    tte = build_time_to_event(panel)
    print(tte.event_code.value_counts().rename({0: "censored", 1: "default", 2: "prepaid"}).to_string())
    aj = aalen_johansen_cif(tte, event_of_interest=1)
    emit()
    emit("### Competing risks: Aalen-Johansen vs naive 1 - Kaplan-Meier")
    emit("```")
    emit(tte.event_code.value_counts()
         .rename({0: "censored", 1: "default", 2: "prepaid"}).to_string())
    if len(aj):
        snapshot = aj.iloc[:: max(1, len(aj) // 10)].round(5)
        print(snapshot.to_string())
        emit(snapshot.to_string())
        final = aj.iloc[-1]
        overstate = final["naive_1_minus_km"] / max(final["aalen_johansen_cif"], 1e-9)
        emit("```")
        emit("At the end of the observation window the naive 1 - KM curve "
             "reads {:.4f} against the Aalen-Johansen CIF of {:.4f} -- an "
             "overstatement of {:.1f}x, caused by treating prepayment as "
             "censoring rather than as a competing event.".format(
                 final["naive_1_minus_km"], final["aalen_johansen_cif"], overstate))
    else:
        emit("```")

    # ------------------------------------------------------ cross-check
    section("8. Cross-check: chain-implied 12m default vs next_12m_default_flag")
    panel_max = panel[TIME_COLUMN].max()
    mature_cutoff = panel_max - pd.DateOffset(months=HORIZON)

    cc_split = time_aware_split(
        tf.data, purge_months=HORIZON, train_end=CROSSCHECK_TRAIN_END
    )
    cc_model = DiscreteTimeTransitionModel().fit(cc_split.train)

    eval_rows = cc_split.test[
        (cc_split.test[TIME_COLUMN] <= mature_cutoff)
        & (~cc_split.test[STATE_COLUMN].isin(ABSORBING_STATES))
        & (cc_split.test["next_12m_default_flag"].notna())
    ].reset_index(drop=True)

    emit()
    emit("### Cross-check against the supervised 12-month label")
    emit("A purged evaluation of a 12-month label needs 12 clean months "
         "between train and test AND 12 months of panel left for the label "
         "to mature. On a {}-month panel that forces an earlier, smaller "
         "training window than the one-step evaluation uses -- train ends "
         "{}, evaluation rows run to {}.".format(
             panel[TIME_COLUMN].dt.to_period("M").nunique(),
             pd.Timestamp(CROSSCHECK_TRAIN_END).date(),
             mature_cutoff.date()))

    if len(eval_rows) == 0:
        emit("No rows survive both constraints; cross-check skipped.")
        cc_metrics = {}
    else:
        cc_sim = simulate_paths(
            cc_model, eval_rows, eval_rows[STATE_COLUMN], horizon=HORIZON
        )
        p_default = cc_sim.terminal_probability("Default")
        cc_metrics = evaluate_horizon_against_label(
            p_default, eval_rows["next_12m_default_flag"].to_numpy()
        )
        cal = cc_metrics.pop("calibration")
        print(json.dumps(cc_metrics, indent=2))
        print(cal.round(5).to_string(index=False))
        emit("```")
        emit(json.dumps(cc_metrics, indent=2))
        emit()
        emit("Calibration by predicted-probability decile:")
        emit(cal.round(5).to_string(index=False))
        emit("```")

    # --------------------------------------------------------- scenarios
    section("9. Scenarios")
    macro = pd.read_csv(DATA_DIR / "macro_scenarios.csv")
    print(macro.to_string(index=False))

    all_curves = []
    terminal = {}
    for _, row in macro.iterrows():
        name = row["scenario_name"]
        mults = row.to_dict()
        s = simulate_paths(
            model, active, active[STATE_COLUMN], horizon=HORIZON,
            scenario_multipliers=mults, scenario_name=name,
        )
        all_curves.append(
            pd.DataFrame(
                {
                    "scenario": name,
                    "month": np.arange(HORIZON + 1),
                    "cif_default": s.portfolio_curve("Default"),
                    "cif_prepaid": s.portfolio_curve("Prepaid"),
                    "delinquency_share": s.delinquency_curve(),
                }
            )
        )
        terminal[name] = {
            "default_12m": float(s.terminal_probability("Default").mean()),
            "prepaid_12m": float(s.terminal_probability("Prepaid").mean()),
            "delinquency_12m": float(s.delinquency_curve()[-1]),
            "_per_loan_default": s.terminal_probability("Default"),
            "_per_loan_prepaid": s.terminal_probability("Prepaid"),
        }

    curves = pd.concat(all_curves, ignore_index=True)
    curves.to_csv(DERIVED_DIR / "scenario_curves.csv", index=False)

    summary = pd.DataFrame(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         for k, v in terminal.items()}
    ).T.round(5)
    print("\n12-month portfolio outcomes by scenario:")
    print(summary.to_string())
    emit()
    emit("### Scenario projections (12-month horizon)")
    emit("```")
    emit(summary.to_string())
    emit("```")

    # ------------------------------------------------- segment breakdown
    section("10. Segment-level scenario impacts")
    seg_cols = [c for c in ("vintage", "credit_score_band", "state", "servicer_name")
                if c in active.columns]
    seg_rows = []
    for col in seg_cols:
        for name, vals in terminal.items():
            g = pd.DataFrame({
                "segment_type": col,
                "segment": active[col].to_numpy(),
                "scenario": name,
                "default_12m": vals["_per_loan_default"],
                "prepaid_12m": vals["_per_loan_prepaid"],
            })
            seg_rows.append(g)
    seg = pd.concat(seg_rows, ignore_index=True)
    seg_agg = (seg.groupby(["segment_type", "segment", "scenario"])
                  .agg(n=("default_12m", "size"),
                       default_12m=("default_12m", "mean"),
                       prepaid_12m=("prepaid_12m", "mean"))
                  .reset_index())
    seg_agg.to_csv(DERIVED_DIR / "segment_impacts.csv", index=False)

    wide = seg_agg.pivot_table(
        index=["segment_type", "segment", "n"], columns="scenario", values="default_12m"
    )
    if "adverse_credit" in wide.columns and "base" in wide.columns:
        wide["uplift_pp"] = (wide["adverse_credit"] - wide["base"]) * 100
        top = wide.sort_values("uplift_pp", ascending=False).head(12).round(5)
        print("Largest adverse-scenario default uplift by segment:")
        print(top.to_string())
        emit()
        emit("### Segments most exposed to the adverse-credit scenario")
        emit("```")
        emit(top.to_string())
        emit("```")

    # ---------------------------------------------------------- persist
    section("11. Persist")
    joblib.dump(
        {"model": model, "baseline": baseline, "states": STATE_ORDER,
         "absorbing": ABSORBING_STATES, "horizon": HORIZON},
        MODELS_DIR / "transition_model.joblib",
    )
    artifacts = {
        "one_step_comparison": comparison.round(6).to_dict(),
        "scenario_summary": summary.to_dict(),
        "crosscheck_12m_default": cc_metrics,
        "transition_frame": {
            "input_rows": tf.n_input_rows,
            "terminal_rows_dropped": tf.n_terminal_rows_dropped,
            "gap_rows_dropped": tf.n_gap_rows_dropped,
            "label_disagreements": tf.label_disagreements,
            "usable": len(tf.data),
        },
    }
    (MODELS_DIR / "transition_artifacts.json").write_text(
        json.dumps(artifacts, indent=2, default=str), encoding="utf-8"
    )

    header = [
        "# Survival / Transition Model Report",
        "",
        "Generated by `scripts/run_transition_engine.py`. Every number below "
        "is computed at run time from the data pack in `data/`.",
        "",
    ]
    (DOCS_DIR / "survival_report.md").write_text(
        "\n".join(header + report) + "\n", encoding="utf-8"
    )
    print("\nWrote models/transition_model.joblib")
    print("Wrote models/transition_artifacts.json")
    print("Wrote docs/survival_report.md")
    print("Wrote data/derived/scenario_curves.csv")
    print("Wrote data/derived/segment_impacts.csv")


if __name__ == "__main__":
    main()
