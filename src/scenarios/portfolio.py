"""
Portfolio-level macro scenario simulation, driven by macro_scenarios.csv
(base / adverse_credit / high_prepayment).

Applies each scenario's multipliers/shocks to the scored portfolio,
projects delinquency/default/prepayment rates under each scenario, and
breaks results down by segment (vintage, credit_score_band, state,
servicer_name) -- explicitly required by the judging rubric's Scenario
and Stress Simulation criterion.

Produces the data behind docs/scenario_report.md.

PHASE: 9
STATUS: not yet implemented.
"""


def run_portfolio_scenario(scored_df, scenario_row, segment_columns):
    raise NotImplementedError("Step 9: implement portfolio.run_portfolio_scenario")


def run_all_scenarios(scored_df, macro_scenarios_df, segment_columns):
    raise NotImplementedError("Step 9: implement portfolio.run_all_scenarios")
