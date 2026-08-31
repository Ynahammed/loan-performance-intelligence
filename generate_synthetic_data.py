"""
Synthetic Loan Performance Data Pack Generator
================================================
Generates a data pack SHAPED to the Intain Campus FinTech Challenge 2026
schema (loan_monthly_performance_train/test.csv, loan_static_attributes.csv,
servicer_updates.csv, validation_rules.json, macro_scenarios.csv,
data_dictionary.md, submission_template.csv).

THIS IS 100% SYNTHETIC DATA. It is NOT real loan performance data and must
never be presented as such. It exists only as a drop-in development
foundation until the real organizer-provided data pack is available.

Design choices are documented inline so the "why" is auditable later for
the AI Development Log.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_LOANS = 2000
ORIGINATION_START = pd.Timestamp("2019-01-01")
ORIGINATION_END = pd.Timestamp("2024-03-01")
CUTOFF_DATE = pd.Timestamp("2024-12-01")      # last month we simulate internally
TRAIN_CUTOFF = pd.Timestamp("2023-12-01")     # rows <= this -> train (labeled)
TEST_END = pd.Timestamp("2024-06-01")         # rows in (TRAIN_CUTOFF, TEST_END] -> test (unlabeled)

OUT_DIR = Path("/home/claude/data_pack")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# REFERENCE DATA / ORDINAL MAPS
# ----------------------------------------------------------------------
CREDIT_BANDS = ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]
CREDIT_ORD = {b: i for i, b in enumerate(CREDIT_BANDS)}
CREDIT_WEIGHTS = [0.08, 0.12, 0.18, 0.24, 0.22, 0.16]

LTV_BANDS = ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"]
LTV_ORD = {b: i for i, b in enumerate(LTV_BANDS)}
LTV_WEIGHTS = [0.12, 0.18, 0.28, 0.24, 0.12, 0.06]

DTI_BANDS = ["<=20", "21-30", "31-36", "37-43", "44-50", ">50"]
DTI_ORD = {b: i for i, b in enumerate(DTI_BANDS)}
DTI_WEIGHTS = [0.10, 0.22, 0.24, 0.22, 0.14, 0.08]

STATES = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "NC", "PA", "MI", "WA", "AZ"]
STATE_WEIGHTS = [0.16, 0.13, 0.11, 0.10, 0.07, 0.06, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06]

LOAN_PURPOSE = ["Purchase", "Refinance", "Cash-Out Refinance"]
LOAN_PURPOSE_W = [0.55, 0.25, 0.20]

PROPERTY_TYPE = ["Single Family", "Condo", "Multi-Family", "Manufactured"]
PROPERTY_TYPE_W = [0.68, 0.18, 0.09, 0.05]

OCCUPANCY_TYPE = ["Primary", "Second Home", "Investment"]
OCCUPANCY_TYPE_W = [0.78, 0.09, 0.13]

SERVICERS = ["Sunrise Servicing", "Meridian Loan Services", "Apex Mortgage Co",
             "Highland Capital Servicing", "Cornerstone Servicing"]

SOURCE_SYSTEMS = ["LOS_A", "LOS_B", "Core"]

STATES_LOWER_VARIANTS = {  # for injecting inconsistent casing
    "CA": ["ca", "California"], "TX": ["tx", "Texas"], "NY": ["ny", "New York"]
}

# Monthly macro path spanning the whole simulation window (2019-01 .. 2024-12).
# Deliberately shaped so 2019-2021 is a low-rate regime, 2022-2023 a rising-rate
# regime, and 2024 a plateau -- this creates REAL train/test distribution
# drift for the drift-detection task, not just noise.
ALL_MONTHS = pd.date_range(ORIGINATION_START, CUTOFF_DATE, freq="MS")
n_months_total = len(ALL_MONTHS)


def build_macro_path():
    rate = np.zeros(n_months_total)
    stress = np.zeros(n_months_total)
    r, s = 3.2, 0.15
    for i, m in enumerate(ALL_MONTHS):
        if m.year <= 2021:
            r += rng.normal(0.0, 0.03)
            s += rng.normal(0.0, 0.01)
        elif m.year in (2022, 2023):
            r += rng.normal(0.08, 0.05)   # rising-rate regime
            s += rng.normal(0.02, 0.02)   # rising stress
        else:  # 2024 plateau / mild relief
            r += rng.normal(-0.02, 0.04)
            s += rng.normal(-0.01, 0.015)
        r = float(np.clip(r, 2.5, 8.0))
        s = float(np.clip(s, 0.0, 1.0))
        rate[i] = r
        stress[i] = s
    return dict(zip(ALL_MONTHS, rate)), dict(zip(ALL_MONTHS, stress))


MARKET_RATE, MACRO_STRESS = build_macro_path()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# 1. STATIC ATTRIBUTES
# ----------------------------------------------------------------------
def gen_static_attributes():
    loan_ids = [f"LN{100000 + i}" for i in range(N_LOANS)]
    origination_days = rng.integers(0, (ORIGINATION_END - ORIGINATION_START).days, N_LOANS)
    origination_month = [
        (ORIGINATION_START + pd.Timedelta(days=int(d))).to_period("M").to_timestamp()
        for d in origination_days
    ]

    credit_score_band = rng.choice(CREDIT_BANDS, N_LOANS, p=CREDIT_WEIGHTS)
    ltv_band = rng.choice(LTV_BANDS, N_LOANS, p=LTV_WEIGHTS)
    dti_band = rng.choice(DTI_BANDS, N_LOANS, p=DTI_WEIGHTS)
    state = rng.choice(STATES, N_LOANS, p=STATE_WEIGHTS)
    loan_purpose = rng.choice(LOAN_PURPOSE, N_LOANS, p=LOAN_PURPOSE_W)
    property_type = rng.choice(PROPERTY_TYPE, N_LOANS, p=PROPERTY_TYPE_W)
    occupancy_type = rng.choice(OCCUPANCY_TYPE, N_LOANS, p=OCCUPANCY_TYPE_W)
    servicer_name = rng.choice(SERVICERS, N_LOANS)

    original_balance = np.round(np.exp(rng.normal(12.35, 0.45, N_LOANS)), -2)  # ~ lognormal ~230k
    original_balance = np.clip(original_balance, 60000, 900000)

    term_months = rng.choice([180, 360], N_LOANS, p=[0.22, 0.78])

    credit_ord = np.array([CREDIT_ORD[b] for b in credit_score_band])
    ltv_ord = np.array([LTV_ORD[b] for b in ltv_band])

    orig_month_rate = np.array([MARKET_RATE[m] for m in origination_month])
    interest_rate = (
        orig_month_rate
        + (5 - credit_ord) * 0.30
        + ltv_ord * 0.08
        + rng.normal(0, 0.20, N_LOANS)
    )
    interest_rate = np.clip(interest_rate, 2.5, 9.75)

    df = pd.DataFrame({
        "loan_id": loan_ids,
        "origination_month": origination_month,
        "original_balance": original_balance,
        "original_term_months": term_months,
        "original_interest_rate": np.round(interest_rate, 3),
        "credit_score_band": credit_score_band,
        "ltv_band": ltv_band,
        "dti_band": dti_band,
        "state": state,
        "loan_purpose": loan_purpose,
        "property_type": property_type,
        "occupancy_type": occupancy_type,
        "servicer_name": servicer_name,
        "vintage": [m.year for m in origination_month],
    })
    return df


static_df = gen_static_attributes()

# ----------------------------------------------------------------------
# 2. MONTHLY PANEL SIMULATION (state-transition process)
# ----------------------------------------------------------------------
STATE_CURRENT, STATE_30, STATE_60, STATE_90, STATE_DEFAULT, STATE_PREPAID, STATE_MATURED = (
    "Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Matured"
)
DPD_MAP = {STATE_CURRENT: 0, STATE_30: 30, STATE_60: 60, STATE_90: 95,
           STATE_DEFAULT: 150, STATE_PREPAID: 0, STATE_MATURED: 0}


def monthly_transition(state, credit_ord, ltv_ord, dti_ord, loan_age, mkt_rate,
                        loan_rate, stress):
    """Return next state for one loan-month given current state + covariates."""
    age_factor = np.exp(-((loan_age - 30) ** 2) / (2 * 20 ** 2))  # hazard peaks ~month 30
    refi_incentive = max(0.0, (loan_rate - mkt_rate) - 0.5)

    risk_score = (credit_ord * -0.35 + ltv_ord * 0.30 + dti_ord * 0.30
                  + stress * 2.2 + age_factor * 0.9)

    if state == STATE_CURRENT:
        p_delinquent = sigmoid(risk_score - 4.0) * 0.10
        p_prepay = sigmoid(refi_incentive * 3 - 2.5) * 0.06 + (0.002 if loan_age > 12 else 0.0)
        p_stay = max(0.0, 1 - p_delinquent - p_prepay)
        return rng.choice([STATE_30, STATE_PREPAID, STATE_CURRENT],
                           p=_norm([p_delinquent, p_prepay, p_stay]))
    if state == STATE_30:
        p_cure = 0.42
        p_progress = sigmoid(risk_score - 3.0) * 0.35
        p_prepay = 0.02
        p_stay = max(0.0, 1 - p_cure - p_progress - p_prepay)
        return rng.choice([STATE_CURRENT, STATE_60, STATE_PREPAID, STATE_30],
                           p=_norm([p_cure, p_progress, p_prepay, p_stay]))
    if state == STATE_60:
        p_cure = 0.22
        p_progress = sigmoid(risk_score - 2.5) * 0.40
        p_stay = max(0.0, 1 - p_cure - p_progress)
        return rng.choice([STATE_30, STATE_90, STATE_60],
                           p=_norm([p_cure, p_progress, p_stay]))
    if state == STATE_90:
        p_cure = 0.10
        p_progress = sigmoid(risk_score - 2.0) * 0.45
        p_stay = max(0.0, 1 - p_cure - p_progress)
        return rng.choice([STATE_60, STATE_DEFAULT, STATE_90],
                           p=_norm([p_cure, p_progress, p_stay]))
    return state  # Default / Prepaid / Matured are absorbing


def _norm(p):
    p = np.clip(np.array(p, dtype=float), 0, None)
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)


rows = []
loss_severity_bands = ["Low(<10%)", "Moderate(10-25%)", "High(25-40%)", "Severe(>40%)"]

for _, loan in static_df.iterrows():
    loan_id = loan["loan_id"]
    credit_ord = CREDIT_ORD[loan["credit_score_band"]]
    ltv_ord = LTV_ORD[loan["ltv_band"]]
    dti_ord = DTI_ORD[loan["dti_band"]]
    orig_month = loan["origination_month"]
    term = loan["original_term_months"]
    loan_rate = loan["original_interest_rate"]
    balance = loan["original_balance"]

    state = STATE_CURRENT
    month_index = 0
    cur_month = orig_month
    modification_flag = 0

    while cur_month <= CUTOFF_DATE and month_index < term:
        loan_age = month_index
        mkt_rate = MARKET_RATE.get(cur_month, loan_rate)
        stress = MACRO_STRESS.get(cur_month, 0.15)

        # amortize / adjust balance
        if state == STATE_CURRENT:
            monthly_principal_frac = 1.0 / (term - month_index + 1)
            balance = max(0.0, balance * (1 - monthly_principal_frac * 0.55))
        elif state in (STATE_30, STATE_60, STATE_90):
            balance = balance  # payments paused while delinquent (simplified)

        days_past_due = DPD_MAP[state]
        remaining_term = max(0, term - month_index)

        # rare modification event when deep delinquent
        if state == STATE_90 and modification_flag == 0 and rng.random() < 0.04:
            modification_flag = 1
            loan_rate = max(2.5, loan_rate - 0.75)

        document_status = rng.choice(
            ["Complete", "Pending", "Missing"],
            p=[0.80, 0.14, 0.06] if state == STATE_CURRENT else [0.55, 0.30, 0.15]
        )
        source_system = rng.choice(SOURCE_SYSTEMS, p=[0.5, 0.3, 0.2])
        last_updated_at = cur_month + pd.Timedelta(days=int(rng.integers(0, 10)))

        rows.append({
            "loan_id": loan_id,
            "month_index": month_index,
            "reporting_month": cur_month,
            "origination_month": orig_month,
            "loan_age_months": loan_age,
            "remaining_term_months": remaining_term,
            "original_balance": loan["original_balance"],
            "current_balance": round(balance, 2),
            "interest_rate": round(loan_rate, 3),
            "credit_score_band": loan["credit_score_band"],
            "ltv_band": loan["ltv_band"],
            "dti_band": loan["dti_band"],
            "state": loan["state"],
            "loan_purpose": loan["loan_purpose"],
            "occupancy_type": loan["occupancy_type"],
            "property_type": loan["property_type"],
            "servicer_name": loan["servicer_name"],
            "current_status": state,
            "days_past_due": days_past_due,
            "modification_flag": modification_flag,
            "prepayment_flag": 1 if state == STATE_PREPAID else 0,
            "default_flag": 1 if state == STATE_DEFAULT else 0,
            "loss_severity_band": rng.choice(loss_severity_bands) if state == STATE_DEFAULT else None,
            "last_updated_at": last_updated_at,
            "source_system": source_system,
            "document_status": document_status,
        })

        if state in (STATE_DEFAULT, STATE_PREPAID):
            break  # absorbing, loan exits panel

        next_state = monthly_transition(state, credit_ord, ltv_ord, dti_ord,
                                         loan_age, mkt_rate, loan_rate, stress)
        state = next_state
        month_index += 1
        cur_month = orig_month + pd.DateOffset(months=month_index)

    else:
        pass  # loop exhausted naturally (matured or hit cutoff)

panel = pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 3. FORWARD-LOOKING LABELS (computed per loan_id using its own trajectory)
# ----------------------------------------------------------------------
panel = panel.sort_values(["loan_id", "month_index"]).reset_index(drop=True)

label_frames = []
for loan_id, g in panel.groupby("loan_id", sort=False):
    g = g.reset_index(drop=True)
    states = g["current_status"].tolist()
    n = len(g)
    next_3m, next_6m, next_12m_def, next_12m_prepay, next_state_col = [], [], [], [], []
    for i in range(n):
        window3 = states[i + 1:i + 4]
        window6 = states[i + 1:i + 7]
        window12 = states[i + 1:i + 13]
        next_3m.append(int(any(s in (STATE_30, STATE_60, STATE_90, STATE_DEFAULT) for s in window3)))
        next_6m.append(int(any(s in (STATE_30, STATE_60, STATE_90, STATE_DEFAULT) for s in window6)))
        next_12m_def.append(int(STATE_DEFAULT in window12))
        next_12m_prepay.append(int(STATE_PREPAID in window12))
        next_state_col.append(states[i + 1] if i + 1 < n else states[i])
    label_frames.append(pd.DataFrame({
        "loan_id": loan_id,
        "month_index": g["month_index"],
        "next_3m_delinquency_flag": next_3m,
        "next_6m_delinquency_flag": next_6m,
        "next_12m_default_flag": next_12m_def,
        "next_12m_prepayment_flag": next_12m_prepay,
        "next_state": next_state_col,
    }))

labels = pd.concat(label_frames, ignore_index=True)
panel = panel.merge(labels, on=["loan_id", "month_index"], how="left")

# ----------------------------------------------------------------------
# 4. EXCEPTION / DATA-QUALITY LABELS (independent of performance labels)
# ----------------------------------------------------------------------
n_rows = len(panel)
exception_type = np.full(n_rows, "none", dtype=object)
exception_required = np.zeros(n_rows, dtype=int)

# a) status/DPD inconsistency
mask = (
    ((panel["current_status"] == STATE_CURRENT) & (panel["days_past_due"] > 0)) |
    ((panel["current_status"] == STATE_90) & (panel["days_past_due"] < 60))
)
exception_type[mask.values] = "delinquency_inconsistency"

# b) document gap for closed/prepaid loans
mask = (panel["current_status"].isin([STATE_DEFAULT, STATE_PREPAID])) & (panel["document_status"] == "Missing")
exception_type[mask.values] = "document_gap"

# c) random balance inconsistency injection (~1.2%)
inject_idx = rng.choice(n_rows, size=int(n_rows * 0.012), replace=False)
panel.loc[inject_idx, "current_balance"] = panel.loc[inject_idx, "current_balance"] * rng.uniform(2.5, 4.0)
exception_type[inject_idx] = "balance_inconsistency"

# d) invalid date injection (~0.3%)
inject_idx2 = rng.choice(n_rows, size=int(n_rows * 0.003), replace=False)
panel.loc[inject_idx2, "reporting_month"] = panel.loc[inject_idx2, "origination_month"] - pd.DateOffset(months=2)
exception_type[inject_idx2] = "date_invalid"

exception_required = (exception_type != "none").astype(int)
panel["exception_required"] = exception_required
panel["exception_type"] = exception_type

# ----------------------------------------------------------------------
# 5. GENERAL MESSINESS INJECTION (missingness, duplicates, casing, outliers)
# ----------------------------------------------------------------------
def inject_missing(df, col, frac, condition_mask=None):
    idx = df.index if condition_mask is None else df.index[condition_mask]
    n = int(len(idx) * frac)
    if n == 0:
        return
    sel = rng.choice(idx, size=n, replace=False)
    df.loc[sel, col] = np.nan


inject_missing(panel, "dti_band", 0.04)
inject_missing(panel, "document_status", 0.03)
inject_missing(panel, "servicer_name", 0.01)
inject_missing(panel, "loss_severity_band", 0.02, condition_mask=(panel["current_status"] == STATE_DEFAULT).values)

# inconsistent state casing for a subset of rows (CA/TX/NY variants)
for canon, variants in STATES_LOWER_VARIANTS.items():
    mask = panel["state"] == canon
    idx = panel.index[mask]
    n = int(len(idx) * 0.10)
    if n > 0:
        sel = rng.choice(idx, size=n, replace=False)
        panel.loc[sel, "state"] = rng.choice(variants, size=n)

# duplicate rows (~0.3%)
dup_idx = rng.choice(panel.index, size=int(len(panel) * 0.003), replace=False)
panel = pd.concat([panel, panel.loc[dup_idx]], ignore_index=True)

# a handful of contradictory status-reversal rows (Prepaid -> Current next month)
reversal_candidates = panel[panel["current_status"] == STATE_PREPAID].sample(
    n=min(8, (panel["current_status"] == STATE_PREPAID).sum()), random_state=1
)
for _, r in reversal_candidates.iterrows():
    bad_row = r.copy()
    bad_row["month_index"] = r["month_index"] + 1
    bad_row["reporting_month"] = r["reporting_month"] + pd.DateOffset(months=1)
    bad_row["current_status"] = STATE_CURRENT
    bad_row["days_past_due"] = 0
    bad_row["exception_type"] = "status_reversal"
    bad_row["exception_required"] = 1
    panel = pd.concat([panel, pd.DataFrame([bad_row])], ignore_index=True)

panel = panel.sort_values(["loan_id", "month_index"]).reset_index(drop=True)

# ----------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT (chronological, per real data pack convention)
# ----------------------------------------------------------------------
train_df = panel[panel["reporting_month"] <= TRAIN_CUTOFF].copy()
test_df = panel[(panel["reporting_month"] > TRAIN_CUTOFF) & (panel["reporting_month"] <= TEST_END)].copy()

label_cols = ["next_3m_delinquency_flag", "next_6m_delinquency_flag",
              "next_12m_default_flag", "next_12m_prepayment_flag", "next_state",
              "exception_required", "exception_type"]
test_df = test_df.drop(columns=label_cols)

# ----------------------------------------------------------------------
# 7. SERVICER UPDATES (second source with conflicts / staleness)
# ----------------------------------------------------------------------
sample_n = int(len(panel) * 0.15)
servicer_sample = panel.sample(n=sample_n, random_state=7).copy()

servicer_updates = pd.DataFrame({
    "loan_id": servicer_sample["loan_id"].values,
    "reporting_month": servicer_sample["reporting_month"].values,
    "current_status": servicer_sample["current_status"].values,
    "current_balance": servicer_sample["current_balance"].values,
    "days_past_due": servicer_sample["days_past_due"].values,
    "last_updated_at": servicer_sample["last_updated_at"].values,
    "source_system": "Servicer_Feed",
})

conflict_idx = servicer_updates.sample(frac=0.30, random_state=8).index
half = len(conflict_idx) // 2
balance_conflict_idx = conflict_idx[:half]
stale_idx = conflict_idx[half:]

servicer_updates.loc[balance_conflict_idx, "current_balance"] = (
    servicer_updates.loc[balance_conflict_idx, "current_balance"] * rng.uniform(0.7, 1.4, len(balance_conflict_idx))
)
servicer_updates.loc[stale_idx, "last_updated_at"] = (
    pd.to_datetime(servicer_updates.loc[stale_idx, "reporting_month"]) - pd.DateOffset(months=4)
)

# ----------------------------------------------------------------------
# 8. VALIDATION RULES (starter deterministic checks)
# ----------------------------------------------------------------------
validation_rules = {
    "rules": [
        {
            "rule_id": "balance_non_increasing",
            "description": "current_balance should not increase month-over-month unless modification_flag=1",
            "severity": "high",
            "applies_to": "loan_monthly_performance"
        },
        {
            "rule_id": "date_order_valid",
            "description": "reporting_month must be >= origination_month",
            "severity": "high",
            "applies_to": "loan_monthly_performance"
        },
        {
            "rule_id": "dpd_status_consistency",
            "description": "days_past_due must be consistent with current_status band (e.g. Current=0, 90DPD>=60)",
            "severity": "medium",
            "applies_to": "loan_monthly_performance"
        },
        {
            "rule_id": "terminal_state_immutability",
            "description": "once a loan reaches Default or Prepaid, no subsequent row may show current_status=Current for the same loan without an explicit correction record",
            "severity": "high",
            "applies_to": "loan_monthly_performance"
        },
        {
            "rule_id": "document_completeness_at_closure",
            "description": "document_status should be 'Complete' when current_status is Default or Prepaid",
            "severity": "medium",
            "applies_to": "loan_monthly_performance"
        },
        {
            "rule_id": "servicer_source_agreement",
            "description": "current_balance and current_status reported by servicer_updates.csv should match the main panel within tolerance for the same loan_id/reporting_month",
            "severity": "medium",
            "applies_to": "cross_source"
        }
    ]
}

# ----------------------------------------------------------------------
# 9. MACRO SCENARIOS
# ----------------------------------------------------------------------
macro_scenarios = pd.DataFrame([
    {"scenario_name": "base", "rate_shock_bps": 0, "unemployment_shock_pts": 0.0,
     "hpi_shock_pct": 0.0, "default_multiplier": 1.00, "delinquency_multiplier": 1.00,
     "prepayment_multiplier": 1.00},
    {"scenario_name": "adverse_credit", "rate_shock_bps": 150, "unemployment_shock_pts": 2.5,
     "hpi_shock_pct": -8.0, "default_multiplier": 1.85, "delinquency_multiplier": 1.55,
     "prepayment_multiplier": 0.55},
    {"scenario_name": "high_prepayment", "rate_shock_bps": -125, "unemployment_shock_pts": -0.3,
     "hpi_shock_pct": 3.0, "default_multiplier": 0.90, "delinquency_multiplier": 0.92,
     "prepayment_multiplier": 2.20},
])

# ----------------------------------------------------------------------
# 10. DATA DICTIONARY
# ----------------------------------------------------------------------
data_dictionary_md = """# Data Dictionary (Synthetic Data Pack)

> **This describes SYNTHETIC data** generated to match the shape of the
> Intain Campus FinTech Challenge 2026 data pack. Field names and formats
> mirror the real spec; values are simulated and carry no real-world meaning.

## loan_static_attributes.csv
| Field | Description |
|---|---|
| loan_id | Unique loan identifier |
| origination_month | Month the loan originated |
| original_balance | Balance at origination (USD) |
| original_term_months | Original loan term in months (180 or 360) |
| original_interest_rate | Note rate at origination (%) |
| credit_score_band | Borrower credit score band at origination |
| ltv_band | Loan-to-value band at origination |
| dti_band | Debt-to-income band at origination |
| state | US state of the property |
| loan_purpose | Purchase / Refinance / Cash-Out Refinance |
| property_type | Single Family / Condo / Multi-Family / Manufactured |
| occupancy_type | Primary / Second Home / Investment |
| servicer_name | Servicing entity |
| vintage | Origination year |

## loan_monthly_performance_train.csv / _test.csv
| Field | Description |
|---|---|
| loan_id | Loan identifier (joins to static attributes) |
| month_index | Months since origination (0-indexed) |
| reporting_month | Calendar month of this record |
| origination_month | Loan's origination month |
| loan_age_months | Same as month_index, kept for clarity |
| remaining_term_months | Months remaining on the note |
| original_balance | Balance at origination |
| current_balance | Balance as of reporting_month |
| interest_rate | Current note rate (may change after modification) |
| credit_score_band / ltv_band / dti_band | Carried from static attributes |
| state / loan_purpose / occupancy_type / property_type / servicer_name | Carried from static attributes |
| current_status | Current / 30DPD / 60DPD / 90DPD / Default / Prepaid / Matured |
| days_past_due | Days past due, consistent with current_status |
| modification_flag | 1 if the loan has been modified |
| prepayment_flag | 1 if current_status == Prepaid this month |
| default_flag | 1 if current_status == Default this month |
| loss_severity_band | Loss severity band, populated only at Default |
| last_updated_at | Timestamp this record was last updated by source system |
| source_system | Originating system for this record |
| document_status | Complete / Pending / Missing |
| next_3m_delinquency_flag | TARGET (train only): reaches 30DPD+ within 3 months |
| next_6m_delinquency_flag | TARGET (train only): reaches 30DPD+ within 6 months |
| next_12m_default_flag | TARGET (train only): reaches Default within 12 months |
| next_12m_prepayment_flag | TARGET (train only): reaches Prepaid within 12 months |
| next_state | TARGET (train only): current_status at t+1 |
| exception_required | TARGET (train only): 1 if this record has a data-quality exception |
| exception_type | TARGET (train only): type of exception, or 'none' |

## servicer_updates.csv
Second-source feed for the same loan_id/reporting_month pairs. Deliberately
contains a subset of conflicting current_balance values and a subset of
stale last_updated_at timestamps, for reconciliation logic to detect.

## validation_rules.json
Deterministic starter rules for balance consistency, date validity,
delinquency/DPD consistency, terminal-state integrity, document completeness,
and cross-source agreement.

## macro_scenarios.csv
Assumption sets for base / adverse_credit / high_prepayment scenarios used
by the portfolio scenario simulator.
"""

# ----------------------------------------------------------------------
# 11. SUBMISSION TEMPLATE
# ----------------------------------------------------------------------
submission_template = pd.DataFrame(columns=[
    "loan_id", "reporting_month",
    "next_3m_delinquency_prob", "next_6m_delinquency_prob",
    "next_12m_default_prob", "next_12m_prepayment_prob",
    "next_state_pred", "next_state_confidence",
    "exception_required_pred", "exception_type_pred", "exception_confidence",
    "anomaly_score", "top_drivers", "recommended_action", "model_confidence"
])

# ----------------------------------------------------------------------
# WRITE EVERYTHING
# ----------------------------------------------------------------------
static_df.to_csv(OUT_DIR / "loan_static_attributes.csv", index=False)
train_df.to_csv(OUT_DIR / "loan_monthly_performance_train.csv", index=False)
test_df.to_csv(OUT_DIR / "loan_monthly_performance_test.csv", index=False)
servicer_updates.to_csv(OUT_DIR / "servicer_updates.csv", index=False)
macro_scenarios.to_csv(OUT_DIR / "macro_scenarios.csv", index=False)
submission_template.to_csv(OUT_DIR / "submission_template.csv", index=False)

with open(OUT_DIR / "validation_rules.json", "w") as f:
    json.dump(validation_rules, f, indent=2)

with open(OUT_DIR / "data_dictionary.md", "w") as f:
    f.write(data_dictionary_md)

# ----------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------
print("=== SYNTHETIC DATA PACK SUMMARY ===")
print(f"Loans (static):              {len(static_df)}")
print(f"Train panel rows:            {len(train_df)}  ({train_df['reporting_month'].min().date()} to {train_df['reporting_month'].max().date()})")
print(f"Test panel rows:             {len(test_df)}  ({test_df['reporting_month'].min().date()} to {test_df['reporting_month'].max().date()})")
print(f"Servicer update rows:        {len(servicer_updates)}")
print()
print("Train current_status distribution:")
print(train_df["current_status"].value_counts())
print()
print("Train target rates:")
for c in ["next_3m_delinquency_flag", "next_6m_delinquency_flag", "next_12m_default_flag", "next_12m_prepayment_flag"]:
    print(f"  {c}: {train_df[c].mean():.3%}")
print()
print("Exception rate (train):", f"{train_df['exception_required'].mean():.3%}")
print(train_df["exception_type"].value_counts())
