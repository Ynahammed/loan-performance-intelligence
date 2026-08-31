"""
Feature name -> reviewer-readable label, in one place.

Three consumers need the same vocabulary and must not drift apart:

  - the anomaly detector, when it says why a record was flagged
  - the SHAP layer, when it lists what drove a prediction
  - the LLM reviewer, whose grounded context object carries these strings

If each built its own phrasing, the dashboard would call the same column
three different things on three different pages and the LLM would be
handed a fourth. So the mapping lives here and everything imports it.

Labels are written for a loan reviewer, not a data scientist: no column
names, no underscores, no units the reader has to decode.

PHASE: 7-10
STATUS: implemented.
"""
from __future__ import annotations

FEATURE_LABELS = {
    # Raw panel columns
    "loan_age_months": "months since origination",
    "remaining_term_months": "months left on the term",
    "interest_rate": "interest rate",
    "original_interest_rate": "rate at origination",
    "original_balance": "original loan amount",
    "current_balance": "current balance",
    "days_past_due": "days past due",
    "modification_flag": "loan has been modified",
    "current_status": "payment status",
    "credit_score_band": "credit score band",
    "ltv_band": "loan-to-value band",
    "dti_band": "debt-to-income band",
    "state": "state",
    "loan_purpose": "loan purpose",
    "occupancy_type": "occupancy type",
    "property_type": "property type",
    "servicer_name": "servicer",
    "document_status": "document status",
    "source_system": "source system",
    "vintage": "origination year",
    "original_term_months": "original term",
    # Engineered: amortisation
    "balance_ratio": "share of the original balance still outstanding",
    "term_progress": "how far through its term the loan is",
    "scheduled_balance_ratio": "balance the payment schedule expects",
    "amortisation_gap": "distance from the contractual payment schedule",
    # Engineered: rate incentive
    "market_rate": "prevailing market rate that month",
    "rate_incentive": "rate paid above the market rate",
    "rate_incentive_pctile": "refinance incentive versus other loans that month",
    # Engineered: delinquency history
    "months_delinquent_to_date": "months spent delinquent so far",
    "ever_delinquent": "has ever been delinquent",
    "max_dpd_to_date": "worst delinquency reached",
    "delinquency_streak": "consecutive months currently delinquent",
    "n_status_changes_to_date": "number of status changes so far",
    "balance_change_3m": "balance movement over the last three months",
    "ever_modified": "has ever been modified",
    # Engineered: bands
    "credit_score_band_ordinal": "credit score",
    "ltv_band_ordinal": "loan-to-value ratio",
    "dti_band_ordinal": "debt-to-income ratio",
    # Data quality
    "rule_violation_count": "number of validation rules broken",
    "source_conflict": "servicer feed disagrees with the panel",
    "source_stale": "reporting source updated late",
    "servicer_update_lag_days": "days between the month and its last update",
    "data_quality_score": "record data-quality score",
}

# Plain-English descriptions of each exception type, for reviewer notes.
EXCEPTION_TYPE_LABELS = {
    "none": "no exception",
    "balance_inconsistency": "balance rose without a recorded modification",
    "date_invalid": "reporting date precedes the origination date",
    "document_gap": "documents missing on a closed loan",
    "status_reversal": "loan reported active after reaching a terminal state",
}

RULE_LABELS = {
    "balance_non_increasing": "balance increased with no modification on file",
    "date_order_valid": "reporting month is earlier than origination",
    "dpd_status_consistency": "days past due does not match the reported status",
    "terminal_state_immutability": "loan reported active after closing",
    "document_completeness_at_closure": "documents incomplete at closure",
    "servicer_source_agreement": "servicer feed disagrees with the panel",
    "stale_record": "record updated long after the month it describes",
}


def label(name: str) -> str:
    """Human label for a feature, falling back to a de-underscored name.

    The fallback is deliberate rather than an error: engineered features
    get added faster than the vocabulary does, and a reviewer reading
    'balance change 3m' is inconvenienced, not misled.
    """
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return name.replace("_", " ")


def describe_exception(exception_type: str) -> str:
    return EXCEPTION_TYPE_LABELS.get(exception_type, exception_type)


def describe_rule(rule_id: str) -> str:
    return RULE_LABELS.get(rule_id, rule_id.replace("_", " "))
