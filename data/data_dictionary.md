# Data Dictionary (Synthetic Data Pack)

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
