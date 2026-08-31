# Development Decisions Log

Running log of major implementation decisions, kept from day one so it
can be reformatted into the required AI Development Log at the end.
Each entry: what was tried, what was rejected and why, final approach.

## Phase 1 -- Architecture
- Decided on ML-first architecture with LLM strictly as a narrator over
  structured ML outputs (never raw dataframes) -- structurally prevents
  hallucinated numbers.
- Decided against a database; local filesystem artifacts (joblib,
  CSV/parquet) are sufficient for a single-analyst demo tool and keep
  the project simple to run/judge.
- Decided to keep survival/transition modeling as a standalone module
  (survival.py) rather than folding it into train.py, since it is a
  separately graded rubric category (15 pts) with distinct baseline-
  comparison requirements (censoring, KM/Cox).

## Phase 1 -- Synthetic data foundation
- Generated a synthetic data pack (generate_synthetic_data.py, seed=42)
  shaped exactly to the real schema, so Phase 2+ code is a drop-in once
  the real organizer data pack is available.
- Deliberately injected: missingness, duplicates, invalid dates,
  outliers, inconsistent categorical casing, cross-source conflicts
  (servicer_updates.csv), and a real macro regime shift between the
  train (2019-2023) and test (2024) periods so the drift-detection task
  has genuine signal to find, not a synthetic no-op.
