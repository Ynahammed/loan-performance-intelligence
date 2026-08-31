"""
Scoring: turn a fitted set of artifacts into predictions for new rows.

THE PANEL CONTINUITY PROBLEM
----------------------------
The test file is not an independent sample. It is the SAME loans, six
months later: 1,455 of its 1,523 loans also appear in training, and its
months run 2024-01 to 2024-06 where training ends 2023-12.

That matters because half the engineered features are histories --
months delinquent to date, worst delinquency reached, current
delinquency streak. Computing them on the test file alone would reset
every loan's history to zero in January 2024, so a borrower who spent
2023 rolling through 30 and 60 DPD would be scored as though they had a
spotless record.

So features are engineered ONCE on the concatenated panel, and the rows
are split apart afterwards. This is not leakage: every history feature
uses an expanding window over rows up to and including the row being
scored, so a 2024-03 row sees 2023 and nothing after March. The
`tests/test_submission.py` fixture pins that.

The 68 test loans with no training history genuinely start from zero,
and are counted in the scoring report rather than hidden -- their
history features are structurally less informative and a reviewer should
know which records those are.

PHASE: 12
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.anomaly.detector import detect_anomalies
from src.anomaly.rules import evaluate_rules
from src.config import LOAN_ID_COLUMN, STATE_COLUMN, TIME_COLUMN
from src.data.loader import attach_static_attributes, sort_panel
from src.data.profiler import record_quality_scores
from src.data.reconciliation import reconcile
from src.features.engineering import engineer_features

logger = logging.getLogger(__name__)

TRAIN_TAG = "train"
SCORE_TAG = "score"


@dataclass
class ScoringFrame:
    """Engineered features for both panels, kept aligned and separable."""

    combined: pd.DataFrame
    train_index: pd.Index
    score_index: pd.Index
    rule_flags: pd.DataFrame
    reconciliation_flags: pd.DataFrame
    quality: pd.DataFrame
    loans_without_history: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def train(self) -> pd.DataFrame:
        return self.combined.loc[self.train_index]

    @property
    def score(self) -> pd.DataFrame:
        return self.combined.loc[self.score_index]

    def summary(self) -> str:
        return "\n".join([
            "combined rows            : {:,}".format(len(self.combined)),
            "training rows            : {:,}".format(len(self.train_index)),
            "rows to score            : {:,}".format(len(self.score_index)),
            "scored loans w/o history : {:,}".format(
                len(self.loans_without_history)),
        ])


def build_scoring_frame(
    train_panel: pd.DataFrame,
    score_panel: pd.DataFrame,
    static: pd.DataFrame = None,
    servicer: pd.DataFrame = None,
) -> ScoringFrame:
    """Engineer features across both panels at once, then split them back.

    Row identity is preserved: `score_index` selects exactly the rows of
    `score_panel`, in its original order, including any duplicated
    (loan_id, reporting_month) pairs. The submission has to line up with
    the test file row for row, so deduplicating here would silently
    change the output shape.
    """
    notes = []
    train = train_panel.copy()
    score = score_panel.copy()
    train["_origin"] = TRAIN_TAG
    score["_origin"] = SCORE_TAG
    score["_score_order"] = np.arange(len(score))

    combined = pd.concat([train, score], ignore_index=True)
    combined = attach_static_attributes(combined, static)
    combined = sort_panel(combined)

    recon = reconcile(combined, servicer)
    engine = evaluate_rules(combined, reconciliation=recon.record_flags)
    quality = record_quality_scores(
        combined, engine.severity_score(), recon.record_flags
    )
    features, _ = engineer_features(
        combined, engine.flags, recon.record_flags, quality["data_quality_score"]
    )

    train_index = features.index[features["_origin"] == TRAIN_TAG]
    score_rows = features[features["_origin"] == SCORE_TAG]
    # Restore the score panel's original row order so the submission can be
    # written straight against the test file.
    score_index = score_rows.sort_values("_score_order").index

    train_loans = set(train_panel[LOAN_ID_COLUMN])
    scored_loans = set(score_panel[LOAN_ID_COLUMN])
    without = sorted(scored_loans - train_loans)
    if without:
        notes.append(
            "{:,} of {:,} scored loans have no training history; their "
            "history features start from zero and are less informative."
            .format(len(without), len(scored_loans))
        )

    dupes = int(score_panel.duplicated([LOAN_ID_COLUMN, TIME_COLUMN]).sum())
    if dupes:
        notes.append(
            "The panel to score contains {:,} duplicate (loan_id, "
            "reporting_month) rows. They are scored individually and kept, "
            "so the output matches the input row for row -- deduplicating "
            "would change the submission's shape.".format(dupes)
        )

    return ScoringFrame(
        combined=features,
        train_index=train_index,
        score_index=score_index,
        rule_flags=engine.flags,
        reconciliation_flags=recon.record_flags,
        quality=quality,
        loans_without_history=without,
        notes=notes,
    )


def score_binary_targets(
    frame: ScoringFrame, models: dict
) -> pd.DataFrame:
    """Calibrated probabilities for each binary target.

    `models` maps target -> (fitted pipeline, columns, calibration or None).
    Calibration is applied where it exists, because the calibrated
    probability is the one the risk categories and the submission use.
    """
    out = pd.DataFrame(index=frame.score_index)
    score_df = frame.score
    for target, (model, cols, calibration) in models.items():
        p = model.predict_proba(score_df[cols])[:, 1]
        if calibration is not None:
            p = calibration.apply(p)
        out[target] = np.clip(p, 0.0, 1.0)
    return out


def score_next_state(
    frame: ScoringFrame, transition_model, states: list
) -> pd.DataFrame:
    """One-month-ahead state distribution from the transition engine.

    Uses the calibrated transition model rather than the supervised
    multiclass one: it is the component that was blended and validated for
    probability quality, and `next_state_confidence` is a probability.
    """
    score_df = frame.score
    proba = np.zeros((len(score_df), len(states)))
    origins = score_df[STATE_COLUMN].astype(str)

    for state in origins.unique():
        mask = (origins == state).to_numpy()
        sub = score_df.loc[mask]
        if state in transition_model.absorbing:
            if state in states:
                proba[mask, states.index(state)] = 1.0
            continue
        if state not in states:
            logger.warning("unseen origin state %r; predicting persistence", state)
            continue
        proba[mask, :] = transition_model.row_probabilities(sub, state)

    proba = np.clip(proba, 1e-9, None)
    proba = proba / proba.sum(axis=1, keepdims=True)
    best = proba.argmax(axis=1)
    return pd.DataFrame(
        {
            "next_state_pred": np.array(states)[best],
            "next_state_confidence": proba[np.arange(len(proba)), best],
        },
        index=frame.score_index,
    )


def score_anomalies(frame: ScoringFrame):
    """Fit the detector on the training rows, score the rows to score.

    Fitting on the combined frame would let the scored population define
    its own notion of normal, which is the wrong question: a record should
    be judged unusual relative to the population the model was built on.
    """
    return detect_anomalies(frame.score, fit_on=frame.train)
