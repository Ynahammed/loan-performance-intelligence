"""
Multi-target supervised training.

FIVE TARGETS, ONE PIPELINE
--------------------------
next_3m_delinquency, next_6m_delinquency, next_12m_default,
next_12m_prepayment (binary) and next_state (multiclass). Each gets its
own purged split, its own imbalance handling, and its own calibrator,
because their prevalences differ by an order of magnitude (10.2% for
prepayment, 0.92% for default) and a single global recipe would be wrong
for at least one of them.

BASELINE vs IMPROVED
--------------------
The rubric asks for a comparison, so both are trained on identical splits:

  BASELINE  logistic regression on the raw shipped columns only. Not a
            strawman -- it is regularised, class-weighted, and gets the
            same preprocessing. If the engineered features add nothing,
            this will say so.
  IMPROVED  gradient boosting on the engineered feature set, including
            the amortisation, rate-incentive and delinquency-history
            families plus the phase-2 data-quality signals.

LABEL MATURITY
--------------
A 12-month label on a row 6 months from the end of the panel cannot have
observed 12 months. Those rows are dropped from BOTH training and
evaluation via `mature_label_mask`, rather than being scored against a
label that was manufactured rather than observed. On this panel that
removes a meaningful share of the 12-month targets, and keeping them
would flatter every 12-month metric.

WHAT IS DELIBERATELY NOT A FEATURE
----------------------------------
Every column in LEAKAGE_PRONE_COLUMNS, which includes the other targets.
`next_state` would trivially predict `next_3m_delinquency_flag`; a model
that uses it scores beautifully and is worthless. Also excluded:
loss_severity_band (post-outcome and 99.95% missing) and the raw
identifiers.

PHASE: 4
STATUS: implemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    LEAKAGE_PRONE_COLUMNS,
    LOAN_ID_COLUMN,
    RANDOM_SEED,
    TIME_COLUMN,
)
from src.models.calibration import (
    calibration_by_segment,
    expected_calibration_error,
    reliability_diagram_data,
    select_calibrator,
)
from src.models.validation import TARGET_HORIZON_MONTHS, time_aware_split

logger = logging.getLogger(__name__)

BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]

# Columns that identify a row rather than describe a loan's condition.
# Excluded from every feature set: loan_id would let a tree memorise
# individual loans, and the raw dates encode position in the panel.
IDENTIFIER_COLUMNS = [
    LOAN_ID_COLUMN, TIME_COLUMN, "origination_month", "last_updated_at",
    "month_index",
]

# The baseline sees only columns shipped in the data pack.
BASELINE_COLUMNS = [
    "loan_age_months", "remaining_term_months", "interest_rate",
    "original_balance", "current_balance", "days_past_due",
    "modification_flag", "current_status", "credit_score_band", "ltv_band",
    "dti_band", "state", "loan_purpose", "occupancy_type", "property_type",
    "servicer_name", "document_status",
]


@dataclass
class TargetResult:
    target: str
    kind: str
    metrics: pd.DataFrame
    champion: str
    models: dict = field(default_factory=dict)
    calibration: object = None
    reliability: pd.DataFrame = None
    segment_calibration: pd.DataFrame = None
    split_description: str = ""
    n_train: int = 0
    n_test: int = 0
    prevalence: float = 0.0
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature selection and preprocessing
# ---------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame, restrict_to: list = None) -> list:
    """Everything that is neither a target, a leakage risk, nor an id."""
    banned = set(LEAKAGE_PRONE_COLUMNS) | set(IDENTIFIER_COLUMNS)
    cols = [c for c in df.columns if c not in banned]
    if restrict_to is not None:
        cols = [c for c in cols if c in restrict_to]
    # Drop constants: they carry no information and confuse the encoder.
    return [c for c in cols if df[c].nunique(dropna=False) > 1]


def build_preprocessor(df: pd.DataFrame, cols: list, scale: bool) -> ColumnTransformer:
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), num),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
                ]),
                cat,
            ),
        ],
        remainder="drop",
    )


def make_baseline(df: pd.DataFrame, cols: list, random_state=RANDOM_SEED) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(df, cols, scale=True)),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])


def make_improved(df: pd.DataFrame, cols: list, random_state=RANDOM_SEED) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(df, cols, scale=False)),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=random_state)),
    ])


# ---------------------------------------------------------------------------
# Label maturity
# ---------------------------------------------------------------------------


def mature_label_mask(df: pd.DataFrame, target: str) -> pd.Series:
    """True where the target's forward window fits inside the panel.

    Without this, the last H months of the panel carry labels that could
    not have been observed, and every metric on an H-month target is
    computed partly against fiction.
    """
    horizon = TARGET_HORIZON_MONTHS.get(target, 0)
    if horizon <= 0:
        return pd.Series(True, index=df.index)
    cutoff = df[TIME_COLUMN].max() - pd.DateOffset(months=horizon)
    return df[TIME_COLUMN] <= cutoff


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_precision(y_true, y_score, min_precision: float = 0.5) -> float:
    """Highest recall achievable while holding precision at or above the bar.

    The operational question for a review queue: if we only act on alerts
    that are right half the time, what share of real events do we catch?
    ROC-AUC does not answer it and PR-AUC only summarises it.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n": len(y_true),
        "prevalence": round(float(y_true.mean()), 5),
        "roc_auc": None,
        "pr_auc": None,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 6),
        "ece": round(float(expected_calibration_error(y_true, y_score)), 6),
    }
    if len(set(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
        # Lift over the base rate: PR-AUC alone is not comparable across
        # targets with different prevalence.
        out["pr_auc_lift"] = round(out["pr_auc"] / max(y_true.mean(), 1e-9), 2)
        out["recall_at_p50"] = round(recall_at_precision(y_true, y_score, 0.50), 4)
        out["recall_at_p80"] = round(recall_at_precision(y_true, y_score, 0.80), 4)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_binary_target(
    df: pd.DataFrame,
    target: str,
    test_fraction: float = 0.2,
    calibration_fraction: float = 0.25,
    random_state: int = RANDOM_SEED,
) -> TargetResult:
    """Train baseline and improved models for one binary target."""
    notes = []

    usable = df[df[target].notna()]
    mature = mature_label_mask(usable, target)
    dropped = int((~mature).sum())
    if dropped:
        notes.append(
            "dropped {:,} rows whose {}-month label window extends past the "
            "end of the panel and could not have been observed".format(
                dropped, TARGET_HORIZON_MONTHS.get(target, 0))
        )
    usable = usable[mature]

    split = time_aware_split(
        usable, purge_months=TARGET_HORIZON_MONTHS.get(target, 0),
        test_fraction=test_fraction,
    )
    if len(split.test) == 0 or split.test[target].nunique() < 2:
        notes.append("test window has no usable label variation; target skipped")
        return TargetResult(target, "binary", pd.DataFrame(), "none",
                            notes=notes, split_description=split.describe())

    # Inner slice for calibration, held out of model fitting entirely.
    inner = time_aware_split(
        split.train, purge_months=TARGET_HORIZON_MONTHS.get(target, 0),
        test_fraction=calibration_fraction,
    )
    fit_df = inner.train if len(inner.test) > 200 else split.train
    calib_df = inner.test if len(inner.test) > 200 else None
    if calib_df is None:
        notes.append("calibration slice too small; probabilities left uncalibrated")

    improved_cols = feature_columns(fit_df)
    baseline_cols = feature_columns(fit_df, restrict_to=BASELINE_COLUMNS)

    y_fit = fit_df[target].astype(int)
    rows = []
    models = {}

    for name, cols, factory in (
        ("baseline (logistic, raw columns)", baseline_cols, make_baseline),
        ("improved (gradient boosting, engineered)", improved_cols, make_improved),
    ):
        model = factory(fit_df, cols, random_state)
        if factory is make_improved:
            # HistGB has no class_weight; sample weights are the equivalent,
            # and are applied only to the improved model so the two differ
            # in features and estimator, not in whether imbalance is handled.
            pos = max(int(y_fit.sum()), 1)
            neg = max(len(y_fit) - pos, 1)
            w = np.where(y_fit == 1, neg / pos, 1.0)
            model.fit(fit_df[cols], y_fit, clf__sample_weight=w)
        else:
            model.fit(fit_df[cols], y_fit)
        models[name] = (model, cols)

        p_test = model.predict_proba(split.test[cols])[:, 1]
        m = binary_metrics(split.test[target], p_test)
        m["model"] = name
        m["n_features"] = len(cols)
        m["calibrated"] = False
        rows.append(m)

    # Champion on PR-AUC: with prevalence between 0.9% and 10%, ROC-AUC is
    # dominated by the majority class and flatters everything.
    scored = [r for r in rows if r.get("pr_auc") is not None]
    champion = max(scored, key=lambda r: r["pr_auc"])["model"] if scored else "none"
    model, cols = models[champion]

    calibration = None
    reliability = None
    segment_cal = None
    if calib_df is not None:
        p_calib = model.predict_proba(calib_df[cols])[:, 1]
        calibration = select_calibrator(p_calib, calib_df[target].astype(int))

        p_test_raw = model.predict_proba(split.test[cols])[:, 1]
        p_test_cal = calibration.apply(p_test_raw)
        m = binary_metrics(split.test[target], p_test_cal)
        m["model"] = champion + " + " + calibration.method
        m["n_features"] = len(cols)
        m["calibrated"] = True
        rows.append(m)

        reliability = reliability_diagram_data(split.test[target], p_test_cal)
        # Calibrated overall does not mean calibrated everywhere: errors in
        # opposite directions cancel in the aggregate. Checked by credit
        # band AND by vintage, because a model can be well calibrated on
        # today's borrowers and badly calibrated on a particular cohort.
        segment_cal = {}
        for column in ("credit_score_band", "vintage"):
            if column in split.test.columns:
                table = calibration_by_segment(
                    split.test[target], p_test_cal, split.test[column])
                if len(table):
                    segment_cal[column] = table

    metrics = pd.DataFrame(rows)
    front = ["model", "n_features", "calibrated", "n", "prevalence", "roc_auc",
             "pr_auc", "pr_auc_lift", "recall_at_p50", "recall_at_p80", "f1",
             "brier", "ece"]
    metrics = metrics[[c for c in front if c in metrics.columns]]

    return TargetResult(
        target=target,
        kind="binary",
        metrics=metrics,
        champion=champion,
        models=models,
        calibration=calibration,
        reliability=reliability,
        segment_calibration=segment_cal,
        split_description=split.describe(),
        n_train=len(fit_df),
        n_test=len(split.test),
        prevalence=float(split.test[target].mean()),
        notes=notes,
    )


def train_all_binary_targets(df: pd.DataFrame, targets: list = None) -> dict:
    targets = targets or BINARY_TARGETS
    out = {}
    for target in targets:
        if target not in df.columns:
            logger.warning("target %r not in dataframe; skipped", target)
            continue
        logger.info("Training %s", target)
        out[target] = train_binary_target(df, target)
    return out


# ---------------------------------------------------------------------------
# next_state: hierarchical multiclass
# ---------------------------------------------------------------------------


def train_next_state(
    df: pd.DataFrame,
    target: str = "next_state",
    test_fraction: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> TargetResult:
    """Predict next month's state, hierarchically.

    A flat 6-class model is the obvious approach and the wrong one here.
    98.1% of transitions are Current -> Current, and `Default` appears 49
    times in the whole panel; a flat model spends all its capacity on the
    majority class and predicts it everywhere, scoring 96% accuracy and
    0.16 macro-F1. So the problem is split in two:

        STAGE A  will the state change at all?  (binary, ~2% positive)
        STAGE B  given that it changes, to what? (multiclass, on the ~2%)

    Stage B trains on a balanced-by-construction subpopulation, so the
    rare destinations are no longer competing with 45,000 rows of
    Current -> Current. Recombining is exact:

        P(s) = (1 - p_change)              if s is the current state
             = p_change * P_B(s)           otherwise

    Compared against two baselines: persistence (always predict the
    current state), which is very hard to beat on accuracy and trivially
    beaten on macro-F1, and a flat multiclass model.
    """
    notes = []
    usable = df[df[target].notna() & df["current_status"].notna()]
    split = time_aware_split(usable, purge_months=1, test_fraction=test_fraction)
    if len(split.test) == 0:
        return TargetResult(target, "multiclass", pd.DataFrame(), "none",
                            notes=["empty test window"])

    states = sorted(set(usable[target].dropna()) | set(usable["current_status"].dropna()))
    cols = feature_columns(split.train)

    train_df, test_df = split.train, split.test
    y_train = train_df[target].astype(str)
    y_test = test_df[target].astype(str)
    changed_train = (y_train != train_df["current_status"].astype(str)).astype(int)

    def _proba_frame(values, index):
        return pd.DataFrame(values, columns=states, index=index)

    results = {}

    # --- baseline 1: persistence ---------------------------------------
    persist = np.zeros((len(test_df), len(states)))
    for i, s in enumerate(test_df["current_status"].astype(str)):
        if s in states:
            persist[i, states.index(s)] = 1.0
    results["persistence baseline"] = persist

    # --- baseline 2: flat multiclass -----------------------------------
    flat = Pipeline([
        ("pre", build_preprocessor(train_df, cols, scale=False)),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=random_state)),
    ])
    flat.fit(train_df[cols], y_train)
    flat_p = np.zeros((len(test_df), len(states)))
    for j, cls in enumerate(flat.named_steps["clf"].classes_):
        if cls in states:
            flat_p[:, states.index(cls)] = flat.predict_proba(test_df[cols])[:, j]
    results["flat multiclass"] = flat_p

    # --- improved: hierarchical ----------------------------------------
    # Stage A is deliberately NOT class-weighted, unlike the binary targets.
    # Its output is not thresholded for recall -- it is multiplied into a
    # probability recombination. Re-weighting a 2%-positive problem inflates
    # p_change roughly fiftyfold, which flips the argmax away from the
    # current state far too often: the first run scored 0.33 macro-F1,
    # losing to a persistence baseline that predicts no change at all.
    stage_a = make_improved(train_df, cols, random_state)
    stage_a.fit(train_df[cols], changed_train)

    change_rows = train_df[changed_train.astype(bool).to_numpy()]
    hierarchical_ok = len(change_rows) >= 100 and change_rows[target].nunique() >= 2
    if not hierarchical_ok:
        notes.append("too few observed state changes for a stage-B model")
    else:
        stage_b = Pipeline([
            ("pre", build_preprocessor(change_rows, cols, scale=False)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, random_state=random_state)),
        ])
        stage_b.fit(change_rows[cols], change_rows[target].astype(str))

        p_change = stage_a.predict_proba(test_df[cols])[:, 1]
        p_dest = np.zeros((len(test_df), len(states)))
        raw = stage_b.predict_proba(test_df[cols])
        for j, cls in enumerate(stage_b.named_steps["clf"].classes_):
            if cls in states:
                p_dest[:, states.index(cls)] = raw[:, j]

        combined = p_change[:, None] * p_dest
        # Stage B can put mass back on the current state; that mass belongs
        # to "no change" and would otherwise be double counted.
        for i, s in enumerate(test_df["current_status"].astype(str)):
            if s in states:
                combined[i, states.index(s)] = 0.0
        row_sum = combined.sum(axis=1, keepdims=True)
        combined = np.divide(combined, row_sum, out=np.zeros_like(combined),
                             where=row_sum > 0) * p_change[:, None]
        for i, s in enumerate(test_df["current_status"].astype(str)):
            if s in states:
                combined[i, states.index(s)] = 1.0 - p_change[i]
        results["hierarchical (stage A + stage B)"] = combined

    # --- score everything ----------------------------------------------
    from src.models.survival import multiclass_log_loss

    rows = []
    for name, proba in results.items():
        p = np.clip(proba, 1e-9, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        pred = np.array(states)[p.argmax(axis=1)]
        rows.append({
            "model": name,
            "n": len(test_df),
            "accuracy": round(float((pred == y_test.to_numpy()).mean()), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro",
                                             zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_test, pred, average="weighted",
                                                zero_division=0)), 4),
            "log_loss": round(multiclass_log_loss(y_test.to_numpy(), p, states), 5),
        })

    metrics = pd.DataFrame(rows)

    # Persistence is a reference point, not a candidate. It emits hard 0/1
    # predictions, so it cannot rank loans, cannot fill next_state_confidence
    # in the submission, and cannot be calibrated -- its strong macro-F1
    # comes entirely from the fact that 96% of months see no change. The
    # champion is chosen among the probabilistic models, on log-loss, since
    # the downstream use is a confidence-weighted next-state prediction.
    candidates = metrics[metrics.model != "persistence baseline"]
    champion = candidates.loc[candidates["log_loss"].idxmin(), "model"]

    persistence = metrics[metrics.model == "persistence baseline"]
    if len(persistence):
        p = persistence.iloc[0]
        best = metrics[metrics.model == champion].iloc[0]
        notes.append(
            "Persistence reaches macro-F1 {:.4f} against the champion's "
            "{:.4f}, but at log-loss {:.4f} against {:.4f} -- it wins the "
            "hard-label metric by refusing to predict change at all, while "
            "being {:.1f}x worse as a probability. Reported for honesty; "
            "not selectable.".format(
                p["macro_f1"], best["macro_f1"], p["log_loss"],
                best["log_loss"], p["log_loss"] / max(best["log_loss"], 1e-9))
        )

    return TargetResult(
        target=target, kind="multiclass", metrics=metrics, champion=champion,
        models={}, split_description=split.describe(),
        n_train=len(train_df), n_test=len(test_df), notes=notes,
    )


# ---------------------------------------------------------------------------
# Predictability ceiling
# ---------------------------------------------------------------------------


def predictability_ceiling(
    df: pd.DataFrame,
    target: str,
    test_fraction: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> dict:
    """How much of a target's difficulty is temporal transfer, and how much
    is that the signal was never there?

    A weak out-of-time score has two very different explanations, and the
    remedies are opposite. If the same model scores well under a random
    split, the features carry signal that does not survive the passage of
    time -- a drift problem, worth attacking with period-relative
    features. If it scores badly under BOTH, the signal is absent and no
    amount of feature work will help.

    So this fits the same recipe twice:

      TEMPORAL  chronological, purged. The deployment setting, and the
                only number that should ever be reported as performance.
      RANDOM    loan-disjoint but time-blind. NOT a valid estimate of
                deployment performance -- it lets the model see the
                future. It is a diagnostic ceiling and nothing else.

    The gap between them is the cost of time. The random figure alone is
    the cost of the data.
    """
    from sklearn.model_selection import GroupShuffleSplit

    usable = df[df[target].notna()]
    usable = usable[mature_label_mask(usable, target)]
    horizon = TARGET_HORIZON_MONTHS.get(target, 0)
    cols = feature_columns(usable)
    out = {"target": target, "horizon_months": horizon}

    def _score(train_df, test_df, label):
        y_tr = train_df[target].astype(int)
        y_te = test_df[target].astype(int)
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            return None
        model = make_improved(train_df, cols, random_state)
        pos = max(int(y_tr.sum()), 1)
        neg = max(len(y_tr) - pos, 1)
        model.fit(train_df[cols], y_tr,
                  clf__sample_weight=np.where(y_tr == 1, neg / pos, 1.0))
        p = model.predict_proba(test_df[cols])[:, 1]
        pr = float(average_precision_score(y_te, p))
        prevalence = float(y_te.mean())
        return {
            label + "_roc_auc": round(float(roc_auc_score(y_te, p)), 4),
            label + "_pr_auc": round(pr, 4),
            # PR-AUC is not comparable across splits with different base
            # rates, and these two splits have different ones. Comparing
            # raw PR-AUC made 6-month delinquency look like it IMPROVED
            # out of time by 32%, purely because its temporal test window
            # happened to be 8.5% positive against the random split's
            # 6.0%. Lift over the base rate is the comparable quantity.
            label + "_lift": round(pr / prevalence, 3) if prevalence else None,
            label + "_prevalence": round(prevalence, 5),
            label + "_n_test": int(len(y_te)),
        }

    split = time_aware_split(usable, purge_months=horizon,
                            test_fraction=test_fraction)
    temporal = _score(split.train, split.test, "temporal")
    if temporal:
        out.update(temporal)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction + 0.05,
                            random_state=random_state)
    tr_idx, te_idx = next(gss.split(usable, groups=usable[LOAN_ID_COLUMN]))
    random_scores = _score(usable.iloc[tr_idx], usable.iloc[te_idx], "random")
    if random_scores:
        out.update(random_scores)

    if temporal and random_scores:
        t_lift = out.get("temporal_lift")
        r_lift = out.get("random_lift")
        if t_lift is not None and r_lift:
            out["share_lost_to_time"] = round((r_lift - t_lift) / r_lift, 4)
    return out


# ---------------------------------------------------------------------------
# Uncertainty around the metrics themselves
# ---------------------------------------------------------------------------


def bootstrap_metric_intervals(
    y_true,
    y_score,
    groups=None,
    n_boot: int = 400,
    alpha: float = 0.05,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Percentile confidence intervals for ROC-AUC and PR-AUC.

    CLUSTERED BY LOAN, not by row. The same loan contributes many rows and
    those rows are far from independent -- a borrower who defaults appears
    as a positive in every month leading up to it. Resampling rows
    individually treats that correlated block as many independent
    observations and reports an interval far narrower than the evidence
    supports. Resampling whole loans keeps each borrower's history intact.

    This matters most exactly where it is least convenient: 12-month
    default rests on roughly 66 events, so the interval around its PR-AUC
    is wide, and a point estimate quoted without one invites more
    confidence than the data can carry.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(random_state)

    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in unique}

    roc, pr = [], []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[g] for g in picked])
        yt, ys = y_true[idx], y_score[idx]
        if len(set(yt)) < 2:
            continue
        roc.append(roc_auc_score(yt, ys))
        pr.append(average_precision_score(yt, ys))

    if len(roc) < 30:
        return {"n_boot_usable": len(roc),
                "note": "too few usable resamples for an interval"}

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "n_boot_usable": len(roc),
        "n_clusters": int(len(unique)),
        "roc_auc_point": round(float(roc_auc_score(y_true, y_score)), 4),
        "roc_auc_lo": round(float(np.percentile(roc, lo)), 4),
        "roc_auc_hi": round(float(np.percentile(roc, hi)), 4),
        "pr_auc_point": round(float(average_precision_score(y_true, y_score)), 4),
        "pr_auc_lo": round(float(np.percentile(pr, lo)), 4),
        "pr_auc_hi": round(float(np.percentile(pr, hi)), 4),
    }


def slice_metrics(
    df: pd.DataFrame,
    y_true,
    y_score,
    segment_column: str,
    threshold: float = 0.5,
    min_rows: int = 200,
) -> pd.DataFrame:
    """Performance and selection rate by segment.

    A diagnostic, not a verdict. Differences here have several possible
    causes -- genuinely different risk, different base rates, or the model
    serving one group worse than another -- and this table cannot tell
    them apart. It surfaces where to look; deciding what a gap means needs
    context this system does not have.

    `selection_rate` is the share of a segment flagged at the operating
    threshold. A large gap in selection rate alongside a similar observed
    rate is the pattern worth investigating.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    frame = pd.DataFrame({
        "segment": df[segment_column].astype(str).to_numpy(),
        "y": y_true,
        "p": y_score,
    })

    rows = []
    for segment, group in frame.groupby("segment"):
        if len(group) < min_rows:
            continue
        entry = {
            "segment": segment,
            "n": len(group),
            "observed_rate": round(float(group.y.mean()), 5),
            "mean_prediction": round(float(group.p.mean()), 5),
            "selection_rate": round(float((group.p >= threshold).mean()), 5),
        }
        if group.y.nunique() > 1:
            entry["roc_auc"] = round(float(roc_auc_score(group.y, group.p)), 4)
            entry["pr_auc_lift"] = round(
                float(average_precision_score(group.y, group.p) / group.y.mean()), 2
            )
        rows.append(entry)

    out = pd.DataFrame(rows)
    return out.sort_values("observed_rate", ascending=False).reset_index(drop=True) \
        if len(out) else out
