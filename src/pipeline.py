"""
Orchestration layer. Coordinates data loading -> profiling -> drift ->
reconciliation -> cleaning -> feature engineering -> multi-target training
-> survival modeling -> anomaly/exception detection -> explainability ->
scenario simulation -> LLM review.

The Streamlit app imports from here rather than calling every module
directly, so the dashboard stays thin and the pipeline stays testable
independent of Streamlit.

PHASE: 2-13 (grows incrementally with each phase)
STATUS: not yet implemented -- placeholder for Step 2.
"""


class PipelineState:
    """Holds everything the dashboard needs after a run: dataframes,
    trained models, metrics, SHAP values, anomaly results, etc.
    Populated once per session and cached in st.session_state."""

    def __init__(self):
        self.raw_train = None
        self.raw_test = None
        self.static_attributes = None
        self.servicer_updates = None
        self.profile_report = None
        self.drift_report = None
        self.reconciliation_report = None
        self.cleaned_train = None
        self.feature_manifest = None
        self.trained_models = {}
        self.calibrated_models = {}
        self.metrics = {}
        self.survival_model = None
        self.anomaly_results = None
        self.shap_results = {}
        self.scenario_results = None
