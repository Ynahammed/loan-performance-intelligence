"""
The reviewer copilot: structured ML outputs in, validated prose out.

THE PIPELINE FOR EVERY GENERATION
---------------------------------
    build a typed context object from computed results
        -> serialise to JSON and render into a grounded prompt
        -> call the configured provider
        -> validate the output against the context
        -> on failure, retry once, then fall back to templates
        -> log every attempt, accepted or not

The context object is the whole design. The LLM never sees a dataframe,
so it cannot read a number nobody asked it to use; it sees a small dict
of computed values with the numbers already rounded and labelled. That
makes fabrication unlikely. The validator then makes it non-shippable,
which is a different and stronger guarantee.

WHAT THE FALLBACK MEANS
-----------------------
A rejected generation does not produce a blank. It produces the
deterministic template rendering of the same context, which is grounded
by construction. So the worst case for a reviewer is a plainer note, not
a missing one, and the audit log records that a substitution happened and
why. The system has no state in which unvalidated text reaches a user.

PHASE: 10
STATUS: implemented.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import LOGS_DIR
from src.explainability.labels import describe_exception, label
from src.llm.grounding import validate_output
from src.llm.llm_logging import log_llm_call
from src.llm.prompts import (
    anomaly_explanation_prompt,
    reviewer_summary_prompt,
    scenario_summary_prompt,
)
from src.llm.provider import (
    TemplateFallbackProvider,
    get_provider,
    render_reviewer_note,
    render_scenario_note,
)

logger = logging.getLogger(__name__)

LOG_PATH = LOGS_DIR / "llm_calls.jsonl"
MAX_ATTEMPTS = 2


@dataclass
class ReviewerNote:
    text: str
    context: dict
    provider: str
    model: str
    accepted: bool
    fell_back: bool
    validation: object = None
    attempts: int = 1
    notes: list = field(default_factory=list)


def _round(value, digits=5):
    """Round for the context object so the model is not handed 17 digits
    it might truncate differently than we would."""
    if value is None:
        return None
    try:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def build_context(
    row: pd.Series,
    predictions: dict = None,
    drivers=None,
    anomaly: dict = None,
    exception: dict = None,
    next_state: dict = None,
    model_confidence=None,
) -> dict:
    """Assemble the grounded context object for one loan.

    Only computed values go in. Nothing here is free text from the panel
    that could smuggle an ungrounded figure into the prompt.
    """
    context = {
        "loan_id": str(row.get("loan_id", "unknown")),
        "reporting_month": str(pd.Timestamp(row["reporting_month"]).date())
        if "reporting_month" in row else "unknown",
        "current_status": str(row.get("current_status", "unknown")),
        "predictions": {k: _round(v) for k, v in (predictions or {}).items()},
    }

    if next_state:
        context["next_state"] = {
            "predicted": str(next_state.get("predicted", "unavailable")),
            "confidence": _round(next_state.get("confidence")),
        }

    if drivers is not None and len(drivers):
        rows = drivers.to_dict("records") if hasattr(drivers, "to_dict") else drivers
        context["top_drivers"] = [
            {
                "label": d.get("label") or label(d.get("source_column", "")),
                "direction": d.get("direction", "unknown"),
                "contribution": _round(d.get("contribution")),
            }
            for d in rows[:5]
        ]

    if anomaly:
        context["anomaly"] = {
            "score": _round(anomaly.get("score"), 4),
            "drivers": [
                {
                    "label": d.get("label"),
                    "direction": d.get("direction"),
                    "robust_deviations": _round(d.get("robust_deviations"), 2),
                }
                for d in (anomaly.get("drivers") or [])[:3]
            ],
        }

    if exception:
        context["exception"] = {
            "required": bool(exception.get("required", False)),
            "type": describe_exception(exception.get("type", "none")),
            "decided_by": exception.get("decided_by", "unknown"),
            "rules_broken": exception.get("rules_broken", []),
        }

    if model_confidence is not None:
        context["model_confidence"] = _round(model_confidence, 4)

    return context


def _generate_with_guardrails(
    provider,
    prompt: str,
    context: dict,
    task: str,
    fallback_renderer,
    log_path=LOG_PATH,
) -> ReviewerNote:
    """One generation, validated, retried once, then fallen back."""
    notes = []
    last_validation = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        error = None
        try:
            text = provider.generate(prompt)
        except Exception as exc:            # any provider failure
            text, error = "", str(exc)
            logger.warning("provider %s failed: %s", provider.name, exc)
        latency = (time.perf_counter() - started) * 1000

        validation = validate_output(text, context) if text else None
        accepted = bool(validation and validation.ok)
        last_validation = validation

        log_llm_call(
            log_path, provider.name, provider.model, prompt, text, latency,
            validation_result=validation, context=context, task=task,
            accepted=accepted, attempt=attempt, error=error,
        )

        if accepted:
            return ReviewerNote(
                text=text, context=context, provider=provider.name,
                model=provider.model, accepted=True, fell_back=False,
                validation=validation, attempts=attempt, notes=notes,
            )

        if validation is not None:
            notes.append("attempt {} rejected: {}".format(
                attempt, "; ".join(validation.failed_checks)))
        elif error:
            notes.append("attempt {} failed: {}".format(attempt, error))

    # Deterministic substitute, grounded by construction. Logged too, so
    # the trail shows a substitution happened rather than going quiet.
    started = time.perf_counter()
    text = fallback_renderer(context)
    latency = (time.perf_counter() - started) * 1000
    validation = validate_output(text, context)
    log_llm_call(
        log_path, "template", "deterministic-v1", prompt, text, latency,
        validation_result=validation, context=context,
        task=task + "_fallback", accepted=validation.ok, attempt=MAX_ATTEMPTS + 1,
    )
    notes.append("substituted the deterministic template rendering")

    return ReviewerNote(
        text=text, context=context, provider="template",
        model="deterministic-v1", accepted=validation.ok, fell_back=True,
        validation=validation, attempts=MAX_ATTEMPTS + 1, notes=notes,
    )


def generate_reviewer_note(context: dict, provider=None, log_path=LOG_PATH) -> ReviewerNote:
    provider = provider or get_provider()
    prompt = reviewer_summary_prompt(json.dumps(context, indent=2, default=str))
    return _generate_with_guardrails(
        provider, prompt, context, "reviewer_note", render_reviewer_note, log_path
    )


def generate_anomaly_explanation(context: dict, provider=None, log_path=LOG_PATH) -> ReviewerNote:
    provider = provider or get_provider()
    prompt = anomaly_explanation_prompt(json.dumps(context, indent=2, default=str))
    return _generate_with_guardrails(
        provider, prompt, context, "anomaly_explanation",
        render_reviewer_note, log_path,
    )


def generate_scenario_summary(context: dict, provider=None, log_path=LOG_PATH) -> ReviewerNote:
    provider = provider or get_provider()
    prompt = scenario_summary_prompt(json.dumps(context, indent=2, default=str))
    return _generate_with_guardrails(
        provider, prompt, context, "scenario_summary",
        render_scenario_note, log_path,
    )


def validate_llm_output(llm_text: str, expected_probability: float, tolerance: float = 0.02):
    """Narrow check kept for the dashboard: does a stated percentage match
    the computed probability?

    Superseded by `grounding.validate_output` for the pipeline, which
    checks every number rather than one. Retained because a reviewer
    editing a note by hand wants exactly this question answered.
    """
    from src.llm.grounding import extract_numbers

    stated = [n["value"] / 100.0 for n in extract_numbers(llm_text or "")
              if n["is_percent"]]
    if not stated:
        return True, []
    mismatched = [
        s for s in stated if abs(s - float(expected_probability)) > tolerance
    ]
    return (not mismatched), mismatched
