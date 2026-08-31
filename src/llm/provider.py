"""
LLMProvider abstraction.

The rest of the codebase depends only on this interface, never on a
vendor SDK, so swapping providers never touches reviewer.py or
prompts.py.

  GroqProvider              LLM_PROVIDER=groq, GROQ_API_KEY
  OpenAICompatibleProvider  LLM_PROVIDER=openai, OPENAI_API_KEY,
                            optional OPENAI_BASE_URL for any compatible endpoint
  TemplateFallbackProvider  LLM_PROVIDER=none, or automatic on any error
  FaultInjectionProvider    test harness only -- see the warning below

WHY THE TEMPLATE PROVIDER IS NOT A CONSOLATION PRIZE
----------------------------------------------------
It renders the same grounded context object through deterministic
strings. Because it can only emit values that are in the context, it is
incapable of the failure the guardrails exist to catch. That makes the
whole system fully functional with zero configuration and zero network
calls -- which also means a live demo cannot fail because an API key
expired or a rate limit hit.

FAULT INJECTION, AND AN HONESTY CONSTRAINT
------------------------------------------
The challenge asks for examples where the LLM was wrong, vague or
overconfident. With no API key configured, we have no real LLM output,
and inventing some and captioning it "the LLM said this" would be
fabricating evidence -- precisely the thing the rest of this project
refuses to do.

So FaultInjectionProvider emits deliberately defective text covering the
known failure modes, and everything it produces is labelled in the log
and in the report as fault injection, never as model output. It
demonstrates that the guardrail catches these classes. When a real key is
configured, the identical guardrail runs against real generations and
logs them the same way, and the report then carries both sections
separately.

PHASE: 10
STATUS: implemented.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}

# Minimum seconds between calls to a hosted provider. Free tiers are rate
# limited per minute, and running into that limit is not free: the SDK
# retries with escalating backoff (4s, then 7s, then longer), so a run
# that would take one minute at a polite pace takes ten when it sprints.
# Throttling ahead of the limit is strictly faster than being throttled by
# it. Override with LLM_MIN_INTERVAL_SECONDS.
DEFAULT_MIN_INTERVAL_SECONDS = 2.0


class _Throttle:
    """Process-wide minimum spacing between calls to one provider."""

    def __init__(self, min_interval: float):
        self.min_interval = max(float(min_interval), 0.0)
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last = time.monotonic()


def _min_interval() -> float:
    raw = os.getenv("LLM_MIN_INTERVAL_SECONDS")
    if raw is None:
        return DEFAULT_MIN_INTERVAL_SECONDS
    try:
        return float(raw)
    except ValueError:
        logger.warning("LLM_MIN_INTERVAL_SECONDS=%r is not a number; using %s",
                       raw, DEFAULT_MIN_INTERVAL_SECONDS)
        return DEFAULT_MIN_INTERVAL_SECONDS


class LLMProvider(ABC):
    name = "abstract"
    model = "none"

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


class TemplateFallbackProvider(LLMProvider):
    """Deterministic renderer. Always available, never calls the network.

    Reads the JSON context back out of the prompt and renders it. Because
    every value it emits came from the context, its output is grounded by
    construction -- it still goes through the same validator, because a
    guardrail applied selectively is not a guardrail.
    """

    name = "template"
    model = "deterministic-v1"

    def generate(self, prompt: str, **kwargs) -> str:
        context = _extract_context(prompt)
        if context is None:
            return (
                "Model outputs are available for this record but could not be "
                "rendered into a note. This is a recommendation for review, "
                "not a decision."
            )
        # Dispatch on the prompt, because the provider interface carries no
        # task argument. Without this, a scenario prompt was rendered by the
        # loan-note renderer and came back as "Loan this loan as of the
        # reporting month. Model estimates are unavailable" -- fluent,
        # validated, and about nothing.
        return self._render_for(prompt, context)

    @staticmethod
    def _render_for(prompt: str, context: dict) -> str:
        lowered = (prompt or "").lower()
        if "scenario" in lowered or "scenarios" in context:
            return render_scenario_note(context)
        return render_reviewer_note(context)


def _extract_context(prompt: str):
    """Pull the JSON block back out of a rendered prompt."""
    match = re.search(r"\{.*\}", prompt or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _pct(value) -> str:
    try:
        return "{:.2f}%".format(float(value) * 100)
    except (TypeError, ValueError):
        return "unavailable"


def render_reviewer_note(context: dict) -> str:
    """Deterministic reviewer note built only from context values."""
    loan = context.get("loan_id", "this loan")
    month = context.get("reporting_month", "the reporting month")
    preds = context.get("predictions", {}) or {}
    drivers = context.get("top_drivers", []) or []
    anomaly = context.get("anomaly", {}) or {}
    exception = context.get("exception", {}) or {}

    parts = ["Loan {} as of {}.".format(loan, month)]

    named = {
        "next_3m_delinquency": "3-month delinquency",
        "next_6m_delinquency": "6-month delinquency",
        "next_12m_default": "12-month default",
        "next_12m_prepayment": "12-month prepayment",
    }
    stated = [
        "{} {}".format(label, _pct(preds[key]))
        for key, label in named.items() if key in preds
    ]
    if stated:
        parts.append("Model estimates: " + "; ".join(stated) + ".")
    else:
        parts.append("Model estimates are unavailable for this record.")

    if context.get("next_state"):
        parts.append(
            "The transition model puts next month's most likely state at {} "
            "with confidence {}.".format(
                context["next_state"].get("predicted", "unavailable"),
                _pct(context["next_state"].get("confidence")),
            )
        )

    if drivers:
        up = [d["label"] for d in drivers if d.get("direction", "").startswith("increase")]
        down = [d["label"] for d in drivers if d.get("direction", "").startswith("reduce")]
        if up:
            parts.append("Factors the model weighted toward higher risk: "
                         + ", ".join(up) + ".")
        if down:
            parts.append("Factors weighing the other way: " + ", ".join(down) + ".")
        parts.append("These are the factors the model leaned on, "
                     "not established causes.")

    if exception.get("required"):
        parts.append(
            "A validation rule flagged this record ({}). Verify the record "
            "before relying on the scores above.".format(
                exception.get("type", "unspecified"))
        )
    elif anomaly.get("score") is not None:
        parts.append(
            "Anomaly score {} on a 0-1 scale; this indicates an unusual "
            "pattern requiring review, not wrongdoing.".format(
                round(float(anomaly["score"]), 3))
        )

    if context.get("model_confidence") is not None:
        parts.append(
            "Model confidence for this record is {}, measured as stability "
            "across models trained on different time periods.".format(
                round(float(context["model_confidence"]), 3))
        )

    parts.append("This note is a recommendation for review, not a decision.")
    return " ".join(parts)


def render_scenario_note(context: dict) -> str:
    """Deterministic portfolio scenario narrative."""
    rows = context.get("scenarios", []) or []
    if not rows:
        return ("No scenario results are available. This note is a "
                "recommendation for review, not a decision.")
    parts = ["Portfolio projection over a {}-month horizon.".format(
        context.get("horizon_months", 12))]
    for row in rows:
        parts.append(
            "Under the {} scenario, projected 12-month default reaches {} "
            "and prepayment {}.".format(
                row.get("scenario"), _pct(row.get("default_12m")),
                _pct(row.get("prepaid_12m")))
        )
    parts.append("These are model simulations under stated assumptions, "
                 "not forecasts, and are a recommendation for review.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Live providers
# ---------------------------------------------------------------------------


class GroqProvider(LLMProvider):
    """Groq chat completions, with the model resolved at connect time.

    Hosted model catalogues churn. A hardcoded name that works today
    returns 404 `model_not_found` a few months later, and on this project
    it already did: `llama-3.3-70b-versatile` was retired between the code
    being written and being run. Every call failed, the fallback caught it
    and the run completed on templates -- correct behaviour, and still a
    silent downgrade nobody would notice mid-demo.

    So the model is chosen by asking the API what exists, preferring the
    configured name, then a small ordered preference list, then whatever
    the account actually has. An explicit LLM_MODEL_NAME always wins.
    """

    name = "groq"

    # Ordered by preference, not by size: an instruct-tuned mid-size model
    # writes better short reviewer notes than a large general one, and the
    # task here is two sentences of grounded prose.
    # Ordered by preference, not by size: an instruct-tuned mid-size model
    # writes better short reviewer notes than a large general one, and the
    # task here is two sentences of grounded prose.
    #
    # Kept deliberately long and hosted-catalogue-agnostic. Rate limits on
    # Groq are enforced per organization AND per model, so a list with a
    # single viable entry means one exhausted quota stops the pipeline;
    # several viable entries mean discovery simply moves on. That is also
    # why a second API key does not help -- the quota follows the account.
    PREFERRED = (
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "gemma2-9b-it",
    )

    def __init__(self, api_key: str = None, model: str = None, timeout: float = 30.0):
        from groq import Groq

        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self._timeout = timeout
        self._throttle = _Throttle(_min_interval())
        configured = model or os.getenv("LLM_MODEL_NAME")
        self.model = configured or self._discover_model()

    def _discover_model(self) -> str:
        """Ask the account what it can actually run."""
        try:
            available = {m.id for m in self._client.models.list().data}
        except Exception as exc:
            logger.warning(
                "could not list Groq models (%s); falling back to %s",
                exc, DEFAULT_MODELS["groq"],
            )
            return DEFAULT_MODELS["groq"]

        for candidate in self.PREFERRED:
            if candidate in available:
                logger.info("Groq model resolved to %s", candidate)
                return candidate

        chat_models = sorted(
            m for m in available
            if not any(t in m.lower() for t in ("whisper", "tts", "guard", "embed"))
        )
        if chat_models:
            logger.warning(
                "none of the preferred Groq models are available; using %s",
                chat_models[0],
            )
            return chat_models[0]
        return DEFAULT_MODELS["groq"]

    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        self._throttle.wait()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            timeout=self._timeout,
            **kwargs,
        )
        return response.choices[0].message.content


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: str = None, timeout: float = 30.0):
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL_NAME") or DEFAULT_MODELS["openai"]
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            timeout=timeout,
        )
        self._throttle = _Throttle(_min_interval())

    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        self._throttle.wait()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Fault injection -- TEST HARNESS ONLY
# ---------------------------------------------------------------------------


class FaultInjectionProvider(LLMProvider):
    """Emits deliberately defective text to exercise the guardrails.

    NOT A MODEL. Everything this returns is written by us, and every log
    record and report section carrying its output is labelled as fault
    injection. It exists so the guardrail can be shown catching each
    failure class when no live provider is configured -- never so that
    invented text can be captioned as something a model said.
    """

    name = "fault-injection"
    model = "synthetic-faults-v1"

    FAULTS = {
        "fabricated_statistic": (
            "This loan carries a 4.32% three-month delinquency estimate, "
            "which sits well below the 8.7% portfolio average and the 12.4% "
            "figure for its vintage cohort. Recommendation for review."
        ),
        "causal_overreach": (
            "The elevated risk here is because the borrower has been unable "
            "to keep up with payments, and the loan-to-value ratio proves "
            "that equity has deteriorated. This is a recommendation."
        ),
        "decision_language": (
            "Given the model output, this loan should be denied further "
            "modification and recommend foreclosure proceedings begin."
        ),
        "false_certainty": (
            "With a score at this level the loan will default within twelve "
            "months; there is zero risk of a cure. Recommendation for review."
        ),
        "missing_disclaimer": (
            "Three-month delinquency estimate is 4.32%. Top factors are "
            "months since origination and the amortisation gap."
        ),
        "vague_non_answer": (
            "There are several considerations here and the picture is mixed. "
            "Various factors could push this either way depending on "
            "circumstances. Recommendation for review."
        ),
    }

    def __init__(self, fault: str = "fabricated_statistic"):
        if fault not in self.FAULTS:
            raise ValueError("unknown fault {!r}; expected one of {}".format(
                fault, sorted(self.FAULTS)))
        self.fault = fault

    def generate(self, prompt: str, **kwargs) -> str:
        return self.FAULTS[self.fault]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def get_provider(name: str = None) -> LLMProvider:
    """Env-driven provider selection that never raises.

    An unreachable or misconfigured provider degrades to templates with a
    warning rather than taking the pipeline down. Everything downstream
    still runs; the log records which provider actually served the call.
    """
    choice = (name or os.getenv("LLM_PROVIDER") or "none").strip().lower()

    if choice in ("none", "", "template", "fallback"):
        return TemplateFallbackProvider()

    try:
        if choice == "groq":
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY is not set")
            return GroqProvider()
        if choice in ("openai", "openai-compatible"):
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            return OpenAICompatibleProvider()
    except Exception as exc:
        logger.warning(
            "LLM provider %r unavailable (%s); falling back to deterministic "
            "templates. The system remains fully functional.", choice, exc
        )
        return TemplateFallbackProvider()

    logger.warning("unknown LLM_PROVIDER %r; using templates", choice)
    return TemplateFallbackProvider()
