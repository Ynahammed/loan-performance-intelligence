"""
Output guardrails: the check that decides whether LLM text is publishable.

WHY THIS IS THE CENTRE OF THE LLM LAYER
---------------------------------------
The structured-context design makes hallucination unlikely -- the model
receives a validated JSON object, never a dataframe. It does not make it
impossible. A model handed "probability: 0.0432" can still write "roughly
a one in twenty chance, similar to the 8% portfolio average", inventing
the second figure entirely. Prompt instructions do not prevent that; a
check on the output does.

So every generation is parsed and tested before it is shown to anyone,
and the result of that test is written to the audit log whether it passed
or failed. A guardrail whose rejections are not counted is a claim, not a
control.

FOUR CHECKS, EACH FOR A DIFFERENT FAILURE
-----------------------------------------
  NUMERIC GROUNDING   every number in the output must trace to a number
                      in the context. This is the one that catches
                      fabricated statistics.
  CAUSAL CLAIMS       SHAP attributions say what moved a model's output,
                      not what caused an outcome. Text that says
                      "because the borrower..." has overstated the
                      evidence, however fluent it is.
  DECISION LANGUAGE   output is advisory. "Recommend denial" is a
                      decision; "recommend review" is not.
  FALSE CERTAINTY     "will default", "guaranteed", "certainly" attach
                      confidence the model does not have.

ON THE NUMERIC POOL
-------------------
A number counts as grounded if it appears anywhere in the serialised
context -- values, keys, and string fields alike -- or is a standard
rendering of one (0.0432 as 4.32%, or rounded to 4.3%). Field names carry
real numbers: "next_3m_delinquency" legitimises "3" in "the next 3
months". Building the pool from the serialised text rather than from
values alone is what makes that work without an arbitrary allowlist of
small integers, which would punch a hole straight through the check.

PHASE: 10
STATUS: implemented.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Numbers are matched WITHOUT a sign, and the sign is decided afterwards
# from what precedes them. A single regex cannot do both jobs: making the
# minus optional inside the pattern reads "2023-09-01" as 2023, -9, -1,
# while forbidding a preceding word character loses the "3" in
# `next_3m_delinquency`. Both mistakes were made, in that order, and both
# corrupted the guardrail -- the first rejected valid output, the second
# let a date ground a fabricated statistic.
NUMBER_PATTERN = re.compile(r"\$?\d[\d,]*\.?\d*\s*%?")

# A hyphen is a minus sign only where a minus sign could start a number:
# at the beginning, or after whitespace or an opening bracket. After a
# letter, digit or underscore it is a separator.
SIGN_CONTEXT = re.compile(r"(?:^|[\s(\[{:,=])-\s*$")

# Typographic dashes and minus signs an LLM may emit, normalised before
# parsing so text and context are compared on the same alphabet. A model
# writing a date with U+2011 was the single largest source of false
# rejections on the first live run.
DASH_TRANSLATION = {ord(c): "-" for c in "‐‑‒–—―−"}
# Non-breaking and narrow spaces appear inside numbers as thousands
# separators and would otherwise split one number into two.
SPACE_TRANSLATION = {ord(c): " " for c in "   "}

RELATIVE_TOLERANCE = 0.02   # a rendering may round; 4.3% for 4.32% is fine
# Deliberately tight. An earlier value of 0.005 was loose enough that a
# fabricated "8.7% portfolio average" -> 0.087 matched 0.09, which had
# entered the pool as the "09" of a reporting date rescaled by the percent
# rule. A date grounded an invented statistic. Rounding tolerance is the
# relative bound's job; this one only absorbs float noise.
ABSOLUTE_TOLERANCE = 1e-6

CAUSAL_PHRASES = (
    "because the borrower", "because this borrower", "caused by",
    "the cause of", "due to the borrower", "as a result of the borrower",
    "this means the borrower", "proves that", "demonstrates that the borrower",
)

DECISION_PHRASES = (
    "recommend denial", "recommend denying", "should be denied",
    "deny the", "approve the loan", "should be approved",
    "recommend foreclosure", "should be foreclosed", "terminate the loan",
    "should be charged off", "recommend rejection", "reject the",
)

CERTAINTY_PHRASES = (
    "will default", "will certainly", "is guaranteed", "guaranteed to",
    "definitely will", "certain to default", "will not default",
    "no risk of", "zero risk", "cannot default", "is certain",
)

DISCLAIMER_MARKERS = (
    "recommendation", "not a decision", "for review", "model estimate",
    "requires review", "not established causes", "advisory",
)


@dataclass
class GroundingResult:
    ok: bool
    checks: dict = field(default_factory=dict)
    ungrounded_numbers: list = field(default_factory=list)
    violations: list = field(default_factory=list)

    @property
    def failed_checks(self) -> list:
        return [name for name, passed in self.checks.items() if not passed]

    def describe(self) -> str:
        lines = ["VERDICT: {}".format("accepted" if self.ok else "REJECTED")]
        for name, passed in self.checks.items():
            lines.append("  {:<20} {}".format(name, "pass" if passed else "FAIL"))
        for v in self.violations:
            lines.append("  - " + v)
        return "\n".join(lines)


def _to_float(token: str):
    cleaned = token.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


# A narrow or non-breaking space BETWEEN digits is a thousands separator,
# not a word break: "13 104.95" is one number written the European way.
# Translating it to a plain space split it into 13 and 104.95, neither of
# which appears in the context, and the guardrail rejected a correctly
# grounded figure. Only ASCII-space it when it is not separating digits.
DIGIT_GROUP_SEPARATOR = re.compile(r"(?<=\d)[   ](?=\d)")


def normalise_text(text: str) -> str:
    """Fold typographic punctuation to ASCII before any parsing."""
    if not text:
        return ""
    text = DIGIT_GROUP_SEPARATOR.sub("", text)
    return text.translate(DASH_TRANSLATION).translate(SPACE_TRANSLATION)


def extract_numbers(text: str) -> list:
    """Every number in the text, with its sign and percent form."""
    normalised = normalise_text(text)
    found = []
    for match in NUMBER_PATTERN.finditer(normalised):
        raw = match.group(0)
        value = _to_float(raw)
        if value is None:
            continue
        preceding = normalised[max(0, match.start() - 12):match.start()]
        negative = bool(SIGN_CONTEXT.search(preceding))
        found.append({
            "raw": ("-" + raw.strip()) if negative else raw.strip(),
            "value": -value if negative else value,
            "is_percent": "%" in raw,
        })
    return found


def _numeric_leaves(obj) -> list:
    """Actual numeric VALUES in the context, recursively."""
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(_numeric_leaves(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_numeric_leaves(value))
    return out


def _literal_numbers(obj) -> list:
    """Numbers that appear inside STRINGS and KEYS of the context.

    Dates, identifiers and field names -- "2023-09-01", "LN100175",
    "next_3m_delinquency". These legitimise a model writing "the next 3
    months" or repeating a reporting date, but they are not quantities and
    must never be rescaled: treating the "09" of a date as a probability
    of 9% is how a date came to ground a fabricated portfolio average.
    """
    out = []
    if isinstance(obj, str):
        return [n["value"] for n in extract_numbers(obj)]
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.extend(n["value"] for n in extract_numbers(str(key)))
            out.extend(_literal_numbers(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_literal_numbers(value))
    return out


def context_number_pool(context) -> tuple:
    """Two pools, because two kinds of number deserve different treatment.

    QUANTITIES  numeric values in the context. Admitted in their own form,
                as a percentage if they are probabilities, and at common
                roundings -- asking a model to write "0.0432" rather than
                "4.3%" would be fighting the one thing it is here to do.

    LITERALS    numbers inside strings and keys: dates, loan ids, the "3"
                in `next_3m_delinquency`. Matched EXACTLY. They carry no
                scale, so rescaling them invents grounding that was never
                there.
    """
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (json.JSONDecodeError, TypeError):
            return ({n["value"] for n in extract_numbers(context)}, set())

    quantities = set()
    for value in _numeric_leaves(context):
        # Magnitude as well as signed value: a delta of -0.06666 is
        # legitimately described as "a 6.666% decline", with the direction
        # carried by the word rather than the sign. Rejecting that would
        # be insisting the model write minus signs into prose.
        for form in {value, abs(value)}:
            quantities.add(form)
            if abs(form) <= 1.0:
                quantities.add(round(form * 100, 6))
                for digits in (0, 1, 2, 3):
                    quantities.add(round(form * 100, digits))
            for digits in (0, 1, 2, 3, 4):
                quantities.add(round(form, digits))

    literals = set(_literal_numbers(context))
    return quantities, literals


def _is_grounded(value: float, quantities: set, literals: set) -> bool:
    # Literals must match exactly: a date component is not a quantity.
    for allowed in literals:
        if abs(value - allowed) <= ABSOLUTE_TOLERANCE:
            return True
    for allowed in quantities:
        if abs(value - allowed) <= ABSOLUTE_TOLERANCE:
            return True
        if allowed != 0 and abs(value - allowed) / abs(allowed) <= RELATIVE_TOLERANCE:
            return True
    return False


def check_numeric_grounding(text: str, context) -> tuple:
    """Every number in `text` must trace to one in `context`."""
    quantities, literals = context_number_pool(context)
    ungrounded = []
    for item in extract_numbers(text):
        value = item["value"]
        candidates = [value]
        if item["is_percent"]:
            candidates.append(value / 100.0)
        if not any(_is_grounded(c, quantities, literals) for c in candidates):
            ungrounded.append(item["raw"])
    return (not ungrounded), ungrounded


def _quantitative_context(context) -> list:
    """Numeric findings the note is expected to report back."""
    if not isinstance(context, dict):
        return []
    values = []
    for key in ("predictions", "next_state", "anomaly"):
        block = context.get(key)
        if isinstance(block, dict):
            values.extend(v for v in block.values()
                          if isinstance(v, (int, float)) and v is not None)
    if isinstance(context.get("model_confidence"), (int, float)):
        values.append(context["model_confidence"])
    return values


def _phrase_hits(text: str, phrases) -> list:
    lowered = normalise_text(text).lower()
    return [p for p in phrases if p in lowered]


def validate_output(
    text: str,
    context,
    require_disclaimer: bool = True,
) -> GroundingResult:
    """Run every guardrail. Returns the verdict and what failed."""
    checks, violations = {}, []

    grounded, ungrounded = check_numeric_grounding(text, context)
    checks["numeric_grounding"] = grounded
    if not grounded:
        violations.append(
            "ungrounded numbers not present in the context: {}".format(
                ", ".join(ungrounded))
        )

    causal = _phrase_hits(text, CAUSAL_PHRASES)
    checks["no_causal_claims"] = not causal
    if causal:
        violations.append(
            "causal language over attributions: {}".format(", ".join(causal))
        )

    decision = _phrase_hits(text, DECISION_PHRASES)
    checks["no_decision_language"] = not decision
    if decision:
        violations.append(
            "instructs a decision rather than a review: {}".format(
                ", ".join(decision))
        )

    certainty = _phrase_hits(text, CERTAINTY_PHRASES)
    checks["no_false_certainty"] = not certainty
    if certainty:
        violations.append(
            "asserts certainty the model does not have: {}".format(
                ", ".join(certainty))
        )

    if require_disclaimer:
        has_disclaimer = bool(_phrase_hits(text, DISCLAIMER_MARKERS))
        checks["carries_disclaimer"] = has_disclaimer
        if not has_disclaimer:
            violations.append(
                "no language marking the output as a recommendation for review"
            )

    if not (text or "").strip():
        checks["non_empty"] = False
        violations.append("empty output")
    else:
        checks["non_empty"] = True

    # The vague non-answer. Fluent, grounded (it cites nothing, so nothing
    # can be ungrounded), disclaimed, and useless -- it passed every other
    # check on the first run. If the context carries model estimates, a
    # note that cites none of them has not reported them.
    expected = _quantitative_context(context)
    if expected:
        cited = bool(extract_numbers(text))
        checks["reports_the_numbers"] = cited
        if not cited:
            violations.append(
                "the context carries {} model estimate(s) and the note cites "
                "none of them; this is a non-answer".format(len(expected))
            )

    return GroundingResult(
        ok=all(checks.values()),
        checks=checks,
        ungrounded_numbers=ungrounded,
        violations=violations,
    )
