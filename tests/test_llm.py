"""
Tests for the LLM layer: guardrails, providers, audit log, retrieval.

The guardrail tests are the important ones. Every other component in this
project fails loudly when it breaks; a weakened output validator keeps
returning "accepted" and nothing looks wrong until a fabricated number is
in front of a reviewer.
"""
from __future__ import annotations

import json

import pytest

from src.llm.grounding import (
    check_numeric_grounding,
    context_number_pool,
    extract_numbers,
    validate_output,
)
from src.llm.llm_logging import log_llm_call, read_log, summarise_log
from src.llm.provider import (
    FaultInjectionProvider,
    TemplateFallbackProvider,
    get_provider,
    render_reviewer_note,
    render_scenario_note,
)
from src.llm.retrieval import DocumentationRetriever, Passage, answer_from_documentation
from src.llm.reviewer import build_context, generate_reviewer_note, validate_llm_output


CONTEXT = {
    "loan_id": "LN100123",
    "reporting_month": "2023-06-01",
    "current_status": "Current",
    "predictions": {"next_3m_delinquency": 0.0432},
    "top_drivers": [
        {"label": "months since origination", "direction": "increases risk",
         "contribution": 0.31},
    ],
    "anomaly": {"score": 0.712, "drivers": []},
}

GOOD_NOTE = (
    "Loan LN100123 as of 2023-06-01. Model estimates: 3-month delinquency "
    "4.32%. The model weighted months since origination toward higher risk. "
    "Anomaly score 0.712 indicates an unusual pattern requiring review. "
    "This note is a recommendation for review, not a decision."
)


# ---------------------------------------------------------- number pool


def test_extract_numbers_handles_percent_currency_and_separators():
    found = extract_numbers("4.32% of $1,250,000 across 3 months, -0.5 delta")
    values = [f["value"] for f in found]
    assert 4.32 in values
    assert 1250000.0 in values
    assert 3.0 in values
    assert any(f["is_percent"] for f in found)


def test_percentage_rendering_of_a_probability_is_grounded():
    ok, ungrounded = check_numeric_grounding("the estimate is 4.32%", CONTEXT)
    assert ok, ungrounded


def test_rounded_rendering_is_grounded():
    ok, _ = check_numeric_grounding("roughly 4.3%", CONTEXT)
    assert ok


def test_numbers_embedded_in_field_names_are_grounded():
    """`next_3m_delinquency` legitimises "3" in "the next 3 months" without
    an allowlist of small integers punching a hole in the check."""
    ok, ungrounded = check_numeric_grounding("over the next 3 months", CONTEXT)
    assert ok, ungrounded


def test_fabricated_statistic_is_caught():
    text = "4.32%, well below the 8.7% portfolio average"
    ok, ungrounded = check_numeric_grounding(text, CONTEXT)
    assert not ok
    assert any("8.7" in u for u in ungrounded)


def test_context_pool_admits_both_probability_and_percent_forms():
    quantities, _ = context_number_pool(CONTEXT)
    assert any(abs(p - 0.0432) < 1e-9 for p in quantities)
    assert any(abs(p - 4.32) < 1e-9 for p in quantities)


def test_literals_are_matched_exactly_and_never_rescaled():
    """A date component is not a quantity.

    Regression: "09" from a reporting date entered the pool as 9, the
    percent rule added 0.09, and a fabricated "8.7% portfolio average"
    (0.087) matched it inside the absolute tolerance. A date grounded an
    invented statistic.
    """
    context = {"reporting_month": "2023-09-01",
               "predictions": {"next_3m_delinquency": 0.788}}
    quantities, literals = context_number_pool(context)
    assert 9.0 in literals          # usable, as written
    assert 0.09 not in quantities   # but never rescaled into a probability

    ok, ungrounded = check_numeric_grounding(
        "below the 8.7% portfolio average", context)
    assert not ok and ungrounded


def test_narrow_space_thousands_separator_is_one_number():
    """Regression: a model wrote 13 104.95, the European convention.
    Translating the narrow space to a plain space split it into 13 and
    104.95, neither of which was in the context, and a correctly grounded
    figure was rejected as ungrounded."""
    from src.llm.grounding import extract_numbers

    for space in (" ", " ", " "):
        values = [n["value"] for n in extract_numbers("13" + space + "104.95")]
        assert values == [13104.95], (space, values)

    context = {"anomaly": {"drivers": [{"robust_deviations": 13104.95}]}}
    ok, ungrounded = check_numeric_grounding(
        "balance movement (13 104.95)", context)
    assert ok, ungrounded


def test_plain_spaces_between_numbers_are_not_joined():
    """The fix must not glue unrelated numbers together. An ASCII space is
    a word break, not a thousands separator."""
    from src.llm.grounding import extract_numbers

    assert [n["value"] for n in extract_numbers("the next 3 months")] == [3.0]
    assert [n["value"] for n in extract_numbers("0.71 123 loans")] == [0.71, 123.0]


def test_typographic_dashes_do_not_break_grounding():
    """Regression: a model writing a date with U+2011 had it parsed as
    2023, 9, 1 while the ASCII context parsed as 2023, -9, -1. The two
    never matched, and 51 of 58 live rejections were this artifact rather
    than hallucination."""
    context = {"reporting_month": "2023-09-01", "predictions": {"p": 0.5}}
    ok, ungrounded = check_numeric_grounding(
        "reported 2023‑09‑01", context)
    assert ok, ungrounded


# ------------------------------------------------------------ guardrails


def test_a_good_note_passes_every_check():
    result = validate_output(GOOD_NOTE, CONTEXT)
    assert result.ok, result.describe()


@pytest.mark.parametrize("fault,expected_check", [
    ("fabricated_statistic", "numeric_grounding"),
    ("causal_overreach", "no_causal_claims"),
    ("decision_language", "no_decision_language"),
    ("false_certainty", "no_false_certainty"),
    ("missing_disclaimer", "carries_disclaimer"),
    ("vague_non_answer", "reports_the_numbers"),
])
def test_every_injected_fault_class_is_caught(fault, expected_check):
    text = FaultInjectionProvider(fault).generate("")
    result = validate_output(text, CONTEXT)
    assert not result.ok
    assert expected_check in result.failed_checks


def test_vague_non_answer_would_pass_without_the_numbers_check():
    """It cites nothing, so nothing can be ungrounded; it is disclaimed and
    makes no causal or decision claim. Only the non-answer check stops it."""
    text = FaultInjectionProvider("vague_non_answer").generate("")
    result = validate_output(text, CONTEXT)
    assert result.checks["numeric_grounding"]
    assert result.checks["no_causal_claims"]
    assert result.checks["carries_disclaimer"]
    assert not result.checks["reports_the_numbers"]


def test_non_answer_check_is_skipped_when_the_context_has_no_numbers():
    result = validate_output(
        "The documentation does not cover this. Recommendation for review.",
        {"question": "what is x"},
    )
    assert "reports_the_numbers" not in result.checks


def test_empty_output_is_rejected():
    assert not validate_output("", CONTEXT).ok


# ------------------------------------------------------------- providers


def test_template_provider_output_is_grounded_by_construction():
    note = render_reviewer_note(CONTEXT)
    result = validate_output(note, CONTEXT)
    assert result.ok, result.describe()


def test_template_provider_dispatches_scenario_prompts():
    """Regression: a scenario prompt was rendered by the loan-note renderer
    and came back fluent, validated, and about nothing."""
    context = {"horizon_months": 12, "scenarios": [
        {"scenario": "base", "default_12m": 0.0135, "prepaid_12m": 0.1604}]}
    prompt = "summarise this scenario simulation\n" + json.dumps(context)
    text = TemplateFallbackProvider().generate(prompt)
    assert "base" in text
    assert "Loan this loan" not in text


def test_scenario_note_reports_each_scenario():
    context = {"horizon_months": 12, "scenarios": [
        {"scenario": "base", "default_12m": 0.0135, "prepaid_12m": 0.1604},
        {"scenario": "adverse_credit", "default_12m": 0.0322, "prepaid_12m": 0.0938},
    ]}
    text = render_scenario_note(context)
    assert "base" in text and "adverse_credit" in text
    assert validate_output(text, context).ok


def test_get_provider_falls_back_without_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    provider = get_provider()
    assert provider.name == "template"


def test_get_provider_never_raises_on_bad_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    assert get_provider().name == "template"


def test_missing_context_does_not_crash_the_template_provider():
    text = TemplateFallbackProvider().generate("no json here")
    assert "recommendation" in text.lower()


# --------------------------------------------------------------- logging


def test_every_call_is_logged_including_rejections(tmp_path):
    path = tmp_path / "calls.jsonl"
    rejected = validate_output(
        FaultInjectionProvider("fabricated_statistic").generate(""), CONTEXT)

    log_llm_call(path, "test", "m1", "p", "good", 12.0,
                 validation_result=validate_output(GOOD_NOTE, CONTEXT),
                 context=CONTEXT, accepted=True)
    log_llm_call(path, "test", "m1", "p", "bad", 15.0,
                 validation_result=rejected, context=CONTEXT, accepted=False)

    records = read_log(path)
    assert len(records) == 2
    assert records[1]["validation"]["failed_checks"]


def test_log_summary_computes_the_rejection_rate(tmp_path):
    path = tmp_path / "calls.jsonl"
    for i in range(8):
        accepted = i % 4 != 0
        log_llm_call(path, "test", "m", "p", "o", 1.0, accepted=accepted,
                     validation_result=validate_output(
                         GOOD_NOTE if accepted else "", CONTEXT))
    stats = summarise_log(path)
    assert stats["total_calls"] == 8
    assert stats["rejected_by_guardrail"] == 2
    assert stats["rejection_rate"] == 0.25


def test_provider_errors_are_not_counted_as_guardrail_rejections(tmp_path):
    """A retired model returning HTTP 404 is an outage, not a hallucination.

    Pooling them reported a 59% "rejection rate" on a run where the
    guardrail rejected 12 generations and 26 calls never reached it.
    """
    path = tmp_path / "calls.jsonl"
    for _ in range(6):
        log_llm_call(path, "groq", "retired-model", "p", "", 5.0,
                     accepted=False, error="404 model_not_found")
    for _ in range(4):
        log_llm_call(path, "groq", "m", "p", GOOD_NOTE, 5.0, accepted=True,
                     validation_result=validate_output(GOOD_NOTE, CONTEXT))
    log_llm_call(path, "groq", "m", "p", "bad 8.7% invented", 5.0, accepted=False,
                 validation_result=validate_output("bad 8.7% invented", CONTEXT))

    stats = summarise_log(path)
    assert stats["total_calls"] == 11
    assert stats["provider_errors"] == 6
    assert stats["reached_the_guardrail"] == 5
    assert stats["rejected_by_guardrail"] == 1
    # 1 of the 5 that actually reached the guardrail, not 7 of 11.
    assert stats["rejection_rate"] == 0.2


def test_malformed_log_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "calls.jsonl"
    log_llm_call(path, "t", "m", "p", "o", 1.0)
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n",
                    encoding="utf-8")
    assert len(read_log(path)) == 1


def test_summary_of_a_missing_log_is_not_an_error(tmp_path):
    assert summarise_log(tmp_path / "absent.jsonl") == {"total_calls": 0}


# ------------------------------------------------------------- retrieval


def _retriever():
    return DocumentationRetriever(passages=[
        Passage("ltv_band: Loan-to-value band at origination",
                "data_dictionary.md", "static", key="ltv_band"),
        Passage("credit_score_band: Borrower credit score band at origination",
                "data_dictionary.md", "static", key="credit_score_band"),
        Passage("days_past_due: Days past due, consistent with current_status",
                "data_dictionary.md", "monthly", key="days_past_due"),
    ])


def test_field_lookup_retrieves_the_right_row():
    hits = _retriever().search("What does ltv_band mean?")
    assert hits
    assert hits[0]["passage"].key == "ltv_band"


def test_off_topic_question_retrieves_nothing():
    """The control that caught the first implementation: this scored 0.198
    against credit_score_band -- higher than two legitimate lookups --
    because that row contains the word "Borrower"."""
    assert _retriever().search("What is the borrower's favourite colour?") == []


def test_unanswerable_question_says_so_rather_than_guessing():
    answer = answer_from_documentation(
        "How do I bake sourdough bread?", _retriever())
    assert not answer["grounded"]
    assert "does not cover" in answer["answer"]
    assert answer["citations"] == []


def test_grounded_answer_carries_its_citation():
    answer = answer_from_documentation("What is days_past_due?", _retriever())
    assert answer["grounded"]
    assert answer["citations"]


def test_empty_corpus_does_not_crash():
    assert DocumentationRetriever(passages=[]).search("anything") == []


# -------------------------------------------------------------- reviewer


def test_rejected_generation_falls_back_and_logs_the_substitution(tmp_path):
    """A rejected generation must produce the grounded template rendering,
    never a blank and never the rejected text."""
    path = tmp_path / "calls.jsonl"
    note = generate_reviewer_note(
        CONTEXT, provider=FaultInjectionProvider("fabricated_statistic"),
        log_path=path,
    )
    assert note.fell_back
    assert note.accepted
    assert "8.7" not in note.text

    records = read_log(path)
    assert len(records) >= 3          # two attempts plus the fallback
    assert any(not r["accepted"] for r in records)
    assert records[-1]["provider"] == "template"


def test_provider_exception_falls_back_rather_than_propagating(tmp_path):
    class _Broken:
        name, model = "broken", "v0"

        def generate(self, prompt, **kwargs):
            raise RuntimeError("connection reset")

    note = generate_reviewer_note(CONTEXT, provider=_Broken(),
                                  log_path=tmp_path / "c.jsonl")
    assert note.fell_back and note.accepted
    records = read_log(tmp_path / "c.jsonl")
    assert any(r.get("error") for r in records)


def test_build_context_contains_only_computed_values():
    import pandas as pd

    row = pd.Series({
        "loan_id": "LN1", "reporting_month": pd.Timestamp("2023-01-01"),
        "current_status": "Current",
    })
    context = build_context(row, predictions={"next_12m_default": 0.0123456789})
    assert context["loan_id"] == "LN1"
    # Rounded, so the model is not handed digits it might truncate its own way.
    assert context["predictions"]["next_12m_default"] == 0.01235


def test_narrow_percentage_check_flags_a_mismatch():
    ok, mismatched = validate_llm_output("the risk is 12%", 0.0432)
    assert not ok and mismatched
    ok, _ = validate_llm_output("the risk is 4.32%", 0.0432)
    assert ok
