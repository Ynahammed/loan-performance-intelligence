"""
Structured audit trail for every LLM call.

One JSON line per call in logs/llm_calls.jsonl. Every call is logged --
accepted, rejected, failed, and fallback -- because a log that only
records successes cannot support the claim the log exists to support.
The rejection rate is computed from this file, not asserted in prose.

Prompts are stored in full alongside a hash. The hash makes it cheap to
tell whether two calls used the same prompt without diffing kilobytes of
text; the full prompt is what makes the log auditable, which is the
point. Nothing in this system sends borrower-identifying data to a
provider, so there is no reason to redact the prompt from our own log.

PHASE: 10
STATUS: implemented.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LOG_NAME = "llm_calls.jsonl"


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def log_llm_call(
    log_path,
    provider: str,
    model: str,
    prompt: str,
    output: str,
    latency_ms: float,
    validation_result=None,
    context=None,
    task: str = "reviewer_note",
    accepted: bool = True,
    attempt: int = 1,
    error: str = None,
) -> dict:
    """Append one call record. Returns the record that was written."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "provider": provider,
        "model": model,
        "attempt": attempt,
        "prompt": prompt,
        "prompt_sha256_16": _hash(prompt),
        "context_sha256_16": _hash(
            json.dumps(context, sort_keys=True, default=str) if context else ""
        ),
        "output": output,
        "latency_ms": round(float(latency_ms), 2),
        "accepted": bool(accepted),
        "error": error,
    }

    if validation_result is not None:
        record["validation"] = {
            "ok": bool(validation_result.ok),
            "checks": validation_result.checks,
            "failed_checks": validation_result.failed_checks,
            "violations": validation_result.violations,
            "ungrounded_numbers": validation_result.ungrounded_numbers,
        }

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record


def read_log(log_path) -> list:
    """Read the audit trail back. Malformed lines are skipped, not fatal."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping malformed log line")
    return records


def summarise_log(log_path) -> dict:
    """The governance numbers, computed from the log rather than claimed."""
    records = read_log(log_path)
    if not records:
        return {"total_calls": 0}

    # A provider error and a guardrail rejection are different failures and
    # must not share a rate. Pooling them reported a "59% rejection rate"
    # when the guardrail had rejected 12 generations and the other 26 were
    # HTTP 404s from a retired model -- a number that reads as "the model
    # hallucinates constantly" when it means "the endpoint was down".
    errored = [r for r in records if r.get("error")]
    # A model returning an empty completion never reached the validator
    # either. Counting it as a guardrail rejection overstates how often
    # the guardrail fires on real content.
    empty = [r for r in records
             if not r.get("error") and not (r.get("output") or "").strip()]
    rejected = [r for r in records
                if not r.get("accepted", True) and not r.get("error")
                and (r.get("output") or "").strip()]
    failure_counts = {}
    for record in rejected:
        for check in record.get("validation", {}).get("failed_checks", []):
            failure_counts[check] = failure_counts.get(check, 0) + 1

    providers = {}
    for record in records:
        key = "{}/{}".format(record.get("provider"), record.get("model"))
        providers[key] = providers.get(key, 0) + 1

    latencies = sorted(r.get("latency_ms", 0) for r in records)
    validated = len(records) - len(errored) - len(empty)
    return {
        "total_calls": len(records),
        "reached_the_guardrail": validated,
        "accepted": validated - len(rejected),
        "rejected_by_guardrail": len(rejected),
        "provider_errors": len(errored),
        "empty_completions": len(empty),
        "rejection_rate": round(len(rejected) / validated, 4) if validated else 0.0,
        "by_provider": providers,
        "rejections_by_check": failure_counts,
        "median_latency_ms": latencies[len(latencies) // 2] if latencies else 0,
    }
