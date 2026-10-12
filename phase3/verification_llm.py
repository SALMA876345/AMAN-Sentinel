"""Phase 3: evidence-grounded LLM verification layer for escalated claims.

Takes a claim (+ language) and an Evidence Pack, builds a strict
evidence-only-reasoning prompt, calls an `LLMProvider`, and returns a
programmatically-validated verdict. The model is never trusted blindly:
every output passes through JSON-repair, citation-existence checking, and an
empty-citation downgrade rule before being returned.

Evidence Pack shape (produced by a parallel retrieval workstream — matched
here exactly, do not change):

    {"claim": "...",
     "retrieved_evidence": [
         {"id": "...", "text": "...", "source": "...",
          "retrieval_method": "...", "retrieval_score": 0.0},
         ...
     ]}

Output schema (validated, not just prompted for):

    {"verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
     "confidence": 0.0-1.0,
     "explanation": "...",
     "evidence_ids": ["..."],
     "claim_language": "ar" | "en" | "mixed"}
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

try:
    from phase3.llm_provider import LLMProvider, MockProvider
except ImportError:
    from llm_provider import LLMProvider, MockProvider

logger = logging.getLogger(__name__)

VALID_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
VALID_LANGUAGES = {"ar", "en", "mixed"}

# Arabic block (covers MSA + dialectal Arabic script + Arabic-Indic digits).
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


# --------------------------------------------------------------------------- #
# Language detection — deliberately simple, character-set ratio only.
# --------------------------------------------------------------------------- #
def detect_language(text: str) -> str:
    """Classify text as 'ar', 'en', or 'mixed' by character-set ratio.

    Not a full language-ID model by design (the task explicitly says "don't
    overengineer") — counts Arabic-script vs Latin-script characters and
    applies a simple dominance threshold.
    """
    if not text or not text.strip():
        return "mixed"
    arabic_count = len(_ARABIC_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    total = arabic_count + latin_count
    if total == 0:
        # No alphabetic signal (e.g. only digits/punctuation) — default to mixed.
        return "mixed"
    arabic_ratio = arabic_count / total
    if arabic_ratio >= 0.85:
        return "ar"
    if arabic_ratio <= 0.15:
        return "en"
    return "mixed"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_SYSTEM_RULES = """You are a strict evidence-verification assistant for an Arabic misinformation \
detector called AMAN Sentinel. You will be given a CLAIM and a list of EVIDENCE PASSAGES, each with \
a unique id. Follow these rules exactly:

1. Reason ONLY from the provided evidence passages. NEVER use outside/world knowledge, and NEVER \
invent facts, sources, names, numbers, or dates that are not in the evidence passages.
2. You may ONLY cite evidence ids that are literally present in the EVIDENCE PASSAGES list given to \
you. Never invent an id. Never cite a passage that does not exist.
3. Distinguish carefully between two different situations:
   - The evidence simply does not discuss the claim at all (no passage is relevant either way) \
-> verdict must be "INSUFFICIENT_EVIDENCE".
   - The evidence actively discusses the claim and says the opposite of what the claim states \
-> verdict must be "CONTRADICTED".
   - The evidence actively discusses the claim and confirms it -> verdict must be "SUPPORTED".
4. If evidence passages disagree with each other about the claim, prefer "CONTRADICTED" or \
"INSUFFICIENT_EVIDENCE" over "SUPPORTED" and explain the disagreement.
5. Handle Modern Standard Arabic, Arabic dialect variation, mixed Arabic/English text, paraphrasing, \
negation, dates, named entities, and numbers carefully — do not be fooled by surface wording \
differences between the claim and the evidence.
6. Write the "explanation" field in the SAME language as the claim (if the claim is Arabic, explain \
in Arabic; if English, explain in English; if mixed, you may mix).
7. Respond with ONLY a single JSON object, no markdown fences, no extra text, in exactly this shape:
{"verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE", "confidence": <0.0-1.0>, \
"explanation": "...", "evidence_ids": ["..."], "claim_language": "ar" | "en" | "mixed"}

If your verdict is SUPPORTED or CONTRADICTED, evidence_ids MUST be non-empty and contain only real \
ids from the EVIDENCE PASSAGES given below. If you cannot support or contradict the claim with the \
given evidence, use INSUFFICIENT_EVIDENCE and evidence_ids may be empty."""


def build_prompt(claim: str, claim_language: str, evidence_pack: dict[str, Any]) -> str:
    evidence_lines = []
    for item in evidence_pack.get("retrieved_evidence", []):
        evidence_lines.append(
            f'- id="{item.get("id")}" source="{item.get("source", "")}": {item.get("text", "")}'
        )
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(no evidence passages provided)"
    valid_ids = [item.get("id") for item in evidence_pack.get("retrieved_evidence", [])]

    return (
        f"{_SYSTEM_RULES}\n\n"
        f"CLAIM (language={claim_language}):\n{claim}\n\n"
        f"EVIDENCE PASSAGES (the ONLY ids you may ever cite: {valid_ids}):\n{evidence_block}\n\n"
        f"Respond with only the JSON object now."
    )


_CORRECTIVE_SUFFIX = (
    "\n\nYour previous response could not be parsed as a single valid JSON object. "
    "Respond again with ONLY the JSON object, no markdown code fences, no commentary, "
    "no text before or after the braces."
)


# --------------------------------------------------------------------------- #
# JSON extraction / repair
# --------------------------------------------------------------------------- #
def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from raw model text.

    Handles: pure JSON, JSON wrapped in ```json ... ``` fences, and JSON with
    leading/trailing prose by locating the first '{' and the matching last
    '}' and attempting to parse that substring.
    """
    if not text:
        return None
    text = text.strip()

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    # Try parsing the whole string directly.
    parsed = _try_parse(text)
    if parsed is not None:
        return parsed

    # Fall back: slice from first '{' to last '}'.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    return None


def _try_parse(candidate: str) -> Optional[dict]:
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# --------------------------------------------------------------------------- #
# Programmatic safeguards
# --------------------------------------------------------------------------- #
def _insufficient_evidence_fallback(claim_language: str, reason: str) -> dict:
    return {
        "verdict": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.0,
        "explanation": f"[SAFEGUARD] {reason}",
        "evidence_ids": [],
        "claim_language": claim_language,
    }


def _validate_and_coerce(parsed: dict, claim_language: str, valid_ids: set[str]) -> tuple[dict, list[str]]:
    """Validate a parsed JSON object against the output schema, repairing/
    coercing where reasonable and recording any safeguard actions taken.

    Returns (result_dict, safeguard_log_messages).
    """
    log: list[str] = []

    verdict = parsed.get("verdict")
    if verdict not in VALID_VERDICTS:
        log.append(f"invalid verdict {verdict!r} — forcing INSUFFICIENT_EVIDENCE")
        return _insufficient_evidence_fallback(claim_language, "model returned an invalid verdict value"), log

    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
        log.append("non-numeric confidence coerced to 0.0")
    confidence = max(0.0, min(1.0, confidence))

    explanation = parsed.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = ""
        log.append("missing/empty explanation")

    raw_evidence_ids = parsed.get("evidence_ids", [])
    if not isinstance(raw_evidence_ids, list):
        raw_evidence_ids = []
        log.append("evidence_ids was not a list — treated as empty")

    # Safeguard (2): every cited id must actually exist in the Evidence Pack.
    cited_ids = [str(i) for i in raw_evidence_ids]
    invalid_citations = [i for i in cited_ids if i not in valid_ids]
    if invalid_citations:
        log.append(
            f"invalid citation(s) {invalid_citations} not present in evidence pack — forcing INSUFFICIENT_EVIDENCE"
        )
        return _insufficient_evidence_fallback(
            claim_language, "model cited evidence id(s) that do not exist in the evidence pack"
        ), log

    claim_lang_out = parsed.get("claim_language", claim_language)
    if claim_lang_out not in VALID_LANGUAGES:
        claim_lang_out = claim_language

    # Safeguard (3): SUPPORTED/CONTRADICTED with no citations is not acceptable.
    if verdict in ("SUPPORTED", "CONTRADICTED") and not cited_ids:
        log.append(
            f"verdict {verdict} had empty evidence_ids — force-downgraded to INSUFFICIENT_EVIDENCE"
        )
        return _insufficient_evidence_fallback(
            claim_lang_out, f"model returned {verdict} without citing any evidence"
        ), log

    result = {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "evidence_ids": cited_ids,
        "claim_language": claim_lang_out,
    }
    return result, log


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def verify_claim_against_evidence(
    claim: str,
    evidence_pack: dict[str, Any],
    provider: LLMProvider,
    claim_language: Optional[str] = None,
) -> dict:
    """Verify a claim against an Evidence Pack using the given LLM provider.

    Args:
        claim: the claim text.
        evidence_pack: {"claim": ..., "retrieved_evidence": [{"id", "text",
            "source", "retrieval_method", "retrieval_score"}, ...]}.
        provider: any LLMProvider (Mock / LocalSmallModel / Ollama / future
            production provider).
        claim_language: 'ar' | 'en' | 'mixed'. If omitted, detected from
            `claim` via `detect_language`.

    Returns:
        A dict matching the output schema, plus a "_safeguards" key listing
        any programmatic corrections that were applied (for transparency /
        smoke-test reporting — not part of the required schema, but harmless
        additional metadata).
    """
    if claim_language is None:
        claim_language = detect_language(claim)
    if claim_language not in VALID_LANGUAGES:
        claim_language = "mixed"

    valid_ids = {str(item.get("id")) for item in evidence_pack.get("retrieved_evidence", [])}

    prompt = build_prompt(claim, claim_language, evidence_pack)

    # MockProvider needs claim/evidence context explicitly since its
    # generate() ignores the prompt text by design.
    if isinstance(provider, MockProvider):
        provider.set_context(claim, sorted(valid_ids))

    all_safeguard_logs: list[str] = []

    raw = provider.generate(prompt)
    parsed = _extract_json_object(raw)

    if parsed is None:
        # Safeguard (1a): malformed JSON — retry once with a corrective note.
        all_safeguard_logs.append("first response was not valid JSON — retrying once")
        logger.warning("Malformed JSON from provider, retrying once.")
        retry_prompt = prompt + _CORRECTIVE_SUFFIX
        if isinstance(provider, MockProvider):
            provider.set_context(claim, sorted(valid_ids))
        raw_retry = provider.generate(retry_prompt)
        parsed = _extract_json_object(raw_retry)

    if parsed is None:
        # Safeguard (1b): still malformed after retry — fall back to INSUFFICIENT_EVIDENCE.
        all_safeguard_logs.append("second response was also not valid JSON — falling back to INSUFFICIENT_EVIDENCE")
        logger.error("Provider returned malformed JSON twice; falling back to INSUFFICIENT_EVIDENCE.")
        result = _insufficient_evidence_fallback(
            claim_language, "model output could not be parsed as JSON after one retry"
        )
        result["_safeguards"] = all_safeguard_logs
        return result

    result, validation_log = _validate_and_coerce(parsed, claim_language, valid_ids)
    all_safeguard_logs.extend(validation_log)
    result["_safeguards"] = all_safeguard_logs
    for msg in validation_log:
        logger.info("Safeguard applied: %s", msg)
    return result
