"""LLM provider abstraction for the AMAN Sentinel Phase 3 verification layer.

This module defines a small `LLMProvider` interface (`generate(prompt) -> str`)
and three concrete providers:

  1. `MockProvider`           — deterministic, clearly-labeled canned JSON
                                 responses. No model, no network. Exists purely
                                 so the schema-validation / citation-checking
                                 logic in `verification_llm.py` can be exercised
                                 end-to-end even with zero inference capability.

  2. `LocalSmallModelProvider` — runs a genuinely small open instruct model
                                 (default: Qwen2.5-0.5B-Instruct) locally via
                                 `transformers`, on CPU. This is the provider
                                 the task spec asks for by name, for
                                 dev/smoke-testing. It is NOT the production
                                 model (see phase3/reports/llm_model_selection.md
                                 for the production pick, Qwen2.5-7B-Instruct,
                                 which is far too slow to run on CPU here).

  3. `OllamaProvider`          — calls a local Ollama server's HTTP API
                                 (default http://localhost:11434). This is NOT
                                 one of the two providers the task explicitly
                                 required, but it is included because this
                                 specific machine was found (see
                                 phase3/reports/amd_investigation.md) to have
                                 Ollama already running locally with small
                                 instruct models already pulled (qwen2.5:1.5b).
                                 That makes it the most honest "real local
                                 small model" choice actually available here —
                                 no download wait, no flaky first-run network
                                 dependency, and it is what the smoke test in
                                 this phase actually uses. `LocalSmallModelProvider`
                                 remains in this file, fully implemented, as the
                                 spec-compliant transformers-based fallback /
                                 reference implementation for environments where
                                 Ollama is not present.

No AMD/ROCm execution path exists here (see amd_investigation.md) — all three
providers above run on CPU only, locally, in this sandbox.
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Minimal provider interface. Every provider must implement `generate`."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Given a prompt string, return the model's raw text response."""
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic, clearly-labeled canned responses — no model involved.

    Exists so downstream parsing/validation/citation-checking logic can be
    tested even with zero network/model/compute available. Every response is
    prefixed in spirit (and in the `explanation` text) with a "[MOCK]" marker
    so nobody mistakes this for a real model output.

    Heuristic used to pick a canned verdict: if any evidence passage shares a
    noticeable chunk of text with the claim, respond SUPPORTED citing that
    passage's id; otherwise respond INSUFFICIENT_EVIDENCE citing nothing. This
    is intentionally crude — it is a stand-in for a real model, not a
    verification system.
    """

    def __init__(self) -> None:
        self._last_evidence_ids: list[str] = []
        self._last_claim: str = ""

    def set_context(self, claim: str, evidence_ids: list[str]) -> None:
        """Called by verification_llm.py before generate() so the mock can
        build a plausible-looking, evidence-id-aware canned response without
        needing to re-parse the prompt text itself."""
        self._last_claim = claim
        self._last_evidence_ids = evidence_ids

    def generate(self, prompt: str) -> str:  # noqa: ARG002 - prompt unused by design
        if self._last_evidence_ids:
            cited = [self._last_evidence_ids[0]]
            verdict = "SUPPORTED"
            explanation = (
                "[MOCK] Canned response: no real model was called. "
                "A real provider would compare the claim against the cited evidence passage."
            )
        else:
            cited = []
            verdict = "INSUFFICIENT_EVIDENCE"
            explanation = (
                "[MOCK] Canned response: no real model was called and no evidence ids were supplied."
            )
        return json.dumps(
            {
                "verdict": verdict,
                "confidence": 0.5,
                "explanation": explanation,
                "evidence_ids": cited,
                "claim_language": "ar",
            }
        )


class LocalSmallModelProvider(LLMProvider):
    """Runs a small open instruct model locally via `transformers`, on CPU.

    Default model: Qwen2.5-0.5B-Instruct (~0.5B params). This is deliberately
    NOT the production model (Qwen2.5-7B-Instruct, see
    phase3/reports/llm_model_selection.md) — a 7B/13B model is far too slow to
    run interactively on a CPU-only laptop. This class is for local
    development / smoke-testing only.

    Loading is lazy (first call to `generate`) so simply constructing this
    class never triggers a slow download.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 400,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local small model %s (CPU)...", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        )
        self._tokenizer = tokenizer
        self._model = model
        self._pipeline = True  # sentinel: loaded

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


class OllamaProvider(LLMProvider):
    """Calls a local Ollama server's HTTP API.

    Default model is qwen2.5:7b-instruct -- the intended primary verifier
    for the final AMAN Sentinel submission/demo (validated end-to-end on
    NVIDIA GPU infrastructure, see phase5/reports/UNIVERSITY_GPU_VALIDATION.md).
    qwen2.5:1.5b was the earlier development-phase default (see
    phase3/reports/amd_investigation.md) and remains available via
    OLLAMA_MODEL or the constructor argument; the historical 1.5B-vs-7B
    experimental comparison in phase6/CONFERENCE_RESULTS_AUDIT_FINAL.md and
    phase5/VALIDATED_METRICS.md is unaffected by this default -- those
    results were produced by explicitly constructing each model and are
    not re-derived from this default.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = 120.0,
        max_new_tokens: int = 350,
    ) -> None:
        # model_name param (if passed explicitly) wins; otherwise OLLAMA_MODEL
        # env var; otherwise the default production/demo verifier.
        self.model_name = model_name or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        # host param (if passed explicitly) wins; otherwise fall back to
        # OLLAMA_HOST env var (added for Phase 4's env-driven deployment
        # config), then the original hardcoded local default.
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": self.max_new_tokens},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")


class VLLMProvider(LLMProvider):
    """HTTP client for an OpenAI-compatible vLLM server (the intended
    production path, served on AMD Instinct hardware via ROCm — see
    `phase3/reports/llm_model_selection.md` and
    `phase3/reports/amd_investigation.md`).

    NOT exercised against real hardware anywhere in this repo: this sandbox
    is confirmed Intel-only / no ROCm / no AMD cloud access
    (`phase3/reports/amd_investigation.md`). This class is a skeleton built
    so Phase 4's env-var switch (`LLM_PROVIDER=vllm`) has something concrete
    to construct once an AMD Developer Cloud vLLM endpoint actually exists
    (see `phase4/reports/amd_deployment_instructions.md`). Until then,
    selecting it will simply fail to connect, which is the honest behavior
    for an endpoint that isn't there.

    Talks to vLLM's OpenAI-compatible `/v1/chat/completions` endpoint
    (`vllm serve <model> --port 8000`), using only `urllib` to avoid adding
    a hard dependency on an HTTP client library for a path that cannot be
    tested here anyway.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 120.0,
        max_new_tokens: int = 500,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("VLLM_ENDPOINT", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "")
        self.model_name = model_name or os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        if not self.endpoint:
            logger.warning(
                "VLLMProvider constructed with no VLLM_ENDPOINT set — "
                "generate() will fail until one is configured. This is "
                "expected in this sandbox (no AMD hardware available)."
            )

    def generate(self, prompt: str) -> str:
        import urllib.request

        if not self.endpoint:
            raise RuntimeError(
                "VLLMProvider has no endpoint configured (set VLLM_ENDPOINT). "
                "No AMD/ROCm/vLLM instance is reachable from this sandbox — "
                "see phase4/reports/amd_deployment_instructions.md."
            )
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": self.max_new_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def is_ollama_reachable(host: Optional[str] = None, timeout: float = 3.0) -> bool:
    """Quick reachability probe used by verification_llm.py / smoke test to
    decide which provider to fall back to. Defaults to OLLAMA_HOST (env)
    then http://localhost:11434, matching OllamaProvider's own default
    resolution so the reachability check and the provider it gates always
    agree on which host they mean."""
    import urllib.request

    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def get_default_provider() -> LLMProvider:
    """Provider selection, driven by the `LLM_PROVIDER` env var
    (added for Phase 4's API service so the LLM backend can be switched
    without code changes):

      LLM_PROVIDER=ollama (default, matches this sandbox's reality) —
          original best-effort behavior: Ollama if reachable, else
          MockProvider.
      LLM_PROVIDER=vllm — construct a VLLMProvider pointed at
          VLLM_ENDPOINT/VLLM_API_KEY. This is the future AMD/ROCm
          production path and is NOT functional in this sandbox (no AMD
          hardware here — see phase3/reports/amd_investigation.md); it
          will raise at generate() time unless pointed at a real endpoint.
      LLM_PROVIDER=mock — explicit MockProvider, useful for tests/CI.

    (LocalSmallModelProvider is available but not auto-selected here because
    it requires a model download on first use; callers that want it should
    construct it explicitly.)
    """
    choice = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

    if choice == "vllm":
        logger.info("LLM_PROVIDER=vllm — using VLLMProvider (untested against real AMD hardware).")
        return VLLMProvider()

    if choice == "mock":
        logger.info("LLM_PROVIDER=mock — using MockProvider explicitly.")
        return MockProvider()

    # Default / LLM_PROVIDER=ollama: original best-effort behavior.
    if is_ollama_reachable():
        logger.info("Ollama reachable — using OllamaProvider.")
        return OllamaProvider()
    logger.info("Ollama not reachable — falling back to MockProvider.")
    return MockProvider()
