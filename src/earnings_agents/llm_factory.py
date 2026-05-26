"""LLM factory — returns either an Ollama or Groq client.

Provider is selected via the ``LLM_PROVIDER`` env var (``"ollama"`` default,
or ``"groq"``). Call sites use a uniform interface::

    from earnings_agents.llm_factory import build_llm
    llm = build_llm(format_json=True)
    response: str = llm.invoke(prompt)

Both backends are wrapped so ``llm.invoke(str) -> str`` works identically.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from earnings_agents.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_REQUEST_TIMEOUT,
    LLM_CACHE_DIR,
    LLM_CACHE_ENABLED,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
)

logger = logging.getLogger(__name__)


class _CachedLLM:
    """Transparent disk-cache wrapper for any LLM with an invoke(str)->str interface.

    Responses are cached to ``LLM_CACHE_DIR`` (default ``.llm_cache/``) keyed
    by ``sha256("{provider}:{model}\n{prompt}")``.  The cache persists across
    runs and is never invalidated automatically — delete the directory to reset.

    This is a **development tool** only.  Never enable in production
    (``LLM_CACHE`` env var must be explicitly set to ``1`` / ``true``).
    """

    def __init__(self, llm: Any, model_tag: str, cache_dir: str) -> None:
        import diskcache  # imported lazily so non-dev envs need not install it
        self._llm = llm
        self._model_tag = model_tag
        self._cache = diskcache.Cache(cache_dir)

    def _cache_key(self, prompt: str) -> str:
        payload = f"{self._model_tag}\n{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def invoke(self, prompt: str) -> str:
        key = self._cache_key(prompt)
        if key in self._cache:
            logger.debug("LLM cache HIT  [%s] key=%s", self._model_tag, key[:12])
            return self._cache[key]  # type: ignore[return-value]
        logger.debug("LLM cache MISS [%s] key=%s", self._model_tag, key[:12])
        response: str = self._llm.invoke(prompt)
        self._cache[key] = response
        return response


class _GroqInvokeAdapter:
    """Groq adapter with token-usage logging."""

    def __init__(self, chat_model: Any) -> None:
        self._chat = chat_model

    def invoke(self, prompt: str) -> str:
        msg = self._chat.invoke(prompt)
        # Log token usage from Groq response metadata (OpenAI-compatible format).
        usage = getattr(msg, "response_metadata", {}).get("token_usage") or {}
        if usage:
            logger.debug(
                "groq tokens — prompt: %s  completion: %s  total: %s",
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
                usage.get("total_tokens", "?"),
            )
        content = getattr(msg, "content", msg)
        return content if isinstance(content, str) else str(content)


def build_llm(
    *,
    format_json: bool = False,
    json_schema: dict | None = None,
    request_timeout: float | None = None,
    provider: str | None = None,
) -> Any:
    """Build an LLM client for the configured provider.

    Args:
        format_json: When True, instruct the backend to return strict JSON.
            For Ollama this sets ``format="json"``; for Groq this enables
            ``response_format={"type": "json_object"}``.
        json_schema: Optional JSON Schema dict describing the expected output shape.
            For Ollama, enforces the exact output structure at the model level
            (passed as ``format=<schema>``). For Groq (llama-4-scout), falls back
            to ``json_object`` mode — scout does not support strict schema
            enforcement at the API level; the schema is embedded in the prompt.
            When provided, takes precedence over ``format_json``.
        request_timeout: Per-request HTTP timeout in seconds. Falls back to
            sensible per-provider defaults when None.
        provider: Explicit provider override (``"ollama"`` or ``"groq"``).
            When given, takes precedence over the ``LLM_PROVIDER`` env var.
            Used by the extraction node to escalate to Groq on retry attempts.
    """
    effective_provider = (provider or LLM_PROVIDER).strip().lower()
    if effective_provider == "groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM_PROVIDER=groq requires the langchain-openai package. "
                "Install it with: uv add langchain-openai"
            ) from exc
        if not GROQ_API_KEY:
            raise ValueError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        kwargs: dict[str, Any] = {
            "model": GROQ_MODEL,
            "temperature": 0,
            "api_key": GROQ_API_KEY,
            "base_url": GROQ_BASE_URL,
            "timeout": request_timeout if request_timeout is not None else GROQ_REQUEST_TIMEOUT,
        }
        if json_schema is not None or format_json:
            # llama-4-scout supports json_object but not strict json_schema mode.
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        llm: Any = _GroqInvokeAdapter(ChatOpenAI(**kwargs))
        if LLM_CACHE_ENABLED:
            llm = _CachedLLM(llm, f"groq:{GROQ_MODEL}", LLM_CACHE_DIR)
        return llm

    # Default: Ollama
    from langchain_ollama import OllamaLLM

    kwargs = {
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "temperature": 0,
        "num_ctx": OLLAMA_NUM_CTX,
    }
    if json_schema is not None:
        kwargs["format"] = json_schema  # strict schema enforcement at model level
    elif format_json:
        kwargs["format"] = "json"
    if request_timeout is not None:
        kwargs["client_kwargs"] = {"timeout": request_timeout}
    llm = OllamaLLM(**kwargs)
    if LLM_CACHE_ENABLED:
        llm = _CachedLLM(llm, f"ollama:{OLLAMA_MODEL}", LLM_CACHE_DIR)
    return llm
