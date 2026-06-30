"""LLM factory — returns either an Ollama, Groq, Gemini, or DeepSeek client.

Provider is selected via the ``LLM_PROVIDER`` env var (``"ollama"`` default,
``"groq"``, ``"gemini"``, or ``"deepseek"``). Call sites use a uniform interface::

    from earnings_agents.llm_factory import build_llm
    llm = build_llm(format_json=True)
    response: str = llm.invoke(prompt)

Both backends are wrapped so ``llm.invoke(str) -> str`` works identically.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from typing import Any

from earnings_agents.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_REQUEST_TIMEOUT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_REQUEST_TIMEOUT,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_REQUEST_TIMEOUT,
    GROQ_RPM,
    GROQ_TPM,
    LLM_CACHE_DIR,
    LLM_CACHE_ENABLED,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
)

logger = logging.getLogger(__name__)


class _GroqRateLimiter:
    """Thread-safe sliding-window rate limiter for Groq API (RPM + TPM budgets).

    Enforces two independent 60-second sliding-window budgets:
    * ``rpm`` — maximum requests per minute.
    * ``tpm`` — maximum tokens per minute (input + output combined).

    Call :meth:`acquire` *before* each request.  It blocks until both budgets
    allow the request through, then records the reservation.  After the API
    returns the real token count, call :meth:`update_actual` so the running
    token total stays accurate for subsequent requests.
    """

    _WINDOW: float = 60.0  # sliding window in seconds

    def __init__(self, rpm: int, tpm: int) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._lock = threading.Lock()
        self._req_times: deque[float] = deque()      # request timestamps
        self._tok_log: list[list] = []               # mutable [timestamp, token_count] pairs

    def _expire(self, now: float) -> None:
        """Drop entries older than the sliding window."""
        cutoff = now - self._WINDOW
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()
        self._tok_log = [e for e in self._tok_log if e[0] >= cutoff]

    def acquire(self, estimated_tokens: int) -> list:
        """Block until the RPM and TPM budgets allow a new request.

        Returns a mutable ``[timestamp, token_count]`` entry that can be
        corrected later via :meth:`update_actual`.
        """
        entry: list = [0.0, estimated_tokens]
        while True:
            with self._lock:
                now = time.monotonic()
                self._expire(now)
                current_rpm = len(self._req_times)
                current_tpm = sum(e[1] for e in self._tok_log)
                rpm_ok = current_rpm < self._rpm
                tpm_ok = current_tpm + estimated_tokens <= self._tpm
                if rpm_ok and tpm_ok:
                    entry[0] = now
                    self._req_times.append(now)
                    self._tok_log.append(entry)
                    return entry
                # Calculate minimum sleep to free up budget
                wait = 0.1
                if not rpm_ok and self._req_times:
                    wait = max(wait, self._WINDOW - (now - self._req_times[0]) + 0.1)
                if not tpm_ok:
                    need = current_tpm + estimated_tokens - self._tpm
                    drained = 0
                    for e in self._tok_log:
                        drained += e[1]
                        if drained >= need:
                            wait = max(wait, self._WINDOW - (now - e[0]) + 0.1)
                            break
                logger.info(
                    "Groq rate limit: waiting %.1fs "
                    "(window=%d req/%d tok, budget=%d rpm/%d tpm)",
                    wait, current_rpm, current_tpm, self._rpm, self._tpm,
                )
            time.sleep(wait)

    def update_actual(self, entry: list, actual_tokens: int) -> None:
        """Replace the estimated token count in *entry* with the real API count."""
        with self._lock:
            entry[1] = actual_tokens


# Module-level singleton — shared across all _GroqInvokeAdapter instances so
# concurrent calls from parallel ticker workers respect the same budget.
_groq_rate_limiter = _GroqRateLimiter(rpm=GROQ_RPM, tpm=GROQ_TPM)


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
    """Groq adapter with rate limiting and token-usage logging."""

    def __init__(self, chat_model: Any) -> None:
        self._chat = chat_model

    def invoke(self, prompt: str) -> str:
        # Estimate token cost: chars→tokens (÷4) plus a conservative output budget.
        # The real count corrects the reservation once the API responds.
        estimated_tokens = len(prompt) // 4 + 800
        token_entry = _groq_rate_limiter.acquire(estimated_tokens)
        try:
            msg = self._chat.invoke(prompt)
        except Exception:
            # Release the reservation on error so the budget isn't permanently consumed.
            _groq_rate_limiter.update_actual(token_entry, 0)
            raise
        # Log and correct token budget with actual usage.
        usage = getattr(msg, "response_metadata", {}).get("token_usage") or {}
        if usage:
            logger.debug(
                "groq tokens — prompt: %s  completion: %s  total: %s",
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
                usage.get("total_tokens", "?"),
            )
            actual_tokens = usage.get("total_tokens", estimated_tokens)
            _groq_rate_limiter.update_actual(token_entry, actual_tokens)
        content = getattr(msg, "content", msg)
        return content if isinstance(content, str) else str(content)


class _GeminiInvokeAdapter:
    """Gemini adapter built on the official ``google-genai`` SDK.

    Wraps ``client.models.generate_content`` so that ``invoke(str) -> str``
    matches the uniform interface used by the rest of the pipeline. JSON mode
    is requested via ``response_mime_type="application/json"``; the expected
    schema (when provided) is embedded in the prompt by the caller, mirroring
    the Groq adapter's behaviour.
    """

    def __init__(self, client: Any, model: str, config: Any) -> None:
        self._client = client
        self._model = model
        self._config = config

    def invoke(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config,
        )
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            logger.debug(
                "gemini tokens — prompt: %s  candidates: %s  total: %s",
                getattr(usage, "prompt_token_count", "?"),
                getattr(usage, "candidates_token_count", "?"),
                getattr(usage, "total_token_count", "?"),
            )
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else str(text)


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
        provider: Explicit provider override (``"ollama"``, ``"groq"`` or
            ``"gemini"``). When given, takes precedence over the
            ``LLM_PROVIDER`` env var. Used by the extraction node to escalate
            to a cloud provider on retry attempts.
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

    if effective_provider == "deepseek":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM_PROVIDER=deepseek requires the langchain-openai package. "
                "Install it with: uv add langchain-openai"
            ) from exc
        if not DEEPSEEK_API_KEY:
            raise ValueError("LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set")
        kwargs = {
            "model": DEEPSEEK_MODEL,
            "temperature": 0,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
            "timeout": request_timeout if request_timeout is not None else DEEPSEEK_REQUEST_TIMEOUT,
        }
        if json_schema is not None or format_json:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        llm = _GroqInvokeAdapter(ChatOpenAI(**kwargs))
        if LLM_CACHE_ENABLED:
            llm = _CachedLLM(llm, f"deepseek:{DEEPSEEK_MODEL}", LLM_CACHE_DIR)
        return llm

    if effective_provider == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM_PROVIDER=gemini requires the google-genai package. "
                "Install it with: uv add google-genai"
            ) from exc
        if not GEMINI_API_KEY:
            raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        timeout_s = (
            request_timeout if request_timeout is not None else GEMINI_REQUEST_TIMEOUT
        )
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            # google-genai expects the HTTP timeout in milliseconds.
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        config_kwargs: dict[str, Any] = {"temperature": 0}
        if json_schema is not None or format_json:
            # gemini-2.5 supports native JSON mode. The schema (when present) is
            # embedded in the prompt by the caller, mirroring the Groq adapter.
            config_kwargs["response_mime_type"] = "application/json"
        gen_config = types.GenerateContentConfig(**config_kwargs)
        llm = _GeminiInvokeAdapter(client, GEMINI_MODEL, gen_config)
        if LLM_CACHE_ENABLED:
            llm = _CachedLLM(llm, f"gemini:{GEMINI_MODEL}", LLM_CACHE_DIR)
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
