"""LLM factory — returns either Ollama, OpenAI, or Groq clients.

Provider is selected via the ``LLM_PROVIDER`` env var (``"ollama"`` default,
``"openai"``, or ``"groq"``). Call sites use a uniform interface::

    from earnings_agents.llm_factory import build_llm
    llm = build_llm(format_json=True)
    response: str = llm.invoke(prompt)

Both backends are wrapped so ``llm.invoke(str) -> str`` works identically.
"""
from __future__ import annotations

import logging
from typing import Any

from earnings_agents.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_REQUEST_TIMEOUT,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class _OpenAIInvokeAdapter:
    """Make ``ChatOpenAI`` expose a string-in / string-out ``invoke`` like Ollama."""

    def __init__(self, chat_model: Any) -> None:
        self._chat = chat_model

    def invoke(self, prompt: str) -> str:
        msg = self._chat.invoke(prompt)
        # ChatOpenAI returns an AIMessage with .content (str)
        content = getattr(msg, "content", msg)
        return content if isinstance(content, str) else str(content)


class _GroqInvokeAdapter:
    """Pass-through Groq adapter — relies on server-side rate limiting."""

    def __init__(self, chat_model: Any) -> None:
        self._chat = chat_model

    def invoke(self, prompt: str) -> str:
        msg = self._chat.invoke(prompt)
        content = getattr(msg, "content", msg)
        return content if isinstance(content, str) else str(content)


def build_llm(*, format_json: bool = False, request_timeout: float | None = None) -> Any:
    """Build an LLM client for the configured provider.

    Args:
        format_json: When True, instruct the backend to return strict JSON.
            For Ollama this sets ``format="json"``; for OpenAI/Groq this
            enables ``response_format={"type": "json_object"}``.
        request_timeout: Per-request HTTP timeout in seconds. Falls back to
            sensible per-provider defaults when None.
    """
    if LLM_PROVIDER == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM_PROVIDER=openai requires the langchain-openai package. "
                "Install it with: uv add langchain-openai"
            ) from exc
        if not OPENAI_API_KEY:
            raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {
            "model": OPENAI_MODEL,
            "temperature": 0,
            "api_key": OPENAI_API_KEY,
            "timeout": request_timeout if request_timeout is not None else OPENAI_REQUEST_TIMEOUT,
        }
        if format_json:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return _OpenAIInvokeAdapter(ChatOpenAI(**kwargs))

    if LLM_PROVIDER == "groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLM_PROVIDER=groq requires the langchain-openai package. "
                "Install it with: uv add langchain-openai"
            ) from exc
        if not GROQ_API_KEY:
            raise ValueError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        kwargs = {
            "model": GROQ_MODEL,
            "temperature": 0,
            "api_key": GROQ_API_KEY,
            "base_url": GROQ_BASE_URL,
            "timeout": request_timeout if request_timeout is not None else GROQ_REQUEST_TIMEOUT,
        }
        if format_json:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return _GroqInvokeAdapter(ChatOpenAI(**kwargs))

    # Default: Ollama
    from langchain_ollama import OllamaLLM

    kwargs = {
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "temperature": 0,
        "num_ctx": OLLAMA_NUM_CTX,
    }
    if format_json:
        kwargs["format"] = "json"
    if request_timeout is not None:
        kwargs["client_kwargs"] = {"timeout": request_timeout}
    return OllamaLLM(**kwargs)
