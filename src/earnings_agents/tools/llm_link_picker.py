"""LLM-based earnings release URL picker for IR page discovery.

Provides :func:`pick_earnings_url`, which asks the LLM to identify the most
recent quarterly earnings press release link from a list of IR page links.

Extracted from ``nodes/discover_earnings_release.py`` so the LLM call,
schema enforcement, retry logic, and URL validation are independently testable.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# JSON schema enforced by Ollama at the model level; used as a json_object hint
# for Groq.  Guarantees the LLM returns {"url": "...", "reason": "..."} without
# markdown fences or extra fields.
DISCOVERY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["url", "reason"],
}

_PROMPT_TEMPLATE = (
    "You are a financial data assistant.\n"
    "Below is a list of links from the investor relations (IR) page of "
    "{company_name} ({ticker}).\n\n"
    "Task: identify the single link that points to the most recent quarterly "
    "earnings press release or earnings results document (PDF or webpage). "
    "Prefer PDF links. Ignore annual reports, proxy statements, ESG reports, "
    "and investor presentations unless no press release exists.\n\n"
    "Respond with ONLY a JSON object — no markdown, no extra text:\n"
    '  {{"url": "<full URL>", "reason": "<one sentence>"}}\n\n'
    "Links:\n{links_text}"
)

_STRICT_PREFIX = (
    "IMPORTANT: You MUST return a valid URL starting with http:// or https://. "
    "Do not return placeholder text, relative paths, or an empty string.\n\n"
)


def _is_valid_url(url: str) -> bool:
    """Return True only for absolute http/https URLs."""
    return bool(url and (url.startswith("http://") or url.startswith("https://")))


def pick_earnings_url(
    links_text: str,
    company_name: str,
    ticker: str,
    llm: Any,
) -> str | None:
    """Ask the LLM to choose the earnings release URL from a formatted link list.

    Parameters
    ----------
    links_text:
        Formatted list of IR page links (markdown ``- [text](url)`` lines).
    company_name:
        Display name of the company.
    ticker:
        Ticker symbol, used in the prompt and log messages.
    llm:
        LLM client with an ``invoke(prompt) -> str`` method.  Should be built
        with ``build_llm(json_schema=DISCOVERY_SCHEMA)`` for schema enforcement.

    Returns
    -------
    str or None
        An absolute https/http URL on success, ``None`` if parsing or
        validation fails after one retry.

    Raises
    ------
    ValueError
        When the LLM returns an invalid URL even after the strict-prefix retry.
    json.JSONDecodeError
        When the LLM response cannot be decoded as JSON.
    """
    prompt = _PROMPT_TEMPLATE.format(
        company_name=company_name,
        ticker=ticker,
        links_text=links_text,
    )

    response: str = llm.invoke(prompt)
    result = json.loads(response.strip())
    file_url: str = result.get("url", "").strip()

    # Validate; retry once with a stricter instruction if the model returned a
    # placeholder or relative path despite the schema.
    if not _is_valid_url(file_url):
        logger.warning(
            "LLM returned invalid URL %r for %s — retrying with stricter prompt",
            file_url, ticker,
        )
        response = llm.invoke(_STRICT_PREFIX + prompt)
        result = json.loads(response.strip())
        file_url = result.get("url", "").strip()

    if not _is_valid_url(file_url):
        raise ValueError(f"LLM returned invalid URL after retry: {file_url!r}")

    logger.info(
        "IR discovery for %s → %s (reason: %s)",
        ticker, file_url, result.get("reason"),
    )
    return file_url
