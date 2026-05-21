from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_ollama import OllamaLLM

from earnings_agents.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EXTRACTION_MAX_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_CONCURRENCY,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
)
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)

# Chunking parameters — sourced from config (which reads .env)
_CHUNK_SIZE = CHUNK_SIZE
_CHUNK_OVERLAP = CHUNK_OVERLAP
_OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "75"))
_CHUNK_MAX_RETRIES = int(os.getenv("CHUNK_MAX_RETRIES", "1"))

# Semaphore that limits the number of concurrent Ollama calls across ALL
# company threads. Prevents timeout storms when running many tickers in
# parallel against a single-threaded local Ollama instance.
_OLLAMA_SEMAPHORE = threading.Semaphore(OLLAMA_CONCURRENCY)

_PROMPT_TEMPLATE = """\
You are a financial data extraction assistant.

Extract ALL numeric financial metrics from the text excerpt below for {company_name} ({ticker}).
This is chunk {chunk_num} of {total_chunks} of the full document.
{focus_hint}{scale_hint}{period_hint}
Return ONLY a flat JSON object — no markdown fences, no extra text, no nested objects.

FIRST field must be "__scale__":
  Set it to the unit of the table in this excerpt:
  - "millions"  → table header says "(In millions)" or "(in millions of dollars)"
  - "thousands" → table header says "(In thousands)"
  - "billions"  → table header says "(In billions)"
  - "as-is"     → values come from narrative prose only (no table, or values already
                   stated in full dollars, e.g. "$82.9 billion", "$4.27 per share")

SECOND field must be "__period__":
  Earnings releases often show side-by-side columns for multiple time periods
  (e.g. "Q1 2026 | Q1 2025" or "Nine Months Ended Mar 31, 2026 | Nine Months Ended Mar 31, 2025").
  - Set it to the label of the MOST RECENT period column, exactly as printed in the document
    (e.g. "Three Months Ended March 31, 2026").
  - If only one period is present, set it to that period's label.
  - If no period label is visible in this excerpt, set it to null.

PERIOD RULE — CRITICAL when multiple period columns are present:
  Extract numeric values ONLY from the MOST RECENT period column (usually the leftmost data column).
  NEVER capture values from prior-period or year-ago comparison columns.
  If a metric appears in multiple columns, take ONLY the value under the most recent column header.

ALL OTHER fields:
  - Keys   : the EXACT metric label as it appears in the document (company's own wording)
  - Values : the RAW numeric value EXACTLY as printed in the table or text, or null

IMPORTANT — report RAW numbers, do NOT scale yourself:
  If the table says "(In millions)" and shows "82,886" → report 82886  (NOT 82886000000).
  The scaling multiplier will be applied automatically from the __scale__ field.

Exceptions (always report as-is regardless of __scale__):
  • EPS / earnings per share: raw decimal (e.g. "4.27" stays 4.27).
  • Percentage values (margin %, growth %): number 0–100 with no scaling.

Other rules:
  1. Use the company’s exact terminology.
  2. Return null for any field not present in this excerpt.
  3. Return {{"__scale__": "as-is", "__period__": null}} if no financial metrics are found.
  4. Do NOT include non-numeric fields (dates, names, addresses, descriptions).  5. METRIC KEY RULES — keys must be SHORT official labels (≤ 8 words):
     • Use the table row header or bold label exactly as printed.
     • REJECT any key that is a full sentence or contains change-description
       verbs: "increased", "decreased", "grew", "expanded", "saw", "reflected",
       "improved", "declined", "up", "down". Those are commentary, not labels.
     • REJECT keys containing comparison phrases like "grew by", "up X%",
       "vs prior year", "rose to", "fell to".
     • If a narrative sentence mentions a metric value, extract ONLY the numeric
       value under the nearest official table label instead.
  6. NON-DOLLAR COUNTS are never scaled — report raw value for any key
     containing: "employee", "headcount", "number of shares", "share count",
     "basis points", "percentage points", "production", "deliveries",
     "stations", "connectors", "subscriptions", "days of supply",
     "lease count", "units".
  7. PERCENTAGE METRICS are never scaled — report as a number 0–100 for any
     key containing: "operating margin", "net margin", "gross margin %",
     "profit margin", "margin %", "growth rate".
Text excerpt:
\"\"\"
{text}
\"\"\"
"""

# Keys whose values are percentages, per-share amounts, or non-dollar counts —
# excluded from the sanity check AND from scale multiplication.
# "margin" intentionally omitted: "Gross margin" is a dollar amount, not a %.
_PCT_OR_PER_SHARE_PATTERNS = re.compile(
    r"(%|percent|\bgrowth\b|\bratio\b|\beps\b|per share|\byield\b|\brate\b|\byoy\b|\bpct\b"
    r"|\bemployee\b|\bheadcount\b|basis points|percentage points"
    # Operational unit counts — physical quantities, never dollar-scaled
    r"|\bproduction\b|\bdeliveries\b|\bdelivered\b"
    r"|(?:super)?charger.{0,12}(?:station|connector)"
    r"|\bstations?\b|\bconnectors?\b"
    r"|\bdays.{0,5}supply\b|\blease count\b"
    r"|\bactive\b.{0,20}\bsubscriptions?\b|\bfsd subscriptions?\b"
    # Percentage-type margins (distinct from 'Gross margin' which is a dollar amount)
    r"|\boperating margin\b|\bnet margin\b|\bprofit margin\b|\bmargin\s*%)",
    re.IGNORECASE,
)

# EPS denominators ("Number of shares used …" / "Shares used in computing …") contain
# "per share" in their key but ARE table-scaled quantities (millions or thousands of shares).
# This pattern overrides _PCT_OR_PER_SHARE_PATTERNS for those keys.
_SHARE_COUNT_PATTERN = re.compile(
    r"\bnumber of shares\b|\bshares used\b|\bweighted.{0,15}average.{0,15}shares\b",
    re.IGNORECASE,
)

# Raw share-count values larger than this are assumed already at full count
# (e.g. 14_673_278_000 already expanded), so don't re-multiply.
_SHARE_COUNT_RAW_MAX = 100_000_000  # 100 M shares in report units → already full if exceeded

# Scale multipliers keyed by the __scale__ sentinel the LLM returns.
_SCALE_MULTIPLIERS: dict[str, int] = {
    "millions": 1_000_000,
    "thousands": 1_000,
    "billions": 1_000_000_000,
}

# Raw table values larger than this are assumed to already be full USD (e.g.
# a narrative value like 82_900_000_000) and won't be re-multiplied.
_TABLE_RAW_MAX = 10_000_000  # 10 M raw → $10T if ×1M — implausible, so skip

# Minimum fraction of the largest revenue-like value that a dollar field must
# have to be considered plausible (filters out residual unscaled cells).
_MIN_DOLLAR_FRACTION = 0.001   # 0.1 % of revenue

# Major financial metrics that should always represent a significant share of
# revenue.  When their post-scale value falls below _MIN_DOLLAR_FRACTION we
# attempt a ×1 000 scale correction (one tier up: millions → billions, etc.)
# before discarding.  Non-major metrics (specific investing / financing line
# items) can be legitimately tiny and are kept as-is.
_MAJOR_METRIC_RX = re.compile(
    r"\brevenue\b|\bnet sales\b|\bsales\b"
    r"|\bgross profit\b|\bgross margin\b"
    r"|\boperating income\b|\boperating profit\b|\boperating loss\b"
    r"|\bebit\b|\bebitda\b"
    r"|\bnet income\b|\bnet earnings\b|\bnet loss\b"
    r"|\boperating cash flow\b|\bcash (?:from|provided by) operations\b",
    re.IGNORECASE,
)
# A ×1 000 rescaled value is accepted only when it stays below this multiple
# of revenue — guards against inflating genuinely-tiny items.
_RESCALE_UPPER_MULTIPLE = 3.0

# Pre-scan patterns applied to the full document text BEFORE chunking.
# Detected scale/period are injected as confirmed ground truth into every
# chunk prompt, eliminating wrong-scale errors that occur when the
# "(In millions)" table header only appears in the first chunk.
_PRESCAN_SCALE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\(in millions\b", re.I), "millions"),
    (re.compile(r"\(in thousands\b", re.I), "thousands"),
    (re.compile(r"\(in billions\b", re.I), "billions"),
]
# Detects when share counts use a DIFFERENT scale than dollar amounts.
# e.g. "(In millions, except number of shares which are reflected in thousands"
_PRESCAN_SHARES_IN_THOUSANDS_RX = re.compile(
    r"shares\s+(?:which\s+are\s+)?(?:reflected\s+)?in\s+thousands"
    r"|number\s+of\s+shares[^)]{0,60}in\s+thousands"
    r"|except[^)]{0,60}shares[^)]{0,60}thousands",
    re.I,
)
_PRESCAN_PERIOD_RX = re.compile(
    # Standard SEC form: "Three Months Ended March 31, 2026"
    r"(?:Three|Six|Nine)\s+Months?\s+Ended\s+"
    r"(?:March|June|September|December|Jan(?:uary)?|Feb(?:ruary)?"
    r"|Apr(?:il)?|May|Jul(?:y)?|Aug(?:ust)?|Oct(?:ober)?|Nov(?:ember)?)\s+"
    r"\d{1,2},\s*\d{4}"
    # Q-style: Q1 2026, Q1-2026, Q1'26 (used by Netflix, Tesla, etc.)
    r"|Q[1-4][\s\-](?:20\d{2})"
    # Spelled-out quarter: "First Quarter 2026", "First Quarter Fiscal 2026"
    r"|(?:First|Second|Third|Fourth)\s+Quarter\s+(?:Fiscal\s+)?20\d{2}",
    re.I,
)


def _invoke_chunk_with_retry(
    prompt: str,
    chunk_num: int,
    total_chunks: int,
    ticker: str,
    shares_multiplier: int = 1,
    max_retries: int = _CHUNK_MAX_RETRIES,
) -> "dict[str, Any] | None":
    """Invoke the LLM for one chunk, retrying with a stricter prefix on parse failure.

    Each worker creates its own Ollama client instance; sharing one instance
    across threads can block intermittently under parallel load.
    """
    llm = OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0,
        num_ctx=OLLAMA_NUM_CTX,
        format="json",
        client_kwargs={"timeout": _OLLAMA_REQUEST_TIMEOUT},
    )
    for attempt in range(max_retries + 1):
        prefix = (
            ""
            if attempt == 0
            else (
                "CRITICAL: Your previous response could not be parsed as JSON. "
                "Respond with ONLY a raw JSON object starting with '{'. "
                "Absolutely no markdown fences, no preamble, no explanation.\n\n"
            )
        )
        logger.info(
            "Chunk %d/%d attempt %d started for %s",
            chunk_num, total_chunks, attempt + 1, ticker,
        )
        try:
            with _OLLAMA_SEMAPHORE:
                response: str = llm.invoke(prefix + prompt)
            logger.debug(
                "Chunk %d/%d attempt %d raw response for %s: %r",
                chunk_num, total_chunks, attempt + 1, ticker, response[:300],
            )
            parsed = _parse_llm_response(response, shares_multiplier)
            if parsed is not None:
                return parsed
            logger.warning(
                "Chunk %d/%d attempt %d returned unparseable response for %s",
                chunk_num, total_chunks, attempt + 1, ticker,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Chunk %d/%d attempt %d failed for %s: %s",
                chunk_num, total_chunks, attempt + 1, ticker, exc,
            )
    return None


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks of *chunk_size* chars."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _prescan_document(raw_text: str) -> tuple[str | None, str | None, str | None]:
    """Scan the full document once for scale and current reporting period.

    Returns (scale, shares_scale, period) — any may be None if not detected.

    ``shares_scale`` is set when the document explicitly states that share
    counts use a different scale than dollar amounts (e.g. Apple's
    "in millions, except number of shares which are reflected in thousands").
    When ``shares_scale`` is None the dollar scale is used for share counts.

    These are injected as confirmed ground truth into every chunk prompt,
    eliminating wrong-scale errors that arise when middle chunks don't see
    the "(In millions)" table header that only appeared in chunk 1.
    """
    scale: str | None = None
    for pattern, scale_name in _PRESCAN_SCALE:
        if pattern.search(raw_text):
            scale = scale_name
            break

    shares_scale: str | None = None
    if _PRESCAN_SHARES_IN_THOUSANDS_RX.search(raw_text):
        shares_scale = "thousands"

    period: str | None = None
    m = _PRESCAN_PERIOD_RX.search(raw_text)
    if m:
        period = m.group(0)

    return scale, shares_scale, period


def _parse_llm_response(response: str, shares_multiplier: int = 1) -> dict[str, Any] | None:
    """Strip markdown fences, parse JSON, and apply the __scale__ multiplier.

    The LLM is asked to report raw table values plus a ``__scale__`` sentinel.
    Python applies the exact multiplication so the LLM never has to do arithmetic.

    ``shares_multiplier`` is applied to share-denominator fields (e.g. shares
    used in EPS calculation) and may differ from the dollar multiplier when the
    document explicitly states a different scale for share counts (e.g. Apple
    reports dollars in millions but share counts in thousands).
    """
    cleaned = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    end_brace = cleaned.rfind("}")
    if end_brace >= 0:
        cleaned = cleaned[: end_brace + 1]
    try:
        parsed: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    # Extract __scale__ and apply the multiplier to all dollar-amount fields.
    scale_str = str(parsed.pop("__scale__", "as-is")).lower()
    multiplier = _SCALE_MULTIPLIERS.get(scale_str, 1)
    if multiplier > 1 or shares_multiplier > 1:
        for k, v in list(parsed.items()):
            if v is None or not isinstance(v, (int, float)):
                continue
            is_share_count = bool(_SHARE_COUNT_PATTERN.search(k))
            if is_share_count and shares_multiplier > 1:
                # Share-denominator fields use the shares multiplier.
                # Skip the _TABLE_RAW_MAX guard — share counts are often > 10 M
                # (e.g. Apple reports ~14.7 M thousands = 14.7 B shares).
                if abs(v) < _SHARE_COUNT_RAW_MAX:
                    parsed[k] = v * shares_multiplier
            elif (
                not is_share_count
                and not _PCT_OR_PER_SHARE_PATTERNS.search(k)
                and multiplier > 1
                and abs(v) < _TABLE_RAW_MAX   # skip values already at full USD scale
            ):
                parsed[k] = v * multiplier

    return parsed


def _merge_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-chunk extraction dicts into one dict of all discovered metrics.

    Strategy per key:
    - First non-null value wins (null from an earlier chunk never blocks a later real value).
    - For string values (guidance narrative etc.) the longest non-null wins.
    - Dollar-amount fields that are implausibly small relative to the largest
      revenue-like value are discarded (unscaled table cells).
    """
    merged: dict[str, Any] = {}

    for result in results:
        for key, value in result.items():
            if value is None:
                merged.setdefault(key, None)
            elif isinstance(value, str):
                existing = merged.get(key)
                if existing is None or len(value) > len(str(existing)):
                    merged[key] = value
            else:
                if merged.get(key) is None:
                    merged[key] = value

    # Sanity check: find the largest "revenue"-labelled value as reference.
    revenue_ref = max(
        (v for k, v in merged.items()
         if v is not None and isinstance(v, (int, float))
         and "revenue" in k.lower()
         and not _PCT_OR_PER_SHARE_PATTERNS.search(k)),
        default=None,
    )

    if revenue_ref:
        threshold = revenue_ref * _MIN_DOLLAR_FRACTION
        rescale_upper = revenue_ref * _RESCALE_UPPER_MULTIPLE
        for key in list(merged.keys()):
            val = merged[key]
            if (
                val is None
                or not isinstance(val, (int, float))
                or _PCT_OR_PER_SHARE_PATTERNS.search(key)
                or abs(val) >= threshold
            ):
                continue  # value is fine as-is

            # Value is below the plausibility threshold.
            if _MAJOR_METRIC_RX.search(key):
                # Major metrics must be large.  Try a ×1 000 scale correction
                # (one tier up, e.g. the LLM returned 74.9 in a billions table
                # instead of 74 900 in a millions table).
                rescaled = val * 1_000
                if threshold <= abs(rescaled) <= rescale_upper:
                    logger.debug(
                        "Scale-correcting major metric %r: %s → %s",
                        key, val, rescaled,
                    )
                    merged[key] = rescaled
                else:
                    logger.warning(
                        "Discarding implausible major metric %r=%s "
                        "(< %.1f%% of revenue ref %s; ×1000 rescale also fails)",
                        key, val, _MIN_DOLLAR_FRACTION * 100, revenue_ref,
                    )
                    merged[key] = None
            # Non-major metrics below threshold are kept as-is — they can be
            # legitimately small (e.g. $26 M investing item for an $80 B company).

    # Drop keys where the final value is None to keep the stored document clean.
    return {k: v for k, v in merged.items() if v is not None}


def _find_first(
    metrics: dict[str, Any], pattern: str
) -> "tuple[str, Any] | tuple[None, None]":
    """Return (key, value) for the first metric key matching *pattern*."""
    rx = re.compile(pattern, re.IGNORECASE)
    for k, v in metrics.items():
        if rx.search(k):
            return k, v
    return None, None


def _validate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic post-merge cross-field consistency checks.

    Catches values the LLM accidentally pulled from the wrong table section
    or period column by verifying known financial identities:
      1. Free Cash Flow ≤ Operating Cash Flow  (FCF = Op CF − Capex, Capex ≥ 0)
      2. "Less: purchases of property and equipment" ≈ "Purchases of property
         and equipment"  (same concept in different table contexts)
    """
    result = dict(metrics)

    # 1. FCF must be ≤ operating cash flow
    _, op_cf = _find_first(result, r"net cash provided by operating")
    fcf_key, fcf_val = _find_first(result, r"\bfree cash flow\b")
    if (
        fcf_key is not None
        and op_cf is not None and fcf_val is not None
        and isinstance(op_cf, (int, float)) and isinstance(fcf_val, (int, float))
        and fcf_val > op_cf
    ):
        logger.warning(
            "Consistency: FCF (%s) > Operating CF (%s) — discarding implausible FCF value",
            fcf_val, op_cf,
        )
        result[fcf_key] = None

    # 2. "Less: purchases of property and equipment" is the FCF reconciliation
    #    entry for capex — it must be within 50% of the direct capex line.
    #    If they diverge significantly, the "Less:" value came from a wrong section.
    _, capex_val = _find_first(result, r"^purchases of property and equipment")
    less_key, less_val = _find_first(result, r"less:?\s+purchases of property")
    if (
        less_key is not None
        and capex_val is not None and less_val is not None
        and isinstance(capex_val, (int, float)) and isinstance(less_val, (int, float))
        and abs(capex_val) > 0
        and not (0.5 <= abs(less_val) / abs(capex_val) <= 2.0)
    ):
        logger.warning(
            "Consistency: dropping %r (%s) — differs >50%% from direct capex (%s)",
            less_key, less_val, capex_val,
        )
        result[less_key] = None

    return {k: v for k, v in result.items() if v is not None}


def extract_financial_metrics_node(state: EarningsAgentState) -> EarningsAgentState:
    """Split raw text into chunks, run LLM extraction on each, merge results.

    Stage 1 (Perceive & Plan) + Stage 2 (Act/Execute) of the agentic loop.
    On retry passes, ``state["extraction_notes"]`` carries focused hints from
    the reflection node that are injected into every chunk prompt.

    The output metrics dict uses the company's own metric labels as keys,
    making it flexible across different companies and document formats.
    """
    raw_text = (state.get("raw_text") or "")[:EXTRACTION_MAX_CHARS]
    ticker = state["ticker"]

    if not raw_text:
        return {**state, "status": "failed", "error": "No raw text available for extraction"}

    # Increment attempt counter (Stage 1 — Perceive & Plan)
    attempt_num = state.get("extraction_attempts", 0) + 1
    logger.info("Extraction pass %d for %s", attempt_num, ticker)

    # Pre-scan the full document once for scale and reporting period.
    # Injecting confirmed values into every chunk prompt eliminates the most
    # common class of scaling errors (header only visible in first chunk).
    doc_scale, doc_shares_scale, doc_period = _prescan_document(raw_text)
    dollar_multiplier = _SCALE_MULTIPLIERS.get(doc_scale, 1) if doc_scale else 1
    # When the document explicitly uses a different scale for share counts
    # (e.g. Apple: dollars in millions, shares in thousands), use that.
    # Otherwise fall back to the dollar multiplier.
    shares_multiplier = (
        _SCALE_MULTIPLIERS.get(doc_shares_scale, dollar_multiplier)
        if doc_shares_scale
        else dollar_multiplier
    )
    scale_hint = (
        f"CONFIRMED SCALE: document header says \"(In {doc_scale})\" — "
        f"set __scale__ = \"{doc_scale}\" for this chunk.\n"
        if doc_scale else ""
    )
    period_hint = (
        f"CONFIRMED PERIOD: current reporting period is \"{doc_period}\" — "
        f"set __period__ = \"{doc_period}\" and extract values from this column only. "
        f"Do NOT extract guidance, forecasts, or next-quarter projections.\n"
        if doc_period else (
            "IMPORTANT: extract values from the MOST RECENT ACTUAL reported quarter only. "
            "Do NOT extract guidance, forecasts, or next-quarter projections.\n"
        )
    )

    # If the reflection node left guidance from a previous pass, inject it
    # into every chunk prompt so the LLM focuses on what was missed.
    extraction_notes = state.get("extraction_notes") or ""
    focus_hint = (
        f"\nAdditional focus from quality review (pass {attempt_num}):\n{extraction_notes}\n"
        if extraction_notes
        else ""
    )

    chunks = _chunk_text(raw_text)
    total = len(chunks)
    logger.info("Extracting metrics for %s across %d chunk(s)", ticker, total)

    # Build all chunk prompts up front
    prompts = [
        _PROMPT_TEMPLATE.format(
            company_name=state["company_name"],
            ticker=ticker,
            chunk_num=i,
            total_chunks=total,
            focus_hint=focus_hint,
            scale_hint=scale_hint,
            period_hint=period_hint,
            text=chunk,
        )
        for i, chunk in enumerate(chunks, start=1)
    ]

    # Parallel execution when OLLAMA_NUM_PARALLEL > 1.
    # Requires Ollama server started with OLLAMA_NUM_PARALLEL set to the same value.
    # Default (1) = sequential — identical behaviour to the previous version.
    num_parallel = min(total, int(os.getenv("OLLAMA_NUM_PARALLEL", "1")))
    ordered: list[dict[str, Any] | None] = [None] * total

    if num_parallel > 1:
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            future_to_idx = {
                executor.submit(_invoke_chunk_with_retry, prompt, i + 1, total, ticker, shares_multiplier): i
                for i, prompt in enumerate(prompts)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered[idx] = future.result()
                completed += 1
                logger.info(
                    "Parallel progress for %s: %d/%d chunk task(s) completed",
                    ticker,
                    completed,
                    total,
                )
    else:
        for i, prompt in enumerate(prompts):
            ordered[i] = _invoke_chunk_with_retry(prompt, i + 1, total, ticker, shares_multiplier)

    chunk_results: list[dict[str, Any]] = []
    for i, result in enumerate(ordered, start=1):
        if result is not None:
            chunk_results.append(result)
            logger.info("Chunk %d/%d extracted for %s: %d metric(s)", i, total, ticker, len(result))
        else:
            logger.warning("Chunk %d/%d could not be parsed for %s after retries", i, total, ticker)

    if not chunk_results:
        return {
            **state,
            "extraction_attempts": attempt_num,
            "status": "failed",
            "error": f"All {total} chunk(s) failed to extract metrics for {ticker} (pass {attempt_num})",
        }

    metrics = _merge_metrics(chunk_results)
    metrics = _validate_metrics(metrics)
    logger.info("Merged %d metric(s) for %s: %s", len(metrics), ticker, list(metrics.keys()))
    return {**state, "metrics": metrics, "extraction_attempts": attempt_num, "status": "extracted"}
