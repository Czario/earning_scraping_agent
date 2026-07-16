from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from earnings_agents.config import (
    EXTRACTION_MAX_CHARS,
    GEMINI_REQUEST_TIMEOUT,
    GROQ_REQUEST_TIMEOUT,
    LLM_PROVIDER,
    SOURCE_GROUNDING,
    default_chunk_size,
)
from earnings_agents.hooks import get_detail_callback, report_call, report_detail
from earnings_agents.llm_factory import build_llm
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.analysis.validators import validate_metrics
from earnings_agents.tools.llm_extractor import (
    _CHUNK_MAX_RETRIES,
    _OLLAMA_REQUEST_TIMEOUT,
    _OLLAMA_SEMAPHORE,
    invoke_chunk_with_retry as _invoke_chunk_with_retry_new,
)
from earnings_agents.extraction.chunker import (
    _BOUNDARY_SNAP,
    _CHUNK_OVERLAP,
    _CHUNK_SIZE,
    _LABEL_TO_SECTION,
    _PRESCAN_PERIOD_RX,
    _PRESCAN_SCALE,
    _PRESCAN_SHARES_IN_THOUSANDS_RX,
    _SECTION_CHUNK_LABELS,
    _SECTION_PRIORITY,
    _UNKNOWN_SECTION_PRIORITY,
    _build_period_hint,
    _build_section_chunks,
    _chunk_text,
    _prescan_document,
    _section_of_chunk,
)
from earnings_agents.extraction.merger import (
    _BALANCE_SHEET_KEY_RX,
    _CASH_FLOW_KEY_RX,
    _INCOME_STATEMENT_KEY_RX,
    _MAJOR_METRIC_RX,
    _MIN_DOLLAR_FRACTION,
    _PCT_OR_PER_SHARE_PATTERNS,
    _RESCALE_UPPER_MULTIPLE,
    _SCALE_MULTIPLIERS,
    _SHARE_COUNT_PATTERN,
    _SHARE_COUNT_RAW_MAX,
    _TABLE_RAW_MAX,
    _find_flagged_chunk_indices,
    _infer_retry_sections,
    _infer_section_for_metric_key,
    _merge_metrics,
    _parse_llm_response,
    _target_year_from_report_date,
)
from earnings_agents.extraction.concept_mapper import (
    _TAXONOMY_PREFIXES,
    _build_concept_prompt_list,
    _llm_map_concepts,
)


# ── Multi-column table detection ─────────────────────────────────────────────
# Bank earnings supplements routinely present income statements with 5-9
# side-by-side columns (current quarter, prior quarters, YTD totals, percentage
# changes).  The HTML-to-text conversion often mangles the column headers so
# the LLM cannot reliably identify which column holds the current-period
# values.  This detector scans the chunk text for repeating quarter patterns
# and injects a column-position hint into the prompt.

# Matches period-column headers like "2Q25", "Q1 2025", "1Q 2026", "Q2-26" etc.
# Group 1 = the full label ("2Q25", "Q1 2025").
_MULTI_COL_Q_RX = re.compile(
    # "2Q25", "1Q26" (digit-Q-2digits)
    r"\b[1-4]Q['′]?\d{2}\b"
    # "2Q 2025", "1Q 2026" (digit-Q-space-4digits)
    r"|\b[1-4]Q\s+\d{4}\b"
    # "Q1 2025", "Q2-26", "Q1'26" (Q-digit-year)
    r"|\bQ[1-4][\s\-–—'′]*(?:20)?\d{2}\b",
    re.I,
)

# Also match short-form year rows under quarter headers, e.g.
#   "2Q  3Q  4Q  1Q  2Q"
#   "25  25  25  26  26"
# These appear when the year row and the quarter-row are on separate lines.
_MULTI_COL_YEAR_ROW_RX = re.compile(
    r"^(?:\s*(?:\d{2,4}|''\d{2}|'\d{2})\s+){4,}"
)


def _detect_multi_column_hint(chunk_text: str) -> str:
    """Return a column-position hint when *chunk_text* contains multiple
    period columns (common in bank supplements).

    Returns an empty string when the chunk has 2 or fewer period columns
    (standard single-period table — no hint needed).
    """
    # Clean HTML entities and non-breaking spaces for matching
    clean = chunk_text.replace("\xa0", " ").replace("&#160;", " ")
    clean = re.sub(r"&#\d+;", " ", clean)
    clean = re.sub(r"&nbsp;", " ", clean, flags=re.I)

    # Find all distinct quarter labels
    q_labels = set(_MULTI_COL_Q_RX.findall(clean))
    # Normalise: "Q1 2025" → "Q125", "1Q25" → "Q125" (unified sortable format)
    _normalised: set[str] = set()
    for lbl in q_labels:
        _norm = re.sub(r"[\s\-–—]+|['′]", "", lbl).upper()
        # Unify "4Q25" → "Q425" so sort by (year, quarter) works on both variants
        _norm = re.sub(r"^(\d)(Q)(\d{2})$", r"\2\1\3", _norm)
        _normalised.add(_norm)

    # Also check for repeated year rows under quarter headers
    # (e.g. "25  25  25  26  26" on a separate line after the Q row)
    year_row_matches = _MULTI_COL_YEAR_ROW_RX.findall(clean)

    col_count = len(_normalised)

    # If 3+ distinct quarter labels OR year rows with 5+ columns
    if col_count >= 3 or len(year_row_matches) >= 1:
        # Sort labels to identify the most recent quarter.
        # Sorted labels look like Q125, Q225, ..., Q126, Q226, ...
        def _q_sort_key(lbl: str) -> tuple[int, int]:
            """Extract (year, quarter) from normalised label like Q125, Q226, or Q12025."""
            m = re.match(r"Q(\d)(\d+)", lbl)
            if m:
                q = int(m.group(1))
                y_str = m.group(2)
                # Handle both 2-digit ("25" → 2025) and 4-digit ("2025" → 2025) years
                if len(y_str) == 2:
                    y = 2000 + int(y_str)
                else:
                    y = int(y_str)
                return (y, q)
            return (0, 0)

        sorted_q = sorted(_normalised, key=_q_sort_key)
        latest = sorted_q[-1] if sorted_q else ""

        # Build a readable column map
        cols_desc = " → ".join(sorted_q)
        logging.getLogger(__name__).info(
            "Multi-column table hint for %d period columns (latest=%s)",
            col_count, latest,
        )
        return (
            "\n"
            f"⚠  MULTI-COLUMN TABLE DETECTED: this table shows {col_count} period columns "
            f"({cols_desc}).\n"
            f"The CURRENT period column is the LAST one: **{latest}**.\n"
            "Column 1 = row labels.  Each subsequent column = one period's values "
            f"in the order shown above.\n"
            f"Extract values ONLY from the {latest} column.\n"
            "Do NOT extract from any percentage-change column, YTD column, "
            "or prior-period column.\n"
        )

    return ""


logger = logging.getLogger(__name__)


# Chunking parameters now come from earnings_agents.extraction.chunker (imported above).
# LLM invocation constants now come from earnings_agents.tools.llm_extractor (imported above).

# Keys whose values are percentages, per-share amounts, or non-dollar counts --
# moved to earnings_agents.extraction.merger (imported above).

# Raw table values larger than this are assumed to already be full USD -- moved to merger.

# ── Targeted extraction (normalize_data mode) ────────────────────────────────

# Prompt used when target_concepts are loaded from normalize_data.
# The LLM must always use the XBRL bracket key as the JSON key — never the
# filing's own row label.  Tier-0 mapping then strips the brackets for a
# direct taxonomy_key → concept_id lookup, eliminating orphaned keys and the
# need for tier-2 semantic matching in the common case.
_TARGETED_PROMPT_TEMPLATE = """\
You are a financial data extraction assistant.

Extract ONLY the income statement metrics listed below from the text excerpt for {company_name} ({ticker}).
This is chunk {chunk_num} of {total_chunks} of the full document.
{focus_hint}{scale_hint}{period_hint}
SCOPE — extract ONLY the concepts listed below.
JSON KEY RULE — MANDATORY: ALWAYS use the bracketed key [ ] exactly as shown
as your JSON key. NEVER use the document's own row label as a key — even if
the document label looks different from the quoted label, the bracketed key
is the only accepted format. If no bracket is shown, use the quoted label.
For dimensional concepts (bracket contains "|"), the bracket encodes what
metric AND what segment/dimension is being measured —
e.g. [us-gaap:Revenues|aapl:AmericasSegmentMember] means: find the Americas
segment row under Revenue and return it with that exact bracketed string as
the JSON key.

{concept_list}
IGNORE — do NOT extract:
  • Balance sheet items, cash flow items, non-GAAP metrics, guidance / forecasts.
  • Any table from a GAAP-to-Non-GAAP reconciliation.
  • Percentage-only metrics (margins, growth rates) UNLESS they appear in the
    concept list above.
  • Values from FOOTNOTES or parenthetical sub-tables (e.g. "Stock-based
    compensation included in the above", supplementary breakdowns). A genuine
    income-statement line item is a primary row of the main statement, not a
    footnote disclosure. If R&D, S&M, or G&A appears both as a primary row
    AND inside a footnote, take ONLY the primary-row value (the larger one).
  EXCEPTION: Segment/geographic/category footnote sections that appear at the
    bottom of the income statement table ARE GAAP disclosures — extract them
    when they match a concept in your list. These sections are labeled like
    "(1) Net sales by reportable segment:", "(1) Net sales by category:",
    "(1) Revenue by geography:", etc. They are NOT the kind of footnotes to
    ignore; they contain primary GAAP segment data.

TABLE PRIORITY: prefer the primary condensed GAAP income statement. Also extract
from GAAP supplementary tables labeled "FINANCIAL DATA" in this document — these
include segment revenue, geographic revenue breakdown, and product-line detail
tables (the FINANCIAL DATA tables are the PRIMARY source for segment/geographic/
product-line concepts whose labels reference a region, segment name, or category).
Skip Non-GAAP tables entirely.

PERIOD RULE — CRITICAL when multiple period columns are present:
  Earnings statements show side-by-side columns (e.g. current quarter | year-ago
  quarter). Extract numeric values ONLY from the MOST RECENT period column —
  the column whose header carries the latest date / highest year. NEVER take a
  value from a prior-year or comparison column. If a metric appears under
  several columns, use ONLY the value beneath the most-recent column header.

INTERNAL CONSISTENCY — sanity-check before returning:
  The values you pick from a single column must reconcile:
    Revenue − Cost of revenue = Gross profit
  If your chosen Cost of revenue is larger than Revenue, or this subtraction
  does not match the printed Gross profit, you have taken a value from the
  WRONG column or the wrong row — re-read the statement and correct it.

Return ONLY a flat JSON object — no markdown fences, no extra text.

EFFICIENCY RULE — OMIT what is not present:
  Extract ONLY the concepts you can actually locate in the text below. If a
  concept from the list does NOT appear in the filing, OMIT its key entirely —
  do NOT include it with a null, 0, or placeholder value. Most earnings press
  releases contain only a subset of the concept list (segment/dimensional rows
  and rarely-reported line items are usually absent). A short JSON containing
  only the metrics that are truly present is the CORRECT answer — do not pad it.

Every value must be a plain number. NEVER write arithmetic expressions or
formulas as a value (e.g. do NOT write "15929 - 540 + 102" — omit the key if you
cannot determine the exact number from the table directly).

FIRST field must be "__scale__":
  "millions", "thousands", "billions", or "as-is" (no table).

SECOND field must be "__period__":
  The most recent period column label exactly as printed, or null.
  MUST include the full duration phrase, not just the end date.
  GOOD:  "Thirteen Weeks Ended May 2, 2026"
         "Three Months Ended March 31, 2026"
         "Six Months Ended June 30, 2026"
  BAD :  "May 2, 2026"   (date alone — quarter cannot be inferred)

ALL OTHER fields: use EXACTLY the label strings in quotes from the concept list above.

IMPORTANT — scale handling:
  The table caption (e.g. "(In millions)") declares the base unit. Set __scale__
  to that unit. Python multiplies every numeric value you return by that multiplier,
  so report the RAW table number exactly as printed — do NOT multiply yourself.
  Example: table says "(In millions)" and shows "82,886" → report 82886.

  INLINE SCALE NOTATION — some cells or column headers override the base unit:
  • Cell suffixes: B = billions, M = millions, T = trillions, K = thousands.
    Convert to the document base unit before reporting:
      table is "(In millions)", cell shows "1.5B"   → report 1500
      table is "(In billions)",  cell shows "500M"   → report 0.5
      table is "(In millions)", cell shows "2.4T"   → report 2400000
      table is "(In millions)", cell shows "150K"   → report 0.15
  • Column-header scale: if a column header declares its own unit
    (e.g. "Revenue ($B)" in a table captioned "(In millions)"), all values
    in that column are in the header's unit — convert to the document base
    unit before reporting.
  • Negative values: parentheses mean negative — "(1,234)" = −1234.
  • EPS, share prices, and percentages are always reported as-is (never
    scaled by __scale__), regardless of the table caption.
{sources_hint}
Text excerpt:
\"\"\"
{text}
\"\"\"
"""

# Optional trailing block requesting per-metric source snippets. Included only
# when SOURCE_GROUNDING is enabled (config), because it roughly doubles the LLM
# output size. When omitted, check_source_grounding degrades to a no-op.
_SOURCES_HINT_BLOCK = """
LAST field must be "__sources__" (verification / "show me"):
  A JSON object mapping EACH metric key you returned above to the EXACT
  verbatim text snippet from the excerpt where you read its value — the row
  label together with the value as printed (e.g.
  {{"Revenue": "Total revenue 82,886"}}).
  Copy the text character-for-character from the excerpt; do NOT paraphrase,
  reformat, or invent. If you cannot point to the exact source text for a
  value, return null for that value instead of guessing.
"""

# _TAXONOMY_PREFIXES, _LLM_MAP_PROMPT, _llm_map_concepts, _build_concept_prompt_list
# moved to earnings_agents.extraction.concept_mapper (imported above).

# _MIN_DOLLAR_FRACTION, _MAJOR_METRIC_RX, _RESCALE_UPPER_MULTIPLE
# moved to earnings_agents.extraction.merger (imported above).

# _PRESCAN_SCALE, _PRESCAN_SHARES_IN_THOUSANDS_RX, _PRESCAN_PERIOD_RX
# moved to earnings_agents.extraction.chunker (imported above).

# _OLLAMA_REQUEST_TIMEOUT, _CHUNK_MAX_RETRIES, _OLLAMA_SEMAPHORE
# imported from earnings_agents.tools.llm_extractor (above).


def _request_timeout_for(provider: str | None) -> float:
    """Return the per-request HTTP timeout for the given LLM provider.

    ``provider`` may be ``None``, in which case the configured
    ``LLM_PROVIDER`` is used. Each cloud provider has its own timeout budget;
    everything else (local Ollama) uses the Ollama default.
    """
    from earnings_agents.config import DEEPSEEK_REQUEST_TIMEOUT
    effective = (provider or LLM_PROVIDER).strip().lower()
    if effective == "groq":
        return GROQ_REQUEST_TIMEOUT
    if effective == "gemini":
        return GEMINI_REQUEST_TIMEOUT
    if effective == "deepseek":
        return DEEPSEEK_REQUEST_TIMEOUT
    return _OLLAMA_REQUEST_TIMEOUT


def _invoke_chunk_with_retry(
    prompt: str,
    chunk_num: int,
    total_chunks: int,
    ticker: str,
    shares_multiplier: int = 1,
    prescan_dollar_multiplier: int = 0,
    max_retries: int = _CHUNK_MAX_RETRIES,
    detail_callback=None,
    report_chunk=None,
    provider: str | None = None,
) -> "dict[str, Any] | None":
    """Thin adapter: calls tools.llm_extractor.invoke_chunk_with_retry.

    Kept here for backward compatibility with tests that import it directly
    from this module.  The parse function is bound to the current
    ``_parse_llm_response`` (from extraction.merger) and the multiplier args.
    The LLM is built here (using this module's ``build_llm``) so that test
    patches on ``earnings_agents.nodes.extract_financial_metrics.build_llm``
    remain effective.
    """
    def _parse(response: str) -> dict | None:
        return _parse_llm_response(response, shares_multiplier, prescan_dollar_multiplier)

    timeout = _request_timeout_for(provider)
    llm = build_llm(format_json=True, request_timeout=timeout, provider=provider)

    return _invoke_chunk_with_retry_new(
        prompt=prompt,
        chunk_num=chunk_num,
        total_chunks=total_chunks,
        ticker=ticker,
        parse_fn=_parse,
        llm=llm,
        max_retries=max_retries,
        detail_callback=detail_callback,
        report_chunk=report_chunk,
        provider=provider,
    )


# _BOUNDARY_SNAP, _chunk_text, _SECTION_CHUNK_LABELS, _LABEL_TO_SECTION,
# _SECTION_PRIORITY, _UNKNOWN_SECTION_PRIORITY, _section_of_chunk
# moved to earnings_agents.extraction.chunker (imported above).

# _INCOME_STATEMENT_KEY_RX, _BALANCE_SHEET_KEY_RX, _CASH_FLOW_KEY_RX,
# _infer_section_for_metric_key, _find_flagged_chunk_indices, _infer_retry_sections,
# _build_section_chunks, _prescan_document, _parse_llm_response, _merge_metrics,
# _target_year_from_report_date
# moved to earnings_agents.extraction.merger / chunker (imported above).


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

    # Targeted extraction requires concepts loaded from normalize_data. Generic
    # extraction has been removed, so a run that reaches this node without
    # target_concepts cannot proceed. In production this never happens because
    # load_company_concepts skips such runs; this is a defensive guard.
    if not (state.get("target_concepts") or []):
        return {
            **state,
            "status": "failed",
            "error": (
                f"No target concepts for {ticker}; cannot extract "
                f"(company missing from normalize_data)."
            ),
        }

    # Increment attempt counter (Stage 1 — Perceive & Plan)
    attempt_num = state.get("extraction_attempts", 0) + 1
    logger.info("Extraction pass %d for %s", attempt_num, ticker)

    # Provider selection: always use the configured LLM_PROVIDER for every
    # attempt. There is no automatic Ollama→Groq escalation — retries stay on
    # the same provider. (Escalation was removed because an unfixable identity
    # violation forced 3 passes onto Groq's free tier and triggered 429s.)
    escalated_provider: str | None = None
    from earnings_agents.config import LLM_PROVIDER as _LLM_PROVIDER
    _display_provider: str = _LLM_PROVIDER or "llm"

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

    # Period hint: prefer the authoritative SEC reportDate when present.
    # The SEC submissions API returns the exact filing period-end date which
    # is always the current quarter — unlike a prescan hit that can pick up
    # the prior-year comparison column header instead.
    sec_report_date_str: str | None = state.get("sec_report_date")  # type: ignore[assignment]
    # Annual (10-K) filings present both the single-quarter (e.g. "Three Months
    # Ended") and the full-year (e.g. "Twelve Months Ended") columns side by
    # side. For these, the full-year column is the one we want — so the
    # duration instruction must flip from "shortest" to "longest". For
    # quarterly (10-Q) filings the single-quarter column is correct.
    is_annual = state.get("detected_period_type") == "annual"
    period_hint = _build_period_hint(sec_report_date_str, doc_period, is_annual)

    # If the reflection node left guidance from a previous pass, inject it
    # into every chunk prompt so the LLM focuses on what was missed.
    extraction_notes = state.get("extraction_notes") or ""
    focus_hint_parts: list[str] = []
    if extraction_notes:
        # On retry passes, frame the notes explicitly as a correction so the
        # LLM treats them as high-priority errors to fix, not advisory hints.
        focus_hint_parts.append(
            f"""⚠  CORRECTION REQUIRED — PASS {attempt_num} RETRY

The previous extraction pass produced values that failed accounting identity
checks. The analysis below shows EXACTLY what was wrong. Read it carefully
before extracting values from this chunk — the error is almost always reading
from the WRONG column (prior-year instead of current-period).

{extraction_notes}

Double-check: after you choose values for Revenue, Cost of revenue and Gross
profit, verify Revenue − Cost of revenue = Gross profit before returning JSON.
"""
        )
    focus_hint = ("\n" + "\n".join(focus_hint_parts) + "\n") if focus_hint_parts else ""

    from earnings_agents.config import CHUNK_SIZE as _CFG_CHUNK_SIZE
    _effective_chunk_size = _CFG_CHUNK_SIZE if _CFG_CHUNK_SIZE > 0 else default_chunk_size(LLM_PROVIDER)
    chunks = _build_section_chunks(
        state.get("raw_sections"),
        state.get("target_concepts"),
        chunk_size=_effective_chunk_size,
    )
    chunk_source = "section"
    if chunks is None:
        chunks = _chunk_text(raw_text, chunk_size=_effective_chunk_size)
        chunk_source = "char"
    total = len(chunks)
    # Section provenance per chunk (parallel to `chunks`). Section chunks carry
    # a `=== LABEL ===` header identifying their GAAP statement; plain char
    # chunks (fallback) are "unknown". Threaded into `_merge_metrics` so
    # income-statement values are never averaged with the same key leaking from
    # a supplementary table.
    chunk_sections_all = (
        [_section_of_chunk(c) for c in chunks]
        if chunk_source == "section"
        else ["unknown"] * total
    )

    scoped_retry = False
    # Track absolute chunk indices (positions in the full _build_section_chunks
    # output) for each element of `chunks` post-scoping. Used to build
    # per-metric chunk provenance that is stored in state and consumed by the
    # next retry pass to target specific chunks rather than whole sections.
    chunk_abs_indices: list[int] = list(range(total))

    if attempt_num > 1 and chunk_source == "section":
        # ── Chunk-level scoped retry (most precise) ───────────────────────
        # Use provenance from the previous pass to target only the specific
        # chunks that produced flagged metrics.  For metrics that are *missing*
        # (not in provenance), fall back to section-level scoping so we still
        # look in the right statement tables.
        prev_sources: dict[str, list[int]] = state.get("chunk_metric_sources") or {}
        findings_for_retry: list[dict] = state.get("findings") or []
        flagged_chunk_ids = _find_flagged_chunk_indices(findings_for_retry, prev_sources)

        retry_sections = _infer_retry_sections(state)
        # Add section-level chunks for metrics that are missing (not in provenance).
        section_chunk_ids = {
            i for i, s in enumerate(chunk_sections_all)
            if s in retry_sections
        }

        selected_set = flagged_chunk_ids | section_chunk_ids
        selected = sorted(selected_set)

        if selected and len(selected) < total:
            chunks = [chunks[i] for i in selected]
            chunk_sections_all = [chunk_sections_all[i] for i in selected]
            chunk_abs_indices = selected
            total = len(chunks)
            scoped_retry = True
            if flagged_chunk_ids and flagged_chunk_ids != section_chunk_ids:
                logger.info(
                    "Retry pass %d for %s: chunk-level scope — %d specific chunk(s) "
                    "%s + %d section chunk(s) from %s = %d total",
                    attempt_num,
                    ticker,
                    len(flagged_chunk_ids),
                    sorted(flagged_chunk_ids),
                    len(section_chunk_ids - flagged_chunk_ids),
                    sorted(retry_sections),
                    total,
                )
            else:
                logger.info(
                    "Retry pass %d for %s scoped to section(s) %s: %d chunk(s)",
                    attempt_num,
                    ticker,
                    sorted(retry_sections),
                    total,
                )
    logger.info(
        "Extracting metrics for %s across %d %s chunk(s)",
        ticker, total, chunk_source,
    )
    detail_callback = get_detail_callback()

    # Shared per-chunk status tracker so the progress bar can show every
    # chunk's state (·=pending, ⟳=running, ✓=done, ✗=failed), not just the
    # most recent one. Updated under a lock from worker threads.
    _chunk_status: list[str] = ["·"] * total
    _chunk_attempts: list[int] = [0] * total
    _status_lock = threading.Lock()
    _glyph = {"pending": "·", "running": "⟳", "done": "✓", "failed": "✗"}

    def _render_chunks() -> str:
        parts = []
        for idx, st in enumerate(_chunk_status, start=1):
            attempt = _chunk_attempts[idx - 1]
            suffix = f"⋅{attempt}" if attempt > 1 and st == "⟳" else ""
            parts.append(f"{idx}{st}{suffix}")
        done = sum(1 for s in _chunk_status if s == "✓")
        return f"chunks {' '.join(parts)} ({done}/{total})"

    def _report_chunk(idx: int, status: str, attempt: int) -> None:
        with _status_lock:
            _chunk_status[idx] = _glyph[status]
            _chunk_attempts[idx] = attempt
            summary = _render_chunks()
        report_detail(summary)

    # Build all chunk prompts up front
    target_concepts: list[dict] = state.get("target_concepts") or []  # type: ignore[assignment]

    # ── Launch roles identification in parallel with extraction ──────────────
    # The roles call only needs concept labels (available now, before extraction
    # starts) — it is independent of extraction output.  Starting it here means
    # its latency overlaps with the extraction LLM call so the critical path
    # shrinks from sum(extraction + roles) to max(extraction, roles).
    #
    # Dimensional/segment concepts ("|" in taxonomy_key) are excluded: they are
    # never P&L derivation operands, so role identification is meaningless for
    # them and sending them inflates the prompt with noise.
    from concurrent.futures import Future as _Future, ThreadPoolExecutor as _RolesTPE
    from earnings_agents.analysis.calculators import identify_role as _identify_role_pre, ALL_ROLES
    from earnings_agents.tools.llm_concept_mapper import llm_identify_roles as _llm_identify_roles

    # Only identify roles for concepts the company actually reports recently —
    # a concept with no recent history is never extracted, so its role is moot.
    _recent_ids_for_roles = set(state.get("recent_concept_ids") or [])
    _roles_need_llm: list[dict] = [
        c for c in target_concepts
        if "|" not in (c.get("taxonomy_key") or "")
        and (not _recent_ids_for_roles or c.get("_id") in _recent_ids_for_roles)
        and _identify_role_pre(c.get("label", ""), c.get("taxonomy_key", "")) is None
    ]
    _roles_future: "_Future | None" = None
    _roles_executor: "_RolesTPE | None" = None
    if _roles_need_llm:
        _unrecognized_labels_pre = [c.get("label", "") for c in _roles_need_llm]
        report_call(
            f"  [roles]  {len(_roles_need_llm)} unrecognized label(s)"
            f"  ({', '.join(_unrecognized_labels_pre[:4])}"
            f"{'…' if len(_unrecognized_labels_pre) > 4 else ''})"
            f"  → calling llm in parallel  ({_display_provider})"
        )
        _role_timeout = _request_timeout_for(escalated_provider)
        _role_llm = build_llm(
            format_json=True,
            request_timeout=_role_timeout,
            provider=escalated_provider,
        )
        _roles_executor = _RolesTPE(max_workers=1)
        _roles_future = _roles_executor.submit(
            _llm_identify_roles, _unrecognized_labels_pre, _role_llm, ALL_ROLES
        )
    # ── End parallel roles launch ─────────────────────────────────────────────
    # Prompt pruning: drop concepts the company has NOT reported in its recent
    # periods (dimensional [Member] rows, retired line items).  These bloat the
    # prompt and are almost never in the current filing.  target_concepts stays
    # full for mapping + derivation; only the LLM prompt is trimmed.  When
    # recent_concept_ids is empty (bootstrap / disabled) the full list is used.
    recent_ids = set(state.get("recent_concept_ids") or [])
    if recent_ids:
        prompt_concepts = [c for c in target_concepts if c.get("_id") in recent_ids]
        # Safety: never send an empty prompt — fall back to full list if the
        # filter would remove everything (e.g. id-format mismatch).
        if not prompt_concepts:
            prompt_concepts = target_concepts
        else:
            _dropped = len(target_concepts) - len(prompt_concepts)
            if _dropped > 0:
                logger.info(
                    "Prompt pruning for %s: %d concept(s) kept, %d dropped "
                    "(no value in recent periods)",
                    ticker, len(prompt_concepts), _dropped,
                )
                report_call(
                    f"  [prompt]  {len(prompt_concepts)}/{len(target_concepts)} concepts"
                    f"  ({_dropped} unreported in recent periods, pruned)"
                )
    else:
        prompt_concepts = target_concepts
    concept_list_str = _build_concept_prompt_list(prompt_concepts)
    # Per-chunk scale detection: each section chunk carries its own scale caption
    # (injected by extract_html_text for each GAAP table, or present in the
    # original HTML).  Running _prescan_document on the individual chunk text
    # gives the most accurate multiplier for that specific table — immune to
    # other sections of the document using a different scale.
    #
    # Example: a press release whose primary GAAP statements are "(In thousands)"
    # but whose supplemental segment table is "(In millions)" will produce the
    # correct ×1 000 or ×1 000 000 multiplier for each chunk independently.
    # Falls back to the document-level scale when the chunk has no recognisable
    # scale caption (e.g. plain char-split chunks from PDF paths).
    #
    # shares_multiplier stays document-level: the "except shares in thousands"
    # clause is a document-wide declaration that lives in the primary statement
    # caption and need not repeat in every section.
    # ── Prior-period reference values ────────────────────────────────────
    # Inject known values from the most recent stored period as a reference
    # for the LLM.  This helps the LLM identify the correct column in
    # multi-column bank supplement tables — the current-period values should
    # be similar in magnitude to the prior period's.
    _prior_ref_str: str = ""
    if state.get("company_cik") and state.get("target_concepts"):
        try:
            from earnings_agents.tools.normalize_data_client import _get_client as _gc, _NORMALIZE_DB as _nd
            _db2 = _gc()[_nd]
            _cik2 = state["company_cik"]
            _period_type2 = state.get("detected_period_type", "quarterly")
            _col2 = "concept_values_quarterly" if _period_type2 == "quarterly" else "concept_values_annual"
            # Find the most recent stored period (excluding current, if already stored)
            _all_periods = sorted(
                _db2[_col2].distinct("reporting_period.end_date", {"company_cik": _cik2}),
                reverse=True,
            )
            _sec_date2 = state.get("sec_report_date")
            _skip_current = False
            if _sec_date2:
                from datetime import date as _dd
                _cfd2 = _dd.fromisoformat(_sec_date2) if isinstance(_sec_date2, str) else _sec_date2
                if _all_periods and _all_periods[0] == _cfd2:
                    _skip_current = True
            _prior_end = _all_periods[1 if _skip_current else 0] if _all_periods else None
            if _prior_end:
                # Get values for the prompt concepts
                _prompt_ids = [c["_id"] for c in prompt_concepts if c.get("_id")]
                _prior_vals = list(_db2[_col2].find({
                    "company_cik": _cik2,
                    "concept_id": {"$in": _prompt_ids},
                    "reporting_period.end_date": _prior_end,
                }))
                if _prior_vals:
                    # Build id→value map
                    _id_to_val: dict[str, float] = {}
                    for _pv in _prior_vals:
                        _cid = str(_pv["concept_id"])
                        _val = _pv.get("value")
                        if isinstance(_val, (int, float)):
                            if _cid not in _id_to_val or abs(_val) > abs(_id_to_val[_cid]):
                                _id_to_val[_cid] = float(_val)
                    # Build id→label map
                    _id_to_label: dict[str, str] = {
                        str(c["_id"]): c.get("label", "") for c in target_concepts if c.get("_id")
                    }
                    # Build reference lines for the most important concepts (revenue, NI, NII, etc.)
                    _ref_lines: list[str] = []
                    # Sort by absolute value descending — most impactful first
                    _sorted_prior = sorted(
                        _id_to_val.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:30]  # cap at 30 to avoid bloating the prompt
                    for _cid, _val in _sorted_prior:
                        _lbl = _id_to_label.get(_cid, "")
                        if _lbl:
                            _ref_lines.append(f"    {_lbl:50s} {_val:>20,.0f}")
                    if _ref_lines:
                        _prior_ref_str = (
                            "\n"
                            f"REFERENCE PRIOR PERIOD ({_prior_end.date()}): "
                            f"The values below are from the most recent stored period. "
                            f"Use them to identify the correct column — current-period values "
                            f"should be in the same magnitude and P&L order.\n"
                            + "\n".join(_ref_lines)
                            + "\n"
                        )
                        logger.info(
                            "Prior-period reference: %d values loaded for %s from %s",
                            len(_ref_lines), ticker, _prior_end.date(),
                        )
        except Exception as _exc:
            import traceback as _tb
            logger.warning(
                "Prior-period reference values unavailable for %s: %s\n%s",
                ticker, _exc, _tb.format_exc(),
            )

    # Append prior-period reference to focus_hint
    if _prior_ref_str:
        focus_hint += _prior_ref_str

    # Detect multi-column table structures and inject column-position hints.
    # Bank supplements routinely pack 5-9 columns side-by-side; the LLM often
    # picks values from the wrong column when the HTML-to-text conversion
    # mangles the column headers.
    chunk_multi_col_hints: list[str] = [_detect_multi_column_hint(c) for c in chunks]

    chunk_dollar_multipliers: list[int] = []
    chunk_scale_hints: list[str] = []
    for _chunk in chunks:
        _chunk_scale, _, _ = _prescan_document(_chunk)
        if _chunk_scale:
            _cm = _SCALE_MULTIPLIERS.get(_chunk_scale, 1)
            _ch = (
                f"CONFIRMED SCALE: this table is labelled \"(In {_chunk_scale})\" — "
                f"set __scale__ = \"{_chunk_scale}\" for this chunk.\n"
            )
        else:
            _cm = dollar_multiplier
            _ch = scale_hint
        chunk_dollar_multipliers.append(_cm)
        chunk_scale_hints.append(_ch)

    # ── Clean HTML entities from chunk text ──────────────────────────────
    # Bank supplement tables pack column data with HTML entities (\u200b
    # zero-width spaces, \xa0 non-breaking spaces, &#160;, &#8203; etc.) that
    # mangle the text layout, making it nearly impossible for the LLM to
    # identify which column a value belongs to.  Strip these before sending
    # to the LLM so it sees clean, parseable text.
    def _clean_chunk(text: str) -> str:
        """Remove HTML entities and invisible characters from chunk text."""
        # Replace non-breaking and zero-width spaces with regular space
        cleaned = text.replace("\xa0", " ")
        cleaned = cleaned.replace("\u200b", " ")
        cleaned = cleaned.replace("\u200e", " ")
        cleaned = cleaned.replace("\u200f", " ")
        cleaned = cleaned.replace("\u2028", " ")
        # Replace HTML numeric entities
        cleaned = re.sub(r"&#\d+;", " ", cleaned)
        cleaned = re.sub(r"&nbsp;", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"&amp;", "&", cleaned, flags=re.I)
        cleaned = re.sub(r"&lt;", "<", cleaned, flags=re.I)
        cleaned = re.sub(r"&gt;", ">", cleaned, flags=re.I)
        # Remove table-cell pipe artifacts that LLMs cannot parse correctly
        # (HTML tables converted to text leave orphaned | characters)
        cleaned = re.sub(r"\|\s*\|\s*\|", " | ", cleaned)  # triple pipes → single pipe
        cleaned = re.sub(r"\|\s*\|", " | ", cleaned)          # double pipes → single pipe
        # Collapse multi-space runs (but preserve newlines)
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        cleaned = re.sub(r"^\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s+$", "", cleaned, flags=re.MULTILINE)
        # Remove entirely-empty lines (lines with only spaces and pipes)
        cleaned = re.sub(r"^[\s\|]+$", "", cleaned, flags=re.MULTILINE)
        return cleaned

    cleaned_chunks = [_clean_chunk(c) for c in chunks]

    # Source-grounding block is opt-in (SOURCE_GROUNDING): it roughly doubles
    # the LLM output size, so it is omitted by default for speed.
    sources_hint = _SOURCES_HINT_BLOCK if SOURCE_GROUNDING else ""
    prompts = [
        _TARGETED_PROMPT_TEMPLATE.format(
            company_name=state["company_name"],
            ticker=ticker,
            chunk_num=i,
            total_chunks=total,
            focus_hint=focus_hint + chunk_multi_col_hints[i - 1],
            scale_hint=chunk_scale_hints[i - 1],
            period_hint=period_hint,
            concept_list=concept_list_str,
            sources_hint=sources_hint,
            text=chunk,
        )
        for i, chunk in enumerate(cleaned_chunks, start=1)
    ]

    # Log a preview of cleaned chunks
    for i, c in enumerate(cleaned_chunks, start=1):
        _interest_lines = [l for l in c.split('\n') if 'interest' in l.lower()][:5]
        logger.info(
            "Chunk %d/%d: %d chars, %d interest-related line(s)",
            i, len(cleaned_chunks), len(c), len(_interest_lines),
        )
        for l in _interest_lines:
            logger.info("  Chunk %d interest line: %s", i, l[:200])
        if not _interest_lines:
            # Show first 200 chars of chunk to understand what's in it
            _preview = c[:300].replace('\n', ' | ')
            logger.info("  Chunk %d preview (no interest found): %s", i, _preview[:200])

    # Parallel execution when OLLAMA_NUM_PARALLEL > 1.
    # Requires Ollama server started with OLLAMA_NUM_PARALLEL set to the same value.
    # Default is provider-aware: cloud APIs default to 3 (parallel chunks), local
    # Ollama defaults to 1 (serial) unless explicitly overridden via env var.
    from earnings_agents.config import default_num_parallel as _default_num_parallel
    _np_env = os.getenv("OLLAMA_NUM_PARALLEL")
    _np_default = int(_np_env) if _np_env else _default_num_parallel(LLM_PROVIDER)
    num_parallel = min(total, _np_default)
    ordered: list[dict[str, Any] | None] = [None] * total

    if num_parallel > 1:
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            future_to_idx = {
                executor.submit(
                    _invoke_chunk_with_retry,
                    prompt,
                    i + 1,
                    total,
                    ticker,
                    shares_multiplier,
                    chunk_dollar_multipliers[i],
                    _CHUNK_MAX_RETRIES,
                    detail_callback,
                    _report_chunk,
                    escalated_provider,
                ): i
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
            ordered[i] = _invoke_chunk_with_retry(
                prompt,
                i + 1,
                total,
                ticker,
                shares_multiplier,
                chunk_dollar_multipliers[i],
                _CHUNK_MAX_RETRIES,
                detail_callback,
                _report_chunk,
                escalated_provider,
            )

    chunk_results: list[dict[str, Any]] = []
    chunk_result_sections: list[str] = []
    # Parallel to chunk_results: absolute chunk index for each successful result.
    chunk_result_abs_indices: list[int] = []
    for i, result in enumerate(ordered, start=1):
        if result is not None:
            chunk_results.append(result)
            chunk_result_sections.append(chunk_sections_all[i - 1])
            chunk_result_abs_indices.append(chunk_abs_indices[i - 1])
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

    # Build per-metric provenance: metric key → list of absolute chunk indices
    # that reported a non-null value for it.  Stored in state so the next retry
    # pass can target specific chunks rather than retrying whole sections.
    new_metric_sources: dict[str, list[int]] = {}
    for result, abs_idx in zip(chunk_results, chunk_result_abs_indices):
        for key, value in result.items():
            if value is not None and not key.startswith("__"):
                new_metric_sources.setdefault(key, []).append(abs_idx)

    # On scoped retry: merge with existing provenance (new pass overwrites
    # provenance for retried keys; untouched keys keep their previous sources).
    if scoped_retry:
        prev_sources_for_merge: dict[str, list[int]] = state.get("chunk_metric_sources") or {}
        chunk_metric_sources_out = {**prev_sources_for_merge, **new_metric_sources}
    else:
        chunk_metric_sources_out = new_metric_sources

    metrics = _merge_metrics(
        chunk_results,
        source_text=raw_text,
        target_year=_target_year_from_report_date(sec_report_date_str),
        sections=chunk_result_sections,
    )

    # Pull out the per-metric source snippets (the "show me" verification
    # evidence) so they never pollute the metrics dict (concept mapping,
    # validation, and persistence all operate on real metric keys only).
    merged_source_snippets = metrics.pop("__sources__", None)
    logger.info(
        "Source grounding for %s: captured %d snippet(s) from LLM __sources__",
        ticker,
        len(merged_source_snippets) if isinstance(merged_source_snippets, dict) else 0,
    )

    # On retry passes, keep untouched metrics from the previous pass and
    # overwrite only keys returned by the retried chunk(s).  This applies to
    # both scoped retries (only a subset of chunks re-run) and full retries
    # (all chunks re-run but some keys the LLM returns as null were already
    # correct from pass 1).
    if scoped_retry or attempt_num > 1:
        prev_metrics = state.get("metrics")
        if isinstance(prev_metrics, dict):
            metrics = {**prev_metrics, **metrics}

    # Combine source snippets across passes so verification covers metrics that
    # were carried over untouched from an earlier pass (parallel to the metrics
    # carry-over above).
    if scoped_retry or attempt_num > 1:
        prev_snippets = state.get("metric_source_snippets") or {}
        metric_source_snippets_out: dict | None = {
            **prev_snippets,
            **(merged_source_snippets or {}),
        } or None
    else:
        metric_source_snippets_out = merged_source_snippets or None

    metrics, identity_warnings = validate_metrics(metrics)
    logger.info("Merged %d metric(s) for %s: %s", len(metrics), ticker, list(metrics.keys()))

    # When running in targeted mode (normalize_data), build concept_metrics:
    # a dict mapping concept_id → value for direct upsert into
    # normalize_data.concept_values_quarterly.
    #
    # Mapping strategy (two-tier):
    #   Tier 1 — Deterministic: exact label match, then normalised (lowercase +
    #             collapsed whitespace) match.  Zero latency, zero hallucination risk.
    #   Tier 2 — Semantic LLM: for concepts still unmapped after Tier 1, call the
    #             LLM with the extracted keys + remaining concept list and ask it to
    #             match semantically.  The LLM can only pick from the supplied keys
    #             (no hallucination) and must return null for no-match.
    concept_metrics: dict[str, float] | None = None
    if target_concepts:
        # ── Tier 1: deterministic label matching ────────────────────────────
        def _norm_label(s: str) -> str:
            import re as _re
            return _re.sub(r"\s+", " ", s).strip().lower()

        exact_label_to_id: dict[str, str] = {c["label"]: c["_id"] for c in target_concepts}
        norm_label_to_id: dict[str, str] = {
            _norm_label(c["label"]): c["_id"] for c in target_concepts
        }
        # Tier-0 lookup: taxonomy_key → concept_id.
        # Handles two formats the LLM may return for the same XBRL concept:
        #   bare key:    us-gaap:NoninterestIncome     (old prompt behaviour)
        #   bracket key: [us-gaap:NoninterestIncome]   (new mandatory format)
        # Both resolve to the same concept_id so either format works.
        # Falls back to the `concept` field when `taxonomy_key` is None
        # (common for bank/financial companies where concept documents store the
        # XBRL key in `concept` rather than `taxonomy_key`).
        taxonomy_key_to_id: dict[str, str] = {}
        bracket_key_to_id: dict[str, str] = {}
        for c in target_concepts:
            key = c.get("taxonomy_key") or c.get("concept") or ""
            if key:
                taxonomy_key_to_id[key] = c["_id"]
                bracket_key_to_id[f"[{key}]"] = c["_id"]
        concept_metrics = {}
        concept_id_to_metric_key: dict[str, str] = {}  # reverse map for mapped_metric_keys
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if key in taxonomy_key_to_id:
                # Tier 0a: bare taxonomy key (legacy / fallback)
                cid = taxonomy_key_to_id[key]
                concept_metrics[cid] = float(value)
                concept_id_to_metric_key[cid] = key
            elif key in bracket_key_to_id:
                # Tier 0b: bracket-wrapped key [us-gaap:X] — new primary format
                cid = bracket_key_to_id[key]
                concept_metrics[cid] = float(value)
                concept_id_to_metric_key[cid] = key
            elif key in exact_label_to_id:
                cid = exact_label_to_id[key]
                concept_metrics[cid] = float(value)
                concept_id_to_metric_key[cid] = key
            elif _norm_label(key) in norm_label_to_id:
                cid = norm_label_to_id[_norm_label(key)]
                concept_metrics[cid] = float(value)
                concept_id_to_metric_key[cid] = key

        # ── Tier 2: LLM semantic mapping for residual unmapped concepts ─────
        #
        # Tier 2's only job: the extraction LLM sometimes echoes the filing's
        # own row label instead of the concept label we gave it.  Those keys
        # are extracted but not tier1-matched ("orphaned").  Tier 2 asks the
        # LLM to match those orphaned keys to the remaining unmapped concepts.
        #
        # If there are NO orphaned keys (every extracted value is already used
        # by a concept), tier 2 can never find a new match → skip it entirely.
        # This is the key insight: the extraction LLM already saw every concept
        # label; anything it returned null for genuinely isn't in the filing.
        mapped_ids = set(concept_metrics.keys())
        unmapped_concepts = [c for c in target_concepts if c["_id"] not in mapped_ids]
        all_numeric_keys = [k for k, v in metrics.items() if isinstance(v, (int, float))]
        # Keys already consumed by tier0/tier1 — a concept's value; re-assigning
        # them to a different concept would be wrong.
        used_keys: set[str] = set(concept_id_to_metric_key.values())
        # Only orphaned keys (extracted but not yet mapped to any concept) can
        # possibly match an unmapped concept in tier 2.
        orphaned_keys = [k for k in all_numeric_keys if k not in used_keys]

        # Pre-filter: drop concepts that can never appear in earnings press
        # releases so the LLM prompt stays small and the call can be skipped
        # entirely when nothing matchable remains.
        #
        # (a) OCI / comprehensive-income items — full-statement-only
        #     disclosures, universally absent from press releases.
        _OCI_SKIP_RX = re.compile(
            r"other\s+comprehensive\s+(?:income|loss)"
            r"|comprehensive\s+(?:income|loss)"
            r"|unrealized\s+(?:gain|loss)"
            r"|foreign\s+currency\s+translation"
            r"|accumulated\s+other\s+comprehensive",
            re.IGNORECASE,
        )
        # (b) Dimensional segment concepts (taxonomy_key contains '|') —
        #     require segment-level extraction; a flat extracted key can
        #     never map to them.
        tier2_candidates = [
            c for c in unmapped_concepts
            if not _OCI_SKIP_RX.search(c.get("label", ""))
            and "|" not in (c.get("taxonomy_key") or "")
        ]

        if tier2_candidates:
            # Use ALL extracted keys for mapping (not just orphans). The
            # extraction LLM may have returned keys that don't match the
            # concept label exactly (e.g. "Deposits" instead of "Interest
            # Expense, Deposits"), but the LLM understands which concept
            # each key represents.  Previously this only ran on orphaned
            # keys, which missed cases where a key was deterministically
            # mapped to the wrong concept.
            _map_keys = orphaned_keys or all_numeric_keys
            logger.info(
                "Targeted mode: %d concept(s) unmatched after label lookup for %s "
                "— running LLM semantic mapping (%d matchable, %d candidate keys)",
                len(unmapped_concepts), ticker, len(tier2_candidates), len(_map_keys),
            )
            report_call(
                f"  [tier2]  {len(unmapped_concepts)} unmapped concept(s)"
                f"  ({len(tier2_candidates)} matchable,"
                f" {len(_map_keys)} candidate keys)"
                f"  → calling llm  ({_display_provider})"
            )
            map_timeout = _request_timeout_for(escalated_provider)
            map_llm = build_llm(format_json=True, request_timeout=map_timeout, provider=escalated_provider)
            llm_matches = _llm_map_concepts(
                _map_keys,             # ALL numeric keys (not just orphans)
                tier2_candidates,
                map_llm,
                # Don't pass already_used_keys — allow the LLM to remap
                # keys that Tier 0/1 may have mapped to the wrong concept
                # (e.g. "Deposits" → balance sheet "Deposits" when it
                # should be "Interest Expense, Deposits").
            )
            for concept_id, matched_key in llm_matches.items():
                value = metrics.get(matched_key)
                if isinstance(value, (int, float)):
                    concept_metrics[concept_id] = float(value)
                    concept_id_to_metric_key[concept_id] = matched_key
                    logger.debug(
                        "LLM semantic mapping: %r → concept_id %s for %s",
                        matched_key, concept_id, ticker,
                    )
            report_call(
                f"  [tier2]  ✓ mapped {len(llm_matches)}/{len(tier2_candidates)} concept(s)"
            )
        else:
            logger.info(
                "Tier2 skipped for %s: all %d unmapped concept(s) are "
                "OCI/dimensional — not extractable from press releases",
                ticker, len(unmapped_concepts),
            )
            report_call(
                f"  [tier2]  skipped ({len(unmapped_concepts)} unmapped, "
                f"all OCI/dimensional — not extractable from press releases)"
            )

        # ── Tier 3: Derive any unmapped concept whose value can be computed ─────
        # Run derivation for ALL concepts that still lack a value — including
        # system/calculated ones now folded into target_concepts.  The engine
        # is a fast no-op when nothing is unmapped, so we always invoke it.
        derived_concept_ids: set[str] = set()
        all_for_derivation = target_concepts or []
        if all_for_derivation:
            from earnings_agents.analysis.calculators import (
                derive_missing_concept_metrics,
                identify_role,
                ALL_ROLES,
            )

            # LLM role identification for concepts whose labels are not matched
            # by the regex patterns.  The LLM call was launched in parallel with
            # extraction above — collect the result here (typically already done).
            role_overrides: dict[str, str] = {}
            if _roles_future is not None:
                try:
                    _label_to_role = _roles_future.result()
                finally:
                    if _roles_executor is not None:
                        _roles_executor.shutdown(wait=False)
                for c in _roles_need_llm:
                    role = _label_to_role.get(c.get("label", ""))
                    if role:
                        role_overrides[c["_id"]] = role
                        logger.info(
                            "LLM role identification: '%s' → '%s' (concept_id %s) for %s",
                            c.get("label", ""), role, c["_id"], ticker,
                        )
                if role_overrides:
                    mapped_roles = [
                        f"'{c.get('label', '')}' → {role_overrides[c['_id']]}"
                        for c in _roles_need_llm if c["_id"] in role_overrides
                    ]
                    report_call(f"  [roles]  ✓ {', '.join(mapped_roles)}")
                else:
                    report_call(f"  [roles]  no roles identified")

            before_derivation_ids = set(concept_metrics.keys())
            concept_metrics = derive_missing_concept_metrics(
                concept_metrics, all_for_derivation, role_overrides=role_overrides,
            )
            derived_concept_ids = set(concept_metrics.keys()) - before_derivation_ids
            id_to_label_derive = {c["_id"]: c.get("label", c["_id"]) for c in all_for_derivation}
            if derived_concept_ids:
                derived_labels = [id_to_label_derive.get(cid, cid) for cid in derived_concept_ids]
                report_call(
                    f"  [derive]  ✓ {len(derived_concept_ids)} value(s) computed:"
                    f"  {', '.join(derived_labels)}"
                )
            else:
                still_unmapped = [c for c in all_for_derivation if c["_id"] not in concept_metrics]
                report_call(
                    f"  [derive]  0 values computed"
                    f"  ({len(still_unmapped)} concept(s) still unmapped)"
                )

        # ── Generalized deterministic fallback for missing concepts ────────
        # The LLM often misses line items in complex multi-column supplement
        # tables (bank interest breakdowns, segment details, etc.).  This
        # fallback scans the classified section text for ANY missing concept
        # label and extracts its value deterministically.
        #
        # Strategy: for each missing IS concept (non-dimensional, non-OCI),
        # search the section text for the concept label followed by a number
        # pattern.  Handle both single-period (1 number) and multi-period
        # (5 numbers = 5 quarterly columns) formats.
        if concept_metrics is not None and target_concepts:
            # Build section text once (tables, not narrative)
            _raw_sections = state.get("raw_sections") or {}
            _section_texts: list[str] = []
            for _key in ("income_statement", "other"):
                for _entry in _raw_sections.get(_key, []):
                    if isinstance(_entry, str):
                        _section_texts.append(_entry)
            # Combine section text (IS + other tables) with the full raw_text
            # (which now includes supplement "other" table data via supp_blocks)
            # for the widest possible search scope
            _raw = "\n".join(_section_texts) if _section_texts else ""
            _raw_text = state.get("raw_text") or ""
            if _raw_text:
                _raw = _raw + "\n" + _raw_text if _raw else _raw_text

            # Clean text once (keep newlines for line-start matching)
            _clean = _raw.replace("\xa0", " ").replace("\u200b", " ")
            _clean = re.sub(r"&#\d+;", " ", _clean)
            _clean = re.sub(r"<[^>]+>", " ", _clean)
            _clean = re.sub(r"[^\S\n]+", " ", _clean)
            _clean_lower = _clean.lower()

            # Find ALL missing non-dimensional concepts
            _missing_any = [
                c for c in target_concepts
                if c["_id"] not in concept_metrics
                and "[member]" not in c.get("label", "").lower()
                and "|" not in (c.get("taxonomy_key") or "")
                and "comprehensive income" not in c.get("label", "").lower()
                and not c.get("abstract", False)
                and len(c.get("label", "")) > 3  # skip short codes
            ]
            if not _missing_any:
                logger.debug("Deterministic fallback: no missing concepts to scan for %s", ticker)
            else:
                _found_count = 0
                for _concept in _missing_any:
                    _label = _concept.get("label", "")
                    _label_lower = _label.lower()

                    # ── Context-aware label matching ──────────────────────────
                    # Many concept labels combine parent context + child item
                    # (e.g. "Interest Expense, Deposits" = parent "Interest expense"
                    # + child "Deposits").  The filing shows these as hierarchical
                    # rows, not a single concatenated label.  Try splitting by
                    # comma and searching for the child under the parent section.
                    _search_text = _clean_lower
                    _search_label = _label_lower
                    _context_prefix = ""
                    _target_text = _clean

                    if "," in _label:
                        _parts = _label_lower.split(",", 1)
                        _parent = _parts[0].strip()
                        _child = _parts[1].strip()
                        if (
                            _parent in _clean_lower
                            and _label_lower not in _clean_lower
                        ):
                            # Try exact child match first, then fall back to
                            # matching the child's KEY WORDS (e.g. "Trading
                            # Liabilities" → match "Trading account liabilities"
                            # in the filing text).
                            _child_words = set(w for w in _child.split() if len(w) > 3)
                            _parent_pos = 0
                            while True:
                                _next_pos = _clean_lower.find(_parent, _parent_pos)
                                if _next_pos < 0:
                                    break
                                _after_parent = _clean[_next_pos + len(_parent):_next_pos + 600]
                                _after_clean = re.sub(r"[^\S\n]+", " ", _after_parent)
                                _after_lower = _after_clean.lower()
                                # Exact child match?
                                _match_found = _child in _after_lower
                                # If not, try partial word match
                                if not _match_found and _child_words:
                                    _after_words = set(_after_lower.split())
                                    _matched_words = _child_words & _after_words
                                    _match_found = len(_matched_words) >= len(_child_words) * 0.5
                                if _match_found:
                                    _context_prefix = _parent + " "
                                    _search_text = _after_lower
                                    _target_text = _after_clean
                                    # For partial matches, find the actual
                                    # text line that matched (filing label may
                                    # differ from normalize_data label)
                                    if _child not in _after_lower:
                                        _lines_after = _after_clean.split('\n')
                                        _best_line = ""
                                        _best_score = 0
                                        for _ln in _lines_after:
                                            _ln_words = set(_ln.lower().split())
                                            _score = len(_child_words & _ln_words)
                                            if _score > _best_score:
                                                _best_score = _score
                                                _best_line = _ln
                                        if _best_line:
                                            _search_label = _best_line.strip().lower()
                                            _target_text = _after_clean
                                        else:
                                            _search_label = _child
                                    else:
                                        _search_label = _child
                                    break
                                _parent_pos = _next_pos + 1

                    if _search_label not in _search_text:
                        continue

                    # Build regex: child label followed by numbers
                    _esc = re.escape(_search_label)
                    # Pattern: multi-column table.  Match ALL numbers after the
                    # label and take the LAST one (most recent period).  Works for
                    # both 5-column (bank) and 7-column (BAC vintages) formats.
                    _pat_multi = re.compile(
                        r"(?:^|[\n])\s*" + _esc
                        + r"[^\d]*?((?:\$?[\s\|]*[\d,]+[\s\|]*)+)",
                        re.I,
                    )
                    _match_multi = _pat_multi.search(_target_text)
                    if _match_multi:
                        # Extract ALL numbers from the match, take the last one
                        _all_nums = re.findall(r"[\d,]+", _match_multi.group(1))
                        if _all_nums:
                            _val_str = _all_nums[-1].replace(",", "")
                            _cols = f"{len(_all_nums)}"
                        else:
                            _val_str = None
                    else:
                        _val_str = None

                    if not _val_str:
                        # Fall back to 1-column (single number after label)
                        _pat1 = re.compile(
                            _esc + r"[^\d]*?(?:\$?[\s\|]*([\d,]+(?:\.\d+)?))",
                            re.I,
                        )
                        _match1 = _pat1.search(_target_text)
                        _val_str = _match1.group(1).replace(",", "") if _match1 else None
                        _cols = "1" if _match1 else None

                    if _val_str:
                        try:
                            _val = float(_val_str)
                            if 1e3 <= _val <= 1e14:
                                if 1900 <= _val <= 2100 and _cols == "1":
                                    continue
                                # ── Balance-sheet guard ──────────────────────
                                # The pipeline extracts INCOME STATEMENT concepts
                                # only.  The deterministic fallback must never
                                # pick up a value from a balance sheet row that
                                # happens to share a keyword with an IS concept
                                # label (e.g. "Deposits" matching both "Interest
                                # Expense, Deposits" (IS) and "Total Deposits"
                                # (BS)).  Check the matched line's surrounding
                                # context for clear balance-sheet signals.
                                if _context_prefix:
                                    # When context-aware matching was used
                                    # (comma-split label like "Interest Expense,
                                    # Deposits"), the search was scoped to text
                                    # AFTER the parent concept — the context
                                    # already guards against BS rows.
                                    pass
                                else:
                                    # For simple label matches, verify the
                                    # matched text isn't in a balance-sheet
                                    # section.  Find the matched region and check
                                    # nearby text for BS signals.
                                    _match_pos = _search_text.find(_search_label)
                                    if _match_pos >= 0:
                                        # Look at 200 chars BEFORE the match for
                                        # BS section headers.
                                        _before = _target_text[max(0, _match_pos - 200):_match_pos]
                                        _after = _target_text[_match_pos:_match_pos + 400]
                                        _surrounding = _before + " " + _after
                                        if (
                                            re.search(r"=== GAAP BALANCE SHEET ===", _before)
                                            or re.search(r"===.*BALANCE SHEET.*===", _before)
                                            or re.search(r"consolidated\s+balance\s+sheets?", _before, re.I)
                                            or re.search(r"statement[s]?\s+of\s+financial\s+(?:position|condition)", _before, re.I)
                                        ):
                                            logger.debug(
                                                "Deterministic fallback: SKIPPING '%s' = %s — "
                                                "matched in balance sheet section",
                                                _label, f"{_val:,.0f}",
                                            )
                                            continue
                                        # Also skip values whose surrounding text
                                        # looks like a balance sheet total (not
                                        # an IS breakdown item).
                                        _bs_total_rx = re.compile(
                                            r"\btotal\s+(?:assets?|liabilit|deposits?|loans?|equity|shareholders|stockholders)\b",
                                            re.I,
                                        )
                                        if _bs_total_rx.search(_surrounding):
                                            # Only skip if the label itself does
                                            # NOT contain IS-specific keywords.
                                            _is_keywords = re.search(
                                                r"interest\s+(?:income|expense)|noninterest|fee\s+(?:revenue|income)"
                                                r"|revenue|expense|income|charge",
                                                _label, re.I,
                                            )
                                            if not _is_keywords:
                                                logger.debug(
                                                    "Deterministic fallback: SKIPPING '%s' = %s — "
                                                    "surrounded by balance-sheet totals",
                                                    _label, f"{_val:,.0f}",
                                                )
                                                continue
                                concept_metrics[_concept["_id"]] = _val
                                _found_count += 1
                                logger.info(
                                    "Deterministic fallback: '%s' = %s (%s-col%s)",
                                    _label, f"{_val:,.0f}", _cols,
                                    ", context" if _context_prefix else "",
                                )
                        except (ValueError, TypeError):
                            pass

                if _found_count:
                    logger.info(
                        "Deterministic fallback: extracted %d/%d missing concepts for %s",
                        _found_count, len(_missing_any), ticker,
                    )
                elif len(_missing_any) > 10:
                    logger.info(
                        "Deterministic fallback: 0/%d found for %s (labels not in text)",
                        len(_missing_any), ticker,
                    )

        # ── Final summary table ──────────────────────────────────────────────
        id_to_label_summary: dict[str, str] = {
            c["_id"]: c.get("label", c["_id"]) for c in target_concepts
        }
        total_concepts = len(target_concepts)
        absent_labels = sorted(
            c["label"] for c in target_concepts
            if c["_id"] not in concept_metrics
        )
        # Separate segment/dimensional concepts from top-level IS concepts in
        # the absent list so retry hints can give the LLM targeted guidance on
        # WHERE to look (FINANCIAL DATA table vs primary income statement).
        absent_segment_labels = [
            c["label"] for c in target_concepts
            if c["_id"] not in concept_metrics
            and "|" in (c.get("taxonomy_key") or "")
        ]
        absent_toplevel_labels = [
            c["label"] for c in target_concepts
            if c["_id"] not in concept_metrics
            and "|" not in (c.get("taxonomy_key") or "")
        ]

        # Build lines: one per mapped concept, sorted by label, derived ones tagged.
        summary_lines: list[str] = []
        for cid, value in sorted(concept_metrics.items(), key=lambda kv: id_to_label_summary.get(kv[0], kv[0]).lower()):
            label = id_to_label_summary.get(cid, cid)
            tag = " [DERIVED]" if cid in derived_concept_ids else ""
            summary_lines.append(f"  {label}: {value}{tag}")

        logger.info(
            "Targeted mode %s: %d/%d concept(s) filled "
            "(%d from filing, %d derived)\n%s%s",
            ticker,
            len(concept_metrics),
            total_concepts,
            len(concept_metrics) - len(derived_concept_ids),
            len(derived_concept_ids),
            "\n".join(summary_lines),
            (
                f"\n  --- not in filing ({len(absent_labels)}): "
                + ", ".join(absent_labels)
                if absent_labels else ""
            ),
        )

    new_state: EarningsAgentState = {
        **state,
        "metrics": metrics,
        "extraction_attempts": attempt_num,
        "status": "extracted",
        # Always overwrite identity_warnings so a successful re-extract clears
        # warnings that were produced by a prior pass.  Only setting this when
        # non-empty would leave stale warnings in state and could incorrectly
        # block a save that should succeed.
        "identity_warnings": identity_warnings,
        "chunk_metric_sources": chunk_metric_sources_out,
    }
    if metric_source_snippets_out is not None:
        new_state["metric_source_snippets"] = metric_source_snippets_out
    if concept_metrics is not None:
        new_state["concept_metrics"] = concept_metrics
        new_state["mapped_metric_keys"] = list(concept_id_to_metric_key.values())
        new_state["derived_concept_ids"] = list(derived_concept_ids)
        if absent_labels:
            new_state["missing_concept_labels"] = absent_labels
        if absent_segment_labels:
            new_state["missing_segment_labels"] = absent_segment_labels
        if absent_toplevel_labels:
            new_state["missing_toplevel_labels"] = absent_toplevel_labels
    return new_state
