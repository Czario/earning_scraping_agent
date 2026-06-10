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
    GROQ_REQUEST_TIMEOUT,
)
from earnings_agents.hooks import get_detail_callback, report_detail
from earnings_agents.llm_factory import build_llm
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.analysis.validators import validate_metrics
from earnings_agents.tools.company_hints_loader import load_company_hints as _load_company_hints
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

logger = logging.getLogger(__name__)

# Human-curated hint files are now loaded via tools/company_hints_loader.
# _HINTS_DIR and _load_company_hints kept as aliases for backward compatibility.


# Chunking parameters now come from earnings_agents.extraction.chunker (imported above).
# LLM invocation constants now come from earnings_agents.tools.llm_extractor (imported above).

_PROMPT_TEMPLATE = """\
You are a financial data extraction assistant.

Extract ONLY income statement metrics from the text excerpt below for {company_name} ({ticker}).
This is chunk {chunk_num} of {total_chunks} of the full document.
{focus_hint}{scale_hint}{period_hint}
SCOPE — extract ONLY these income statement concepts.
Always use the EXACT single row label printed in the document as the JSON key.
The canonical names below are for concept recognition only — never use them as keys
and never combine multiple names into one key.

  Income statement concepts to extract:
  • Revenue
  • Cost of revenue
  • Gross profit
  • Research and development expense
  • Sales and marketing expense
  • General and administrative expense
  • Total operating expenses
  • Operating income
  • Interest income
  • Interest expense
  • Other income (expense), net
  • Income before income taxes
  • Income tax expense
  • Net income
  • Basic earnings per share
  • Diluted earnings per share
  • Weighted-average basic shares outstanding
  • Weighted-average diluted shares outstanding

  Common alternative labels (for recognition only — use whichever the document uses):
    Revenue → "Net revenue", "Net sales"
    Cost of revenue → "Cost of goods sold", "Cost of sales"
    Operating income → "Operating profit", "Operating loss", "Income from operations"
    Income tax expense → "Provision for income taxes"
    Net income → "Net earnings", "Net loss"
    Basic EPS → "Basic net income per share", "Net income per share — Basic"
    Diluted EPS → "Diluted net income per share", "Net income per share — Diluted"

IGNORE — do NOT extract any of the following:
  • Percentage-form metrics: gross margin %, operating margin %, net margin %.
    These are derivable — capture Gross profit (dollar amount) instead.
  • Balance sheet items (Assets, Liabilities, Equity, Inventory, Receivables, etc.)
  • Cash flow items (Operating / Investing / Financing cash flows, Capex, Depreciation, etc.)
  • Guidance, forecasts, or forward-looking projections.
  • Any table whose header or column labels contain "Non-GAAP", "non-GAAP", "Adjusted",
    "Reconciliation", "Supplemental", or lists both a GAAP and a Non-GAAP column side-by-side.
    Skip the ENTIRE table — do not extract any rows from it, even the GAAP column.
    If you are unsure whether a table is a reconciliation, skip it.
  • Any key prefixed with "GAAP " or "Non-GAAP " or containing "impact of", "adjustment".
  • Duplicate summary rows that restate a metric already captured (e.g. "Total operating
    expenses" when "Operating expenses" is already present with the same value).

TABLE PRIORITY — when the same metric appears in multiple tables, prefer this order:
  1. Primary condensed GAAP income statement (usually the first financial table).
  2. Segment or product-line breakdowns — skip unless the primary table is absent.
  3. GAAP-to-Non-GAAP reconciliation — always skip (covered by IGNORE above).
  Use the plain metric label from table (1); never prefix keys with "GAAP " or "Non-GAAP ".

Return ONLY a flat JSON object — no markdown fences, no extra text, no nested objects.
Every value must be a plain number or null. NEVER produce arithmetic expressions or
formulas (e.g. do NOT write "15929 - 540 + 102" — use null if you cannot determine
the exact number). NEVER produce {{"Key": {{"SubKey": value}}}} — that is
a nested object and is illegal. If a metric has sub-categories, create separate flat keys.

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
     key containing: "gross margin", "operating margin", "net margin",
     "gross margin %", "profit margin", "margin %", "growth rate".
Text excerpt:
\"\"\"
{text}
\"\"\"
"""

# Keys whose values are percentages, per-share amounts, or non-dollar counts --
# moved to earnings_agents.extraction.merger (imported above).

# Raw table values larger than this are assumed to already be full USD -- moved to merger.

# ── Targeted extraction (normalize_data mode) ────────────────────────────────

# Prompt used when target_concepts are loaded from normalize_data.
# The LLM is given the company's display label as the JSON key (reliable echo)
# and the XBRL taxonomy key as a bracketed hint for semantic grounding.
# Mapping back to concept_id uses normalized label matching (case/whitespace
# insensitive) so minor label drift does not break the round-trip.
_TARGETED_PROMPT_TEMPLATE = """\
You are a financial data extraction assistant.

Extract ONLY the income statement metrics listed below from the text excerpt for {company_name} ({ticker}).
This is chunk {chunk_num} of {total_chunks} of the full document.
{focus_hint}{scale_hint}{period_hint}
SCOPE — extract ONLY the concepts listed below.
Use EXACTLY the label shown in quotes as the JSON key — not the document's own wording.

{concept_list}
IGNORE — do NOT extract:
  • Balance sheet items, cash flow items, non-GAAP metrics, guidance / forecasts.
  • Any table from a GAAP-to-Non-GAAP reconciliation.
  • Percentage metrics (margins, growth rates).
  • Values from FOOTNOTES or parenthetical sub-tables (e.g. "Stock-based
    compensation included in the above", supplementary breakdowns). A genuine
    income-statement line item is a primary row of the main statement, not a
    footnote disclosure. If R&D, S&M, or G&A appears both as a primary row
    AND inside a footnote, take ONLY the primary-row value (the larger one).

TABLE PRIORITY: prefer the primary condensed GAAP income statement. Skip Non-GAAP tables.

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
Every value must be a plain number or null. NEVER write arithmetic expressions or
formulas as a value (e.g. do NOT write "15929 - 540 + 102" — use null if you cannot
determine the exact number from the table directly).

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

IMPORTANT — report RAW numbers, do NOT scale yourself:
  If the table says "(In millions)" and shows "82,886" → report 82886 (NOT 82886000000).
  EPS values and percentages are always reported as-is regardless of __scale__.

Text excerpt:
\"\"\"
{text}
\"\"\"
"""

# _TAXONOMY_PREFIXES, _LLM_MAP_PROMPT, _llm_map_concepts, _build_concept_prompt_list
# moved to earnings_agents.extraction.concept_mapper (imported above).

# _MIN_DOLLAR_FRACTION, _MAJOR_METRIC_RX, _RESCALE_UPPER_MULTIPLE
# moved to earnings_agents.extraction.merger (imported above).

# _PRESCAN_SCALE, _PRESCAN_SHARES_IN_THOUSANDS_RX, _PRESCAN_PERIOD_RX
# moved to earnings_agents.extraction.chunker (imported above).

# _OLLAMA_REQUEST_TIMEOUT, _CHUNK_MAX_RETRIES, _OLLAMA_SEMAPHORE
# imported from earnings_agents.tools.llm_extractor (above).


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

    timeout = GROQ_REQUEST_TIMEOUT if provider == "groq" else _OLLAMA_REQUEST_TIMEOUT
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

    # Increment attempt counter (Stage 1 — Perceive & Plan)
    attempt_num = state.get("extraction_attempts", 0) + 1
    logger.info("Extraction pass %d for %s", attempt_num, ticker)

    # Provider selection: always use the configured LLM_PROVIDER for every
    # attempt. There is no automatic Ollama→Groq escalation — retries stay on
    # the same provider. (Escalation was removed because an unfixable identity
    # violation forced 3 passes onto Groq's free tier and triggered 429s.)
    escalated_provider: str | None = None

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

    # Load static human-curated company hints (if any) — injected on every pass.
    company_hints = _load_company_hints(ticker)

    # If the reflection node left guidance from a previous pass, inject it
    # into every chunk prompt so the LLM focuses on what was missed.
    extraction_notes = state.get("extraction_notes") or ""
    focus_hint_parts: list[str] = []
    if company_hints:
        focus_hint_parts.append(f"Company-specific extraction hints:\n{company_hints}")
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

    chunks = _build_section_chunks(
        state.get("raw_sections"),
        state.get("target_concepts"),
    )
    chunk_source = "section"
    if chunks is None:
        chunks = _chunk_text(raw_text)
        chunk_source = "char"
    total = len(chunks)
    # Section provenance per chunk (parallel to `chunks`). Section chunks carry
    # a `=== LABEL ===` header identifying their GAAP statement; char/PDF chunks
    # are "unknown". Threaded into `_merge_metrics` so income-statement values
    # are never averaged with the same key leaking from a supplementary table.
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
    if target_concepts:
        concept_list_str = _build_concept_prompt_list(target_concepts)
        prompts = [
            _TARGETED_PROMPT_TEMPLATE.format(
                company_name=state["company_name"],
                ticker=ticker,
                chunk_num=i,
                total_chunks=total,
                focus_hint=focus_hint,
                scale_hint=scale_hint,
                period_hint=period_hint,
                concept_list=concept_list_str,
                text=chunk,
            )
            for i, chunk in enumerate(chunks, start=1)
        ]
    else:
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
                executor.submit(
                    _invoke_chunk_with_retry,
                    prompt,
                    i + 1,
                    total,
                    ticker,
                    shares_multiplier,
                    dollar_multiplier,
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
                dollar_multiplier,
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

    # On retry passes, keep untouched metrics from the previous pass and
    # overwrite only keys returned by the retried chunk(s).  This applies to
    # both scoped retries (only a subset of chunks re-run) and full retries
    # (all chunks re-run but some keys the LLM returns as null were already
    # correct from pass 1).
    if scoped_retry or attempt_num > 1:
        prev_metrics = state.get("metrics")
        if isinstance(prev_metrics, dict):
            metrics = {**prev_metrics, **metrics}

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
        concept_metrics = {}
        concept_id_to_metric_key: dict[str, str] = {}  # reverse map for mapped_metric_keys
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if key in exact_label_to_id:
                cid = exact_label_to_id[key]
                concept_metrics[cid] = float(value)
                concept_id_to_metric_key[cid] = key
            elif _norm_label(key) in norm_label_to_id:
                cid = norm_label_to_id[_norm_label(key)]
                concept_metrics[cid] = float(value)
                concept_id_to_metric_key[cid] = key

        # ── Tier 2: LLM semantic mapping for residual unmapped concepts ─────
        mapped_ids = set(concept_metrics.keys())
        unmapped_concepts = [c for c in target_concepts if c["_id"] not in mapped_ids]
        numeric_keys = [k for k, v in metrics.items() if isinstance(v, (int, float))]
        if unmapped_concepts and numeric_keys:
            logger.info(
                "Targeted mode: %d concept(s) unmatched after label lookup for %s "
                "— running LLM semantic mapping",
                len(unmapped_concepts), ticker,
            )
            map_timeout = GROQ_REQUEST_TIMEOUT if escalated_provider == "groq" else _OLLAMA_REQUEST_TIMEOUT
            map_llm = build_llm(format_json=True, request_timeout=map_timeout, provider=escalated_provider)
            llm_matches = _llm_map_concepts(
                numeric_keys,
                unmapped_concepts,
                map_llm,
                already_used_keys=set(concept_id_to_metric_key.values()),
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

        logger.info(
            "Targeted mode: mapped %d/%d concept(s) to concept_id for %s",
            len(concept_metrics),
            len(target_concepts),
            ticker,
        )
        # Diagnostic: show which target concepts remain unmapped.
        final_mapped_ids = set(concept_metrics.keys())
        absent_from_response = sorted(
            c["label"] for c in target_concepts
            if c["_id"] not in final_mapped_ids
        )
        if absent_from_response:
            logger.debug(
                "Targeted mode: %d concept(s) not extracted (null or not in press release) for %s: %s",
                len(absent_from_response), ticker, absent_from_response,
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
    if concept_metrics is not None:
        new_state["concept_metrics"] = concept_metrics
        new_state["mapped_metric_keys"] = list(concept_id_to_metric_key.values())
    return new_state
