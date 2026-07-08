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
    effective = (provider or LLM_PROVIDER).strip().lower()
    if effective == "groq":
        return GROQ_REQUEST_TIMEOUT
    if effective == "gemini":
        return GEMINI_REQUEST_TIMEOUT
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

    # Source-grounding block is opt-in (SOURCE_GROUNDING): it roughly doubles
    # the LLM output size, so it is omitted by default for speed.
    sources_hint = _SOURCES_HINT_BLOCK if SOURCE_GROUNDING else ""
    prompts = [
        _TARGETED_PROMPT_TEMPLATE.format(
            company_name=state["company_name"],
            ticker=ticker,
            chunk_num=i,
            total_chunks=total,
            focus_hint=focus_hint,
            scale_hint=chunk_scale_hints[i - 1],
            period_hint=period_hint,
            concept_list=concept_list_str,
            sources_hint=sources_hint,
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
        taxonomy_key_to_id: dict[str, str] = {
            c["taxonomy_key"]: c["_id"]
            for c in target_concepts
            if c.get("taxonomy_key")
        }
        # Bracket-wrapped form — the primary format after the prompt change.
        bracket_key_to_id: dict[str, str] = {
            f"[{c['taxonomy_key']}]": c["_id"]
            for c in target_concepts
            if c.get("taxonomy_key")
        }
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

        if unmapped_concepts and orphaned_keys:
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
                logger.info(
                    "Targeted mode: %d concept(s) unmatched after label lookup for %s "
                    "— running LLM semantic mapping (%d matchable, %d orphaned key(s))",
                    len(unmapped_concepts), ticker, len(tier2_candidates), len(orphaned_keys),
                )
                report_call(
                    f"  [tier2]  {len(unmapped_concepts)} unmapped concept(s)"
                    f"  ({len(tier2_candidates)} matchable,"
                    f" {len(orphaned_keys)}/{len(all_numeric_keys)} orphaned keys)"
                    f"  → calling llm  ({_display_provider})"
                )
                map_timeout = _request_timeout_for(escalated_provider)
                map_llm = build_llm(format_json=True, request_timeout=map_timeout, provider=escalated_provider)
                llm_matches = _llm_map_concepts(
                    orphaned_keys,       # only unmapped keys — already-used keys excluded
                    tier2_candidates,
                    map_llm,
                    already_used_keys=used_keys,
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
        elif unmapped_concepts and not orphaned_keys:
            # All extracted keys were already consumed by tier0/tier1.
            # The extraction LLM saw every concept label in its prompt and
            # returned null — the filing genuinely doesn't contain those values.
            # Tier 2 can find nothing new → skip it.
            logger.info(
                "Tier2 skipped for %s: 0 orphaned keys (%d/%d extracted keys already "
                "mapped at tier0/tier1) — unmapped concepts are absent from filing",
                ticker, len(used_keys), len(all_numeric_keys),
            )
            report_call(
                f"  [tier2]  skipped (0 orphaned keys — all {len(all_numeric_keys)}"
                f" extracted values already matched; {len(unmapped_concepts)}"
                f" concepts absent from filing)"
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
