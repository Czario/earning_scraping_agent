from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any, Optional

from earnings_agents.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EXTRACTION_MAX_CHARS,
    GROQ_API_KEY,
    GROQ_REQUEST_TIMEOUT,
    LLM_PROVIDER,
    OLLAMA_CONCURRENCY,
)
from earnings_agents.hooks import get_detail_callback, report_detail, set_detail_callback
from earnings_agents.llm_factory import build_llm
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)

# Human-curated hint files: data/company_hints/{TICKER}.md
_HINTS_DIR = Path(__file__).parents[3] / "data" / "company_hints"


def _load_company_hints(ticker: str) -> str:
    """Return contents of data/company_hints/{TICKER}.md, or empty string."""
    hint_file = _HINTS_DIR / f"{ticker.upper()}.md"
    if hint_file.is_file():
        content = hint_file.read_text(encoding="utf-8").strip()
        if content:
            logger.info("Loaded company hints for %s (%d chars)", ticker, len(content))
            return content
    return ""


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

# Keys whose values are percentages, per-share amounts, or non-dollar counts —
# excluded from the sanity check AND from scale multiplication.
# "gross margin" (standalone) is treated as a percentage; "gross profit" is the dollar form.
_PCT_OR_PER_SHARE_PATTERNS = re.compile(
    r"(%|percent|\bgrowth\b|\bratio\b|\beps\b|per share|\byield\b|\brate\b|\byoy\b|\bpct\b"
    r"|\bemployee\b|\bheadcount\b|basis points|percentage points"
    r"|\bgross margin\b|\boperating margin\b|\bnet margin\b|\bprofit margin\b|\bmargin\s*%"
    # XBRL-style per-share suffixes: "Per Basic Share", "Per Diluted Share", etc.
    r"|\bper\s+(?:basic|diluted|basic\s+and\s+diluted|common)\s+share\b"
    # Operational unit counts — physical quantities, never dollar-scaled
    r"|\bproduction\b|\bdeliveries\b|\bdelivered\b"
    r"|(?:super)?charger.{0,12}(?:station|connector)"
    r"|\bstations?\b|\bconnectors?\b"
    r"|\bdays.{0,5}supply\b|\blease count\b"
    r"|\bactive\b.{0,20}\bsubscriptions?\b|\bfsd subscriptions?\b)",
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


_TAXONOMY_PREFIXES = ("us-gaap:", "ifrs-full:", "dei:", "srt:")

# ── LLM semantic concept mapper ───────────────────────────────────────────────

_LLM_MAP_PROMPT = """\
You are a financial concept mapper.

Below are metric keys extracted from an earnings press release and a list of
target concepts (XBRL tag + display label + concept_id) that we want to map to.

For each target concept, decide which extracted key best matches it (if any).

Rules:
  1. Each extracted key may be assigned to AT MOST ONE concept.
  2. Only assign a key when you are confident — do NOT guess.
  3. If no extracted key fits a concept, return null for that concept.
  4. Do not invent new keys; only use keys from the extracted list.

Extracted metric keys:
{extracted_keys}

Target concepts:
{concept_rows}

Return ONLY a flat JSON object mapping concept_id → matched extracted key (or null):
{{"<concept_id>": "<extracted_key_or_null>", ...}}
"""


def _llm_map_concepts(
    extracted_keys: list[str],
    unmapped_concepts: list[dict],
    llm: object,
    already_used_keys: set[str] | None = None,
) -> dict[str, str]:
    """Ask the LLM to semantically match *extracted_keys* to *unmapped_concepts*.

    Returns a dict of ``concept_id → extracted_key`` for confident matches only.
    Null/missing entries from the LLM response are silently dropped.

    Guardrails:
    - LLM can only pick from the supplied *extracted_keys* list (no hallucination).
    - Each extracted key is used at most once (first assignment wins).
    - Keys in *already_used_keys* (Tier-1 deterministic matches) are excluded
      from the pool so the LLM cannot reassign them to a different concept.
    - Non-string or null LLM return values are discarded.
    """
    if not extracted_keys or not unmapped_concepts:
        return {}

    keys_block = "\n".join(f'  - "{k}"' for k in extracted_keys)
    rows_block = "\n".join(
        f'  - concept_id: "{c["_id"]}"  '
        f'GAAP: {c.get("concept", "")}  '
        f'label: "{c.get("label", "")}"'
        for c in unmapped_concepts
    )
    prompt = _LLM_MAP_PROMPT.format(
        extracted_keys=keys_block,
        concept_rows=rows_block,
    )
    import json as _json
    try:
        raw = llm.invoke(prompt)  # type: ignore[union-attr]
        if hasattr(raw, "content"):
            raw = raw.content
        raw = str(raw).strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        mapping: dict = _json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM concept mapping call failed: %s", exc)
        return {}

    valid_keys = set(extracted_keys)
    used_keys: set[str] = set(already_used_keys) if already_used_keys else set()
    result: dict[str, str] = {}
    for concept_id, matched_key in mapping.items():
        if not isinstance(matched_key, str):
            continue  # null or wrong type
        if matched_key not in valid_keys:
            logger.debug(
                "_llm_map_concepts: ignoring hallucinated key %r for concept %s",
                matched_key, concept_id,
            )
            continue
        if matched_key in used_keys:
            logger.debug(
                "_llm_map_concepts: key %r already used, skipping concept %s",
                matched_key, concept_id,
            )
            continue
        result[concept_id] = matched_key
        used_keys.add(matched_key)
    return result


def _build_concept_prompt_list(target_concepts: list[dict]) -> str:
    """Format concept list for the targeted prompt.

    Each line: ``  • "Label"  [concept_tag]``
    Listed in path order (income statement order).

    The label is the JSON key contract — the LLM reliably echoes the quoted
    string verbatim.  The bracketed concept tag is a semantic grounding hint
    (the LLM knows US-GAAP taxonomy from training) and assists the post-
    extraction LLM mapping step.
    """
    lines: list[str] = []
    for c in target_concepts:
        label = c.get("label", "")
        if not label:
            continue
        concept = c.get("concept", "") or ""
        concept_lc = concept.lower()
        if any(concept_lc.startswith(p) for p in _TAXONOMY_PREFIXES):
            lines.append(f'  \u2022 "{label}"  [{concept}]')
        else:
            lines.append(f'  \u2022 "{label}"')
    return "\n".join(lines)

# Minimum fraction of the largest revenue-like value that a dollar field must
# have to be considered plausible (filters out residual unscaled cells).
_MIN_DOLLAR_FRACTION = 0.001   # 0.1 % of revenue

# Major financial metrics that should always represent a significant share of
# revenue.  When their post-scale value falls below _MIN_DOLLAR_FRACTION we
# attempt a ×1 000 scale correction (one tier up: millions → billions, etc.)
# before discarding.  Non-major metrics (specific investing / financing line
# items) can be legitimately tiny and are kept as-is.
_MAJOR_METRIC_RX = re.compile(
    r"\brevenue\b|\bnet sales\b"
    r"|\bgross profit\b"
    r"|\boperating income\b|\boperating profit\b|\boperating loss\b"
    r"|\bebit\b|\bebitda\b"
    r"|\bnet income\b|\bnet earnings\b|\bnet loss\b",
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
    # Match "(in millions)" or "(Amounts in millions, except ...)" etc.
    (re.compile(r"\([^)]{0,30}?\bin millions\b", re.I), "millions"),
    # Match "(in thousands)" or "(Amounts in thousands, except ...)" etc.
    (re.compile(r"\([^)]{0,30}?\bin thousands\b", re.I), "thousands"),
    # Match "(in billions)" or "(Amounts in billions, except ...)" etc.
    (re.compile(r"\([^)]{0,30}?\bin billions\b", re.I), "billions"),
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
    r"|(?:First|Second|Third|Fourth)\s+Quarter\s+(?:Fiscal\s+)?20\d{2}"
    # Annual periods: "Year Ended December 31, 2025",
    # "Fiscal Year Ended March 31, 2026", "Full Year 2025"
    r"|(?:Fiscal\s+)?Year\s+Ended\s+"
    r"(?:March|June|September|December|Jan(?:uary)?|Feb(?:ruary)?"
    r"|Apr(?:il)?|May|Jul(?:y)?|Aug(?:ust)?|Oct(?:ober)?|Nov(?:ember)?)\s+"
    r"\d{1,2},\s*\d{4}"
    r"|Full\s+Year\s+20\d{2}"
    r"|(?:Fiscal\s+)?Year\s+20\d{2}",
    re.I,
)


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
    """Invoke the LLM for one chunk, retrying with a stricter prefix on parse failure.

    Each worker creates its own LLM client instance; sharing one instance
    across threads can block intermittently under parallel load.

    ``provider`` overrides the configured LLM_PROVIDER for this chunk only
    (used by the escalation path to switch to Groq on retry passes).
    """
    if detail_callback is not None:
        set_detail_callback(detail_callback)

    timeout = GROQ_REQUEST_TIMEOUT if provider == "groq" else _OLLAMA_REQUEST_TIMEOUT
    llm = build_llm(format_json=True, request_timeout=timeout, provider=provider)
    for attempt in range(max_retries + 1):
        if report_chunk is not None:
            report_chunk(chunk_num - 1, "running", attempt + 1)
        else:
            report_detail(f"chunk {chunk_num}/{total_chunks} attempt {attempt + 1}")
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
            # The semaphore throttles concurrent local Ollama calls.
            # Skip it when escalated to Groq (cloud API, no local concurrency limit).
            _ctx = _OLLAMA_SEMAPHORE if provider != "groq" else _nullcontext()
            with _ctx:
                response: str = llm.invoke(prefix + prompt)
            logger.debug(
                "Chunk %d/%d attempt %d raw response for %s: %r",
                chunk_num, total_chunks, attempt + 1, ticker, response[:300],
            )
            parsed = _parse_llm_response(
                response, shares_multiplier, prescan_dollar_multiplier
            )
            if parsed is not None:
                if report_chunk is not None:
                    report_chunk(chunk_num - 1, "done", attempt + 1)
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
    if report_chunk is not None:
        report_chunk(chunk_num - 1, "failed", max_retries + 1)
    return None


# When a character boundary falls mid-line, snap at most this many chars
# backward (for the chunk end) or forward (for the overlap start) to the
# nearest newline, keeping financial table rows intact inside one chunk.
_BOUNDARY_SNAP = 200


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks, snapping boundaries to newlines.

    When a character-based boundary falls mid-line, the split point is moved
    backward to the last newline within ``_BOUNDARY_SNAP`` chars, keeping
    financial table rows intact inside a single chunk.  The overlap window is
    similarly snapped forward to a newline so each chunk begins at a clean
    line boundary.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Snap end backward to the last newline within _BOUNDARY_SNAP chars.
            # Guard: search from at least start+1 so end always advances.
            snap_from = max(start + 1, end - _BOUNDARY_SNAP)
            nl = text.rfind("\n", snap_from, end)
            if nl != -1:
                end = nl + 1  # include the trailing newline in this chunk
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Next chunk overlaps by _overlap_ chars; snap its start forward to the
        # next newline so it begins at a clean line boundary.
        # Search only up to end-1 to guarantee next_start < end (no infinite loop).
        next_start = max(start + 1, end - overlap)
        nl = text.find("\n", next_start, min(next_start + _BOUNDARY_SNAP, end - 1))
        if nl != -1:
            next_start = nl + 1
        start = next_start
    return chunks


# Labels for each GAAP section that becomes its own LLM chunk.
# Order controls which chunk index a given statement type receives, which in
# turn controls which __scale__/__period__ the merge step adopts on conflict.
# Income statement first so it's chunk 1 (highest authority).
_SECTION_CHUNK_LABELS: list[tuple[str, str]] = [
    ("income_statement", "GAAP INCOME STATEMENT"),
    ("balance_sheet",    "GAAP BALANCE SHEET"),
    ("cash_flow",        "GAAP CASH FLOWS"),
    ("other",            "FINANCIAL DATA"),
]

# Reverse map: chunk-header label → section key. Used to recover the source
# statement of a parsed chunk result so the merge step can prefer the value
# from a primary GAAP statement over the same key leaking from a supplementary
# ("other") table.
_LABEL_TO_SECTION: dict[str, str] = {label: key for key, label in _SECTION_CHUNK_LABELS}

# Merge authority — LOWER number wins. A numeric metric is taken from the
# highest-authority section that reported it; values from lower-authority
# sections (e.g. a segment summary in the "other" bucket) are NEVER averaged
# in. A line item belongs to exactly one primary statement, so 0-2 never
# collide for the same real key; the decisive gap is GAAP-statement (0-2) vs
# supplementary/unknown (3-4).
_SECTION_PRIORITY: dict[str, int] = {
    "income_statement": 0,
    "balance_sheet": 1,
    "cash_flow": 2,
    "other": 3,
}
_UNKNOWN_SECTION_PRIORITY = 4


def _section_of_chunk(chunk_text: str) -> str:
    """Recover the section key from a chunk's ``=== LABEL ===`` header.

    Returns ``"unknown"`` for char-split / PDF chunks that carry no header.
    """
    if not chunk_text.startswith("=== "):
        return "unknown"
    end = chunk_text.find(" ===", 4)
    if end == -1:
        return "unknown"
    label = chunk_text[4:end].strip()
    return _LABEL_TO_SECTION.get(label, "unknown")


_INCOME_STATEMENT_KEY_RX = re.compile(
    r"revenue|gross\s+profit|cost\s+of\s+(revenue|sales|goods\s+sold)|"
    r"operating\s+income|net\s+income|earnings\s+per\s+share|"
    r"interest\s+(income|expense)|income\s+tax|research\s+and\s+development|"
    r"selling,\s*general\s+and\s+administrative|operating\s+expenses",
    re.I,
)
_BALANCE_SHEET_KEY_RX = re.compile(
    r"total\s+assets|assets\b|liabilit|equity|shareholders'\s+equity|"
    r"stockholders'\s+equity|inventory|accounts\s+receivable|accounts\s+payable",
    re.I,
)
_CASH_FLOW_KEY_RX = re.compile(
    r"cash\s+flow|net\s+cash|operating\s+activities|investing\s+activities|"
    r"financing\s+activities|capital\s+expenditures|depreciation",
    re.I,
)


def _infer_section_for_metric_key(metric_key: str) -> str | None:
    """Infer likely statement section for a metric key.

    Used only for retry-scoping hints from analysis findings.
    """
    if _INCOME_STATEMENT_KEY_RX.search(metric_key):
        return "income_statement"
    if _BALANCE_SHEET_KEY_RX.search(metric_key):
        return "balance_sheet"
    if _CASH_FLOW_KEY_RX.search(metric_key):
        return "cash_flow"
    return None


def _find_flagged_chunk_indices(
    findings: list[dict],
    chunk_metric_sources: dict[str, list[int]],
) -> set[int]:
    """Return absolute chunk indices that produced high-severity flagged metrics.

    Looks up each flagged metric key in *chunk_metric_sources* (metric key →
    list of absolute chunk indices from the previous pass) to find the exact
    chunks that contributed wrong values.  Only high/critical findings are
    considered — medium/low issues do not warrant a targeted re-extract.

    Returns an empty set when no specific chunks can be identified (e.g., all
    flagged metrics are *missing* rather than wrong — we don't know which chunk
    should have contained them; the caller falls back to section-level scoping
    for those).
    """
    if not chunk_metric_sources:
        return set()
    flagged: set[int] = set()
    for f in findings:
        if not isinstance(f, dict):
            continue
        if str(f.get("severity", "")).lower() not in ("high", "critical", "error"):
            continue
        for k in (f.get("keys") or []):
            if isinstance(k, str) and k in chunk_metric_sources:
                flagged.update(chunk_metric_sources[k])
    return flagged


def _infer_retry_sections(state: EarningsAgentState) -> set[str]:
    """Infer which statement sections should be re-extracted on retry passes.

    Returns an empty set when no scoped retry can be inferred, signalling that
    the caller should process all chunks.
    """
    sections: set[str] = set()
    findings = state.get("findings") or []
    for f in findings:
        if not isinstance(f, dict):
            continue
        if str(f.get("severity", "")).lower() != "high":
            continue

        finding_type = str(f.get("type", "")).lower()
        text_blob = " ".join(
            s
            for s in [
                str(f.get("message", "")),
                str(f.get("suggested_action", "")),
            ]
            if s
        ).lower()

        if finding_type == "identity_violation":
            if "balance" in text_blob:
                sections.add("balance_sheet")
            else:
                sections.add("income_statement")

        if "income statement" in text_blob:
            sections.add("income_statement")
        if "balance sheet" in text_blob:
            sections.add("balance_sheet")
        if "cash flow" in text_blob:
            sections.add("cash_flow")

        for k in f.get("keys") or []:
            if not isinstance(k, str):
                continue
            inferred = _infer_section_for_metric_key(k)
            if inferred:
                sections.add(inferred)

    # Fall back to extraction_notes when findings are absent or coarse.
    notes = str(state.get("extraction_notes") or "").lower()
    if "income statement" in notes:
        sections.add("income_statement")
    if "balance sheet" in notes:
        sections.add("balance_sheet")
    if "cash flow" in notes:
        sections.add("cash_flow")

    return sections


def _build_section_chunks(
    raw_sections: dict | None,
    target_concepts: list[dict] | None = None,
) -> list[str] | None:
    """Return chunks of classified GAAP tables, or None if unavailable.

    When the HTML extractor has classified tables (``raw_sections`` present),
    tables are assembled in statement order (income → balance → cash → other)
    and then split by ``_chunk_text`` using the configured ``_CHUNK_SIZE``.
    This means:
    - With Groq (CHUNK_SIZE=400 000): all tables in one LLM call — no merge needed.
    - With Ollama (CHUNK_SIZE=6 000): split into per-table chunks as before.

    Non-GAAP reconciliation tables are skipped entirely because none of their
    values map to the GAAP income-statement / balance-sheet / cash-flow
    registries.

    When ``target_concepts`` is non-empty (normalize_data mode), only sections
    whose ``statement_type`` matches at least one targeted concept are
    emitted.  Sending the balance-sheet table through an income-statement-only
    targeted prompt produces all-null responses, wasting ~30-60 s per chunk
    per pass.  The ``other`` bucket (unclassified supplementary tables) is
    always included when any target is present.

    Returns ``None`` for PDF documents or the HTML fallback path (no GAAP
    tables classified) so the caller can fall back to ``_chunk_text``.
    """
    if not raw_sections:
        return None

    if target_concepts:
        # Targeted mode (normalize_data): only send sections that match at
        # least one concept's statement_type.  Always include "other" as a
        # catch-all for supplementary tables.
        allowed_keys: set[str] = {
            (c.get("statement_type") or "").strip().lower()
            for c in target_concepts
            if c.get("statement_type")
        }
        allowed_keys.discard("")
        if allowed_keys:
            allowed_keys.add("other")
    else:
        # Generic earnings mode: extraction is scoped to income statement only
        # (balance sheet and cash-flow items are intentionally excluded — see
        # critical_metrics.py).  "other" is kept as a fallback for documents
        # where the HTML extractor failed to classify the income statement
        # (e.g. everything lands in "FINANCIAL DATA").
        allowed_keys = {"income_statement", "other"}

    # Assemble all selected table entries into one ordered text block, then
    # split by _CHUNK_SIZE.  With a large CHUNK_SIZE (Groq) everything lands
    # in one chunk; with a small CHUNK_SIZE (Ollama) it splits per table.
    parts: list[str] = []
    for key, label in _SECTION_CHUNK_LABELS:
        if key not in allowed_keys:
            continue
        for entry in raw_sections.get(key) or []:
            parts.append(f"=== {label} ===\n{entry}")
    if not parts:
        return None
    combined = "\n\n".join(parts)
    return _chunk_text(combined, _CHUNK_SIZE, _CHUNK_OVERLAP)


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


def _parse_llm_response(
    response: str,
    shares_multiplier: int = 1,
    prescan_dollar_multiplier: int = 0,
) -> dict[str, Any] | None:
    """Strip markdown fences, parse JSON, and apply the __scale__ multiplier.

    The LLM is asked to report raw table values plus a ``__scale__`` sentinel.
    Python applies the exact multiplication so the LLM never has to do arithmetic.

    ``shares_multiplier`` is applied to share-denominator fields (e.g. shares
    used in EPS calculation) and may differ from the dollar multiplier when the
    document explicitly states a different scale for share counts (e.g. Apple
    reports dollars in millions but share counts in thousands).

    ``prescan_dollar_multiplier``: when > 0, overrides the LLM's ``__scale__``
    for dollar fields.  This prevents hallucinated scale labels (e.g. "millions"
    when the header says "thousands") from corrupting the merge.
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

    # Extract __scale__ — always pop it to keep the dict clean.
    scale_str = str(parsed.pop("__scale__", "as-is")).lower()
    # If the prescan detected the document scale from the header (e.g. "(In thousands)"),
    # trust that over the LLM's returned label to prevent hallucinated-scale corruption.
    if prescan_dollar_multiplier > 1:
        multiplier = prescan_dollar_multiplier
    else:
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
                # Ambiguous 'gross margin' label: percentage (≤ 100) vs dollar amount (> 100).
                # Skip scaling when the value is clearly already a percentage.
                if "gross margin" in k.lower() and abs(v) <= 100:
                    continue
                parsed[k] = v * multiplier

    return parsed


def _merge_metrics(
    results: list[dict[str, Any]],
    source_text: str = "",
    target_year: int | None = None,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Merge per-chunk extraction dicts into one dict of all discovered metrics.

    Strategy per key:
    - Numeric values: taken from the **highest-authority statement section**
      that reported the key, never averaged across sections.  A financial line
      item belongs to exactly one primary GAAP statement, so a value that
      appears in both the income-statement chunk and a supplementary
      ("other"/segment) chunk must come from the income statement — averaging
      the two corrupts it (the historical "median of two = phantom average"
      bug).  *sections* is a list parallel to *results* giving each chunk's
      section key (``"income_statement"``, ``"balance_sheet"``, ``"cash_flow"``,
      ``"other"`` or ``"unknown"``); when absent (char-split / PDF), all chunks
      share equal authority and the prior median-of-values behaviour applies.
      Within the single winning section, the median is used as a defensive
      tie-breaker if that section was split across multiple chunks.
    - When *target_year* is supplied (from the SEC submissions API
      ``reportDate``), chunks whose ``__period__`` field embeds a *different*
      year are treated as stale (prior-year comparison column) and excluded
      from numeric merging; a stale value is still a last-resort fallback for
      keys absent from every on-target chunk.  Chunks with no ``__period__``
      are treated as on-target so they are never spuriously discarded.
    - String values: longest non-null wins (most descriptive period label,
      narrative text, etc.).
    - Dollar-amount fields that are implausibly small relative to the largest
      revenue-like value are discarded (unscaled table cells) after merging.
    """
    # Per-result section authority (lower = higher authority). Parallel to
    # *results*; defaults to equal/unknown authority when sections unavailable.
    if sections is not None and len(sections) == len(results):
        priorities = [
            _SECTION_PRIORITY.get(s, _UNKNOWN_SECTION_PRIORITY) for s in sections
        ]
    else:
        priorities = [_UNKNOWN_SECTION_PRIORITY] * len(results)
    paired: list[tuple[dict[str, Any], int]] = list(zip(results, priorities))

    # ── Period-year partitioning ──────────────────────────────────────────────
    # Split chunks into "on-target" (period year matches target_year, or unknown)
    # and "stale" (period year is a different year, e.g. prior-year column).
    def _chunk_period_year(chunk: dict[str, Any]) -> int | None:
        period = chunk.get("__period__")
        if not isinstance(period, str):
            return None
        m = re.search(r"\b(20\d{2}|19\d{2})\b", period)
        return int(m.group(1)) if m else None

    if target_year is not None:
        on_target: list[tuple[dict[str, Any], int]] = []
        stale: list[tuple[dict[str, Any], int]] = []
        for chunk, prio in paired:
            cy = _chunk_period_year(chunk)
            if cy is None or cy == target_year:
                on_target.append((chunk, prio))
            else:
                stale.append((chunk, prio))
        if stale:
            logger.info(
                "_merge_metrics: %d/%d chunk(s) declared stale period year "
                "(target=%d) — excluded from numeric merge, kept as fallback",
                len(stale), len(results), target_year,
            )
    else:
        on_target = list(paired)
        stale = []

    # ── Pass 1: collect non-null values per key ──────────────────────────────
    # Numeric values are collected as (value, section_priority) pairs so the
    # merge step can prefer the highest-authority section.
    def _collect(
        chunks: list[tuple[dict[str, Any], int]],
    ) -> tuple[dict[str, list[tuple[float, int]]], dict[str, list[str]]]:
        num: dict[str, list[tuple[float, int]]] = {}
        strs: dict[str, list[str]] = {}
        for result, prio in chunks:
            for key, value in result.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    num.setdefault(key, []).append((float(value), prio))
                elif isinstance(value, str):
                    strs.setdefault(key, []).append(value)
        return num, strs

    num_on, str_on = _collect(on_target)
    num_stale, str_stale = _collect(stale)

    # Build final candidate sets: on-target wins; stale is fallback only.
    numeric_candidates: dict[str, list[tuple[float, int]]] = {}
    for key in set(num_on) | set(num_stale):
        if key in num_on:
            numeric_candidates[key] = num_on[key]
        else:
            logger.debug(
                "_merge_metrics: %r only in stale chunk(s) — including as fallback",
                key,
            )
            numeric_candidates[key] = num_stale[key]

    string_candidates: dict[str, list[str]] = {}
    for key in set(str_on) | set(str_stale):
        string_candidates[key] = str_on.get(key) or str_stale.get(key, [])

    merged: dict[str, Any] = {}

    # Numeric: take the value from the highest-authority section that reported
    # the key (income statement > balance sheet > cash flow > other > unknown).
    # Values from lower-authority sections are NEVER averaged in — a line item
    # has one true value in one statement. The median is only a tie-breaker
    # within the single winning section (defends against a split table).
    for key, pairs in numeric_candidates.items():
        best_prio = min(p for _, p in pairs)
        values = [v for v, p in pairs if p == best_prio]
        n = len(values)
        if n == 1:
            merged[key] = values[0]
        else:
            sorted_vals = sorted(values)
            lo, hi = sorted_vals[0], sorted_vals[-1]
            denom = max(abs(lo), abs(hi))
            # When values differ by ≥5× the smallest, the low value(s) are
            # almost certainly footnote / amortization-breakdown artifacts
            # that share a key label with the primary P&L line (e.g. NVDA's
            # "(A) Acquisition-related costs in Cost of revenue: $47M" vs
            # the real "Cost of revenue: $20,458M").  Take the maximum.
            if abs(lo) > 0 and abs(hi) / abs(lo) >= 5.0:
                logger.warning(
                    "Metric %r: section authority %d — values differ by %.1fx "
                    "%s; taking max %.6g (smaller value(s) likely footnote artifacts)",
                    key, best_prio, abs(hi) / abs(lo), sorted_vals, hi,
                )
                merged[key] = hi
            else:
                median = (
                    sorted_vals[n // 2]
                    if n % 2
                    else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
                )
                if denom > 0 and (hi - lo) / denom > 0.10:
                    logger.warning(
                        "Metric %r: section authority %d reported diverging values "
                        "%s; using median %.6g",
                        key, best_prio, sorted_vals, median,
                    )
                merged[key] = median

    # String: longest non-null wins (most descriptive wins, e.g. full period name).
    # Never clobbers a numeric result for the same key.
    # Exception: __period__ prefers the candidate whose embedded year is the
    # highest (most recent) — "Three Months Ended April 27, 2025" is the
    # prior-year comparison period and is longer than "First Quarter Fiscal
    # 2027" but contains a stale year.
    def _period_year(s: str) -> int:
        m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
        return int(m.group(1)) if m else 0

    for key, values in string_candidates.items():
        if key not in merged:
            if key == "__period__":
                merged[key] = max(values, key=_period_year)
            else:
                merged[key] = max(values, key=len)

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

    # ── Case-duplicate dedup (stale-chunk fallback eviction) ─────────────────
    # When target_year filtering was active, some keys in *merged* came
    # exclusively from stale (prior-year) chunks as a last-resort fallback.
    # If a case-duplicate of that key also exists from on-target chunks, the
    # stale-only variant is wrong and must be dropped so it never reaches
    # concept mapping in normalize_data.
    #
    # Example: "Cost of revenue" = 20.458B (on-target) and
    #          "Cost of Revenue" = 48B     (stale fallback only)
    # → drop "Cost of Revenue"; keep "Cost of revenue".
    #
    # Rule: for each group of keys that share the same lowercased+collapsed
    # form, if at least one came from on-target chunks AND at least one came
    # only from stale chunks, remove the stale-only keys.
    # When ALL keys in a group are stale-only (the metric genuinely only
    # appears in the comparison column), they are all retained as-is.
    if stale:
        def _nk(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip().lower()

        # Which numeric keys came from on-target chunks?
        on_target_keys: set[str] = set(num_on)

        # Build groups: normalized_form → [actual_key, ...]
        norm_groups: dict[str, list[str]] = {}
        for k in list(merged):
            norm_groups.setdefault(_nk(k), []).append(k)

        for group_keys in norm_groups.values():
            if len(group_keys) <= 1:
                continue
            ot = [k for k in group_keys if k in on_target_keys]
            stale_only = [k for k in group_keys if k not in on_target_keys]
            if ot and stale_only:
                for k in stale_only:
                    logger.debug(
                        "_merge_metrics: evicting stale-only case-duplicate %r "
                        "(on-target variant %r = %s retained)",
                        k, ot[0], merged.get(ot[0]),
                    )
                    merged.pop(k, None)

    # Drop keys where the final value is None to keep the stored document clean.
    # Note: duplicate / synonym folding is handled downstream by the
    # constrained LLM cleanup_metrics_node, not here. Keys are preserved
    # exactly as the LLM extracted them (matching company wording).
    return {k: v for k, v in merged.items() if v is not None}


def _target_year_from_report_date(report_date_str: str | None) -> int | None:
    """Extract the calendar year from a ``sec_report_date`` ``'YYYY-MM-DD'`` string.

    Returns ``None`` when the string is absent or unparseable, which disables
    the period-year partition in ``_merge_metrics`` (all chunks treated equally).
    """
    if not report_date_str:
        return None
    try:
        from datetime import date as _d
        return _d.fromisoformat(report_date_str).year
    except ValueError:
        return None


def _find_first(
    metrics: dict[str, Any], pattern: str
) -> "tuple[str, Any] | tuple[None, None]":
    """Return (key, value) for the first metric key matching *pattern*."""
    rx = re.compile(pattern, re.IGNORECASE)
    for k, v in metrics.items():
        if rx.search(k):
            return k, v
    return None, None


def _check_identity(
    name: str,
    lhs: float | None,
    rhs: float | None,
    *,
    tolerance: float = 0.005,
) -> str | None:
    """Return a warning string if |lhs - rhs| / max(|lhs|, |rhs|) > tolerance.

    Both sides must be present and numeric for the check to fire. ``tolerance``
    defaults to 0.5 % which absorbs rounding in published figures.
    """
    if lhs is None or rhs is None:
        return None
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return None
    denom = max(abs(lhs), abs(rhs))
    if denom == 0:
        return None
    drift = abs(lhs - rhs) / denom
    if drift > tolerance:
        return (
            f"{name}: computed {rhs:,.0f} vs reported {lhs:,.0f} "
            f"(drift {drift * 100:.2f}%)"
        )
    return None


def _validate_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply deterministic post-merge cross-field consistency checks.

    Returns ``(metrics, warnings)``. *warnings* lists any accounting identity
    that failed by more than 0.5 %; the caller decides whether to save anyway.

    Checks:
      1. Free Cash Flow ≤ Operating Cash Flow.
      2. "Less: purchases of property and equipment" within 50 % of direct capex.
      3. Gross margin ≈ Revenue − Cost of revenue.
      4. Operating income ≈ Gross margin − (R&D + S&M + G&A).
      5. Income before taxes ≈ Operating income + Other income (expense).
      6. Net income ≈ Income before taxes − Provision for income taxes.
      7. Diluted EPS × Diluted shares ≈ Net income (within 1 ¢ on the EPS).
    """
    result = dict(metrics)
    warnings: list[str] = []

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

    def _num(pattern: str) -> float | None:
        _, v = _find_first(result, pattern)
        return v if isinstance(v, (int, float)) else None

    revenue = _num(r"^revenue$|^total revenue$|^net sales$|^total net sales$")
    cogs = _num(r"^cost of revenue$|^total cost of revenue$|^cost of sales$")
    gross = _num(r"^gross (margin|profit)$")
    rnd = _num(r"^research and development")
    snm = _num(r"^sales and marketing|^selling and marketing")
    gna = _num(r"^general and administrative")
    op_income = _num(r"^operating income$|^operating profit$|^income from operations$")
    other_inc = _num(r"^other income.*expense|^other.*income.*net$|^other,?\s*net$")
    interest_income = _num(r"^interest income")
    interest_expense = _num(r"^interest expense")
    interest_net = _num(r"^interest income.*expense|^interest,?\s*net$|^net interest")
    income_before_tax = _num(r"^income before .*tax")
    tax = _num(r"^provision for income tax|^income tax expense|^income tax provision")
    net_income = _num(r"^net income$|^net earnings$")
    eps_d = _num(r"^diluted earnings per share$")
    shares_d = _num(r"weighted average shares outstanding:?\s*diluted")

    # ------------------------------------------------------------------
    # Identity checks are split into two tiers:
    #
    #   BLOCKING — universal sanity checks that catch catastrophic LLM
    #              errors (wrong scale, hallucinated numbers, swapped
    #              values). Work for every industry — banks, REITs,
    #              insurers, foreign issuers — because they only rely on
    #              numbers every company reports.
    #
    #   ADVISORY — structural decompositions (gross margin = revenue −
    #              COGS, operating income = gross − opex, pre-tax =
    #              op + non-op). These are useful but fragile: banks
    #              have no COGS, REITs use FFO, insurers use claims and
    #              reserves, etc. They are logged for inspection but do
    #              NOT block the save.
    # ------------------------------------------------------------------

    def _advisory(label: str, lhs: float, rhs: float, *, tolerance: float = 0.005) -> None:
        w = _check_identity(label, lhs, rhs, tolerance=tolerance)
        if w:
            logger.warning("Identity check (advisory): %s", w)

    # --- ADVISORY: structural decompositions (informational only) -----
    if revenue is not None and cogs is not None and gross is not None:
        _advisory("Gross margin = Revenue − COGS", gross, revenue - cogs)

    if gross is not None and op_income is not None and (
        rnd is not None and snm is not None and gna is not None
    ):
        _advisory("Operating income = Gross − opex", op_income, gross - rnd - snm - gna)

    non_op_parts = [x for x in (other_inc, interest_income, interest_net) if x is not None]
    non_op_sum: Optional[float] = sum(non_op_parts) if non_op_parts else None
    if non_op_sum is not None and interest_expense is not None:
        non_op_sum -= interest_expense
    if op_income is not None and income_before_tax is not None and non_op_sum is not None:
        _advisory(
            "Income before taxes = Op income + Non-operating items",
            income_before_tax,
            op_income + non_op_sum,
        )

    if income_before_tax is not None and tax is not None and net_income is not None:
        _advisory("Net income = Pre-tax − Tax", net_income, income_before_tax - tax)

    # --- BLOCKING: universal sanity checks ----------------------------
    # 1. Diluted EPS × Diluted shares ≈ Net income.
    #    The single most reliable cross-check: three numbers from the
    #    same column of the income statement, present for virtually
    #    every public company. Catches scale errors (off by 1000×) and
    #    hallucinated values. Tolerance: max(1¢, 2% of computed EPS)
    #    — the 2% band accommodates rounding when reported EPS has
    #    only 2 decimal places and shares are reported in millions.
    if eps_d is not None and shares_d is not None and net_income is not None and shares_d > 0:
        computed_eps = net_income / shares_d
        tol = max(0.01, abs(eps_d) * 0.02)
        if abs(computed_eps - eps_d) > tol:
            warnings.append(
                f"Diluted EPS sanity check: reported {eps_d:.4f} vs computed "
                f"{computed_eps:.4f} (Net income / Diluted shares)"
            )

    # 2. Scale sanity: a Revenue value below $100K or above $10T is
    #    almost certainly a scale-parsing error (the LLM read "in
    #    millions" wrong, or grabbed a per-share value).
    if revenue is not None and revenue > 0 and not (1e5 <= revenue <= 1e13):
        warnings.append(
            f"Scale sanity check: Revenue {revenue:,.0f} is outside the plausible "
            f"range [$100K, $10T] — likely a scale-parsing error"
        )

    # 3. Net income magnitude sanity: |Net income| should not exceed
    #    ~2× Revenue. Catches swapped Revenue/Net income or one-off
    #    extraction noise. (Some loss-heavy companies have NI < 0 with
    #    large magnitude, hence the 2× headroom rather than 1×.)
    if revenue is not None and net_income is not None and revenue > 0:
        if abs(net_income) > 2 * revenue:
            warnings.append(
                f"Magnitude sanity check: |Net income| {net_income:,.0f} > 2× "
                f"Revenue {revenue:,.0f} — values may be swapped or mis-scaled"
            )

    for w in warnings:
        logger.error("Identity check (blocking): %s", w)

    return {k: v for k, v in result.items() if v is not None}, warnings



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
    if sec_report_date_str:
        try:
            from datetime import date as _date
            _rd = _date.fromisoformat(sec_report_date_str)
            _formatted = _rd.strftime("%B %-d, %Y")  # e.g. "April 27, 2026"
            period_hint = (
                f"CONFIRMED PERIOD: the current reporting period ends {_formatted} "
                f"(per SEC filing) — extract values from the column with this date. "
                f"If multiple columns share this end date but cover different durations "
                f"(e.g. both 'Three Months Ended' and 'Nine Months Ended' end on {_formatted}), "
                f"always choose the SHORTEST duration — the single-quarter column, "
                f"NOT the year-to-date column. "
                f"Do NOT extract guidance, forecasts, or next-quarter projections.\n"
            )
        except ValueError:
            sec_report_date_str = None  # fall through to prescan

    if not sec_report_date_str:
        period_hint = (
            f"CONFIRMED PERIOD: current reporting period is \"{doc_period}\" — "
            f"set __period__ = \"{doc_period}\" and extract values from this column only. "
            f"Do NOT extract guidance, forecasts, or next-quarter projections.\n"
            if doc_period else (
                "IMPORTANT: extract values from the MOST RECENT ACTUAL reported quarter only. "
                "Do NOT extract guidance, forecasts, or next-quarter projections.\n"
            )
        )

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

    metrics, identity_warnings = _validate_metrics(metrics)
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
