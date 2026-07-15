"""Text chunking, section splitting, and document pre-scanning helpers."""
from __future__ import annotations

import re

from earnings_agents.config import CHUNK_OVERLAP as _CHUNK_OVERLAP, CHUNK_SIZE as _CHUNK_SIZE, default_chunk_size

# When a character boundary falls mid-line, snap at most this many chars
# backward (for the chunk end) or forward (for the overlap start) to the
# nearest newline, keeping financial table rows intact inside one chunk.
_BOUNDARY_SNAP = 200

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

# Reverse map: chunk-header label -> section key. Used to recover the source
# statement of a parsed chunk result so the merge step can prefer the value
# from a primary GAAP statement over the same key leaking from a supplementary
# ("other") table.
_LABEL_TO_SECTION: dict[str, str] = {label: key for key, label in _SECTION_CHUNK_LABELS}

# Merge authority -- LOWER number wins. A numeric metric is taken from the
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

# Pre-scan patterns applied to the full document text BEFORE chunking.
# Detected scale/period are injected as confirmed ground truth into every
# chunk prompt, eliminating wrong-scale errors that occur when the
# "(In millions)" table header only appears in the first chunk.
#
# Shared prefix for non-parenthesised scale headings. Matches the start of a
# line (re.MULTILINE) optionally beginning with a currency mark and one finance
# qualifier word, immediately followed by "in <unit>".  ``[^\S\n]`` matches
# horizontal whitespace only so the ``^`` anchor stays line-scoped.
_PRESCAN_HEADING_PREFIX = (
    r"^[^\S\n]*\$?[^\S\n]*"
    r"(?:(?:u\.?[^\S\n]?s\.?[^\S\n]+)?(?:dollars|amounts|all[^\S\n]+figures|figures)[^\S\n]+)?"
    r"in[^\S\n]+"
)

# Parenthesised table captions take priority — they are the strongest,
# least ambiguous scale signal. Match "(in millions)",
# "(Amounts in millions, except ...)", "($ in thousands)" etc.
_PRESCAN_SCALE_PARENS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\([^)]{0,30}?\bin millions\b", re.I), "millions"),
    (re.compile(r"\([^)]{0,30}?\bin thousands\b", re.I), "thousands"),
    (re.compile(r"\([^)]{0,30}?\bin billions\b", re.I), "billions"),
]

# Non-parenthesised scale headings that sit on their OWN line above a
# table, e.g. "Dollars in thousands", "$ in millions",
# "In thousands, except per share data",
# "All figures in millions unless otherwise noted".
# Anchored at line start (re.MULTILINE) with only finance qualifier words
# allowed before "in <unit>", so narrative prose such as
# "Revenue was $132.4 million in the quarter" (a number precedes the unit,
# and the line does not begin with "in <unit>") never matches.
_PRESCAN_SCALE_HEADINGS: list[tuple[re.Pattern, str]] = [
    (re.compile(_PRESCAN_HEADING_PREFIX + r"millions\b", re.I | re.M), "millions"),
    (re.compile(_PRESCAN_HEADING_PREFIX + r"thousands\b", re.I | re.M), "thousands"),
    (re.compile(_PRESCAN_HEADING_PREFIX + r"billions\b", re.I | re.M), "billions"),
]

# Backward-compatible flat list (parens first, then headings). Re-exported.
_PRESCAN_SCALE: list[tuple[re.Pattern, str]] = (
    _PRESCAN_SCALE_PARENS + _PRESCAN_SCALE_HEADINGS
)

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


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping chunks, snapping boundaries to newlines.

    When a character-based boundary falls mid-line, the split point is moved
    backward to the last newline within ``_BOUNDARY_SNAP`` chars, keeping
    financial table rows intact inside a single chunk.  The overlap window is
    similarly snapped forward to a newline so each chunk begins at a clean
    line boundary.
    """
    if chunk_size <= 0:
        chunk_size = 400000  # fallback for auto mode (CHUNK_SIZE=0) without explicit pass
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


def _section_of_chunk(chunk_text: str) -> str:
    """Recover the section key from a chunk's ``=== LABEL ===`` header.

    Returns ``"unknown"`` for char-split chunks that carry no header.
    """
    if not chunk_text.startswith("=== "):
        return "unknown"
    end = chunk_text.find(" ===", 4)
    if end == -1:
        return "unknown"
    label = chunk_text[4:end].strip()
    return _LABEL_TO_SECTION.get(label, "unknown")


def _prescan_document(raw_text: str) -> tuple[str | None, str | None, str | None]:
    """Scan the full document once for scale and current reporting period.

    Returns (scale, shares_scale, period) -- any may be None if not detected.

    ``shares_scale`` is set when the document explicitly states that share
    counts use a different scale than dollar amounts (e.g. Apple's
    "in millions, except number of shares which are reflected in thousands").
    When ``shares_scale`` is None the dollar scale is used for share counts.

    These are injected as confirmed ground truth into every chunk prompt,
    eliminating wrong-scale errors that arise when middle chunks don't see
    the "(In millions)" table header that only appeared in chunk 1.
    """
    # Normalise horizontal whitespace before matching. HTML earnings releases
    # routinely separate the words inside a scale caption with non-breaking
    # spaces (e.g. "(in\xa0thousands)"), narrow/thin spaces, or runs of
    # ordinary spaces. The scale patterns use a literal space in "in millions",
    # so without this step those captions are silently missed and the document
    # scale falls through to the LLM's (sometimes wrong) guess. ``[^\S\n]+``
    # collapses every horizontal whitespace run — including Unicode spaces —
    # to a single ASCII space while preserving newlines so the line-anchored
    # heading patterns (re.MULTILINE) stay line-scoped.
    text = re.sub(r"[^\S\n]+", " ", raw_text)

    # Scale detection uses FREQUENCY then POSITION, not list-order.
    #
    # A press release routinely mixes scale captions:
    #   • The primary GAAP financial statements (income, balance sheet, cash
    #     flows) each carry their own "(In thousands…)" caption → typically
    #     3–6 matches for the dominant scale.
    #   • Supplemental sections (Non-GAAP reconciliation, business outlook)
    #     may use a different scale ("(in millions)") → usually 1–2 matches.
    #
    # Rule: the scale with the MOST parenthesised-caption occurrences is the
    # authoritative document scale.  Ties (rare) break by earliest occurrence.
    # If no parenthesised caption exists, fall back to the earliest
    # non-parenthesised heading that matches a known scale.
    #
    # This is strictly more robust than position-only:
    #   • Position-only still works when the primary statements come first
    #     (the common case, including PENG Q3-26).
    #   • Frequency-over-position additionally handles filings where a
    #     supplemental/outlook section appears *before* the GAAP statements
    #     but those statements contain many more matching captions.
    def _dominant_scale(patterns: list[tuple[re.Pattern, str]]) -> str | None:
        counts: dict[str, int] = {}
        first_pos: dict[str, int] = {}
        for pattern, scale_name in patterns:
            for m in pattern.finditer(text):
                counts[scale_name] = counts.get(scale_name, 0) + 1
                if scale_name not in first_pos or m.start() < first_pos[scale_name]:
                    first_pos[scale_name] = m.start()
        if not counts:
            return None
        max_count = max(counts.values())
        candidates = [s for s, c in counts.items() if c == max_count]
        # Tie-break: earliest occurrence wins among equally frequent scales.
        return min(candidates, key=lambda s: first_pos[s])

    scale: str | None = _dominant_scale(_PRESCAN_SCALE_PARENS) or _dominant_scale(
        _PRESCAN_SCALE_HEADINGS
    )

    shares_scale: str | None = None
    if _PRESCAN_SHARES_IN_THOUSANDS_RX.search(text):
        shares_scale = "thousands"

    period: str | None = None
    m = _PRESCAN_PERIOD_RX.search(text)
    if m:
        period = m.group(0)

    return scale, shares_scale, period


def _build_period_hint(
    sec_report_date_str: str | None,
    doc_period: str | None,
    is_annual: bool,
) -> str:
    """Build the CONFIRMED PERIOD instruction injected into every chunk prompt.

    Annual (10-K) filings present both the single-quarter (e.g. "Three Months
    Ended") and the full-year (e.g. "Twelve Months Ended") columns side by
    side. For these the full-year column is the one we want, so the duration
    rule flips from "shortest" to "longest". For quarterly (10-Q) filings the
    single-quarter column is correct.
    """
    if sec_report_date_str:
        try:
            from datetime import date as _date

            _rd = _date.fromisoformat(sec_report_date_str)
            _formatted = _rd.strftime("%B %-d, %Y")  # e.g. "April 27, 2026"
        except ValueError:
            _formatted = None
        if _formatted is not None:
            if is_annual:
                duration_rule = (
                    f"If multiple columns share this end date but cover different durations "
                    f"(e.g. both 'Three Months Ended' and 'Twelve Months Ended' end on {_formatted}), "
                    f"always choose the LONGEST duration — the full-year column "
                    f"(e.g. 'Twelve Months Ended'), NOT the single-quarter column. "
                    f"This is an ANNUAL (full-year) filing. "
                )
            else:
                duration_rule = (
                    f"If multiple columns share this end date but cover different durations "
                    f"(e.g. both 'Three Months Ended' and 'Nine Months Ended' end on {_formatted}), "
                    f"always choose the SHORTEST duration — the single-quarter column, "
                    f"NOT the year-to-date column. "
                )
            return (
                f"CONFIRMED PERIOD: the current reporting period ends {_formatted} "
                f"(per SEC filing) — extract values from the column with this date. "
                f"{duration_rule}"
                f"Do NOT extract guidance, forecasts, or next-quarter projections.\n"
            )

    if doc_period:
        return (
            f"CONFIRMED PERIOD: current reporting period is \"{doc_period}\" — "
            f"set __period__ = \"{doc_period}\" and extract values from this column only. "
            f"Do NOT extract guidance, forecasts, or next-quarter projections.\n"
        )
    return (
        "IMPORTANT: extract values from the MOST RECENT ACTUAL reported quarter only. "
        "Do NOT extract guidance, forecasts, or next-quarter projections.\n"
    )


def _build_section_chunks(
    raw_sections: dict | None,
    target_concepts: list[dict] | None = None,
    chunk_size: int | None = None,
) -> list[str] | None:
    """Return chunks of classified GAAP tables, or None if unavailable.

    When the HTML extractor has classified tables (``raw_sections`` present),
    tables are assembled in statement order (income -> balance -> cash -> other)
    and then split by ``_chunk_text`` using the configured ``_CHUNK_SIZE``.
    This means:
    - With Groq (CHUNK_SIZE=400 000): all tables in one LLM call -- no merge needed.
    - With Ollama (CHUNK_SIZE=6 000): split into per-table chunks as before.

    Non-GAAP reconciliation tables are skipped entirely because none of their
    values map to the GAAP income-statement / balance-sheet / cash-flow
    registries.

    Only income statement and supplementary FINANCIAL DATA ("other") sections
    are sent to the LLM.  Balance-sheet and cash-flow tables are excluded:
    the extraction prompt is scoped to income statement metrics only, and
    sending unrelated statement tables wastes context window tokens while
    producing only null responses for every non-IS concept.

    Targeted concepts are required: generic extraction has been removed, so
    without ``target_concepts`` there is nothing to scope to and ``None`` is
    returned (the caller short-circuits before reaching this point in
    production).

    Returns ``None`` for PDF documents or the HTML fallback path (no GAAP
    tables classified) so the caller can fall back to ``_chunk_text``.
    """
    if not raw_sections:
        return None

    if not target_concepts:
        return None

    # Only income statement + supplementary ("other") tables are sent.
    # Balance-sheet and cash-flow sections are excluded regardless of what
    # statement_type values appear in target_concepts — the extraction prompt
    # is IS-only and those tables waste context window budget.
    allowed_keys: set[str] = {"income_statement", "other"}

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
    effective_size = chunk_size if chunk_size is not None else _CHUNK_SIZE
    if effective_size <= 0:
        effective_size = 400000  # old default when auto is active but no provider context
    return _chunk_text(combined, effective_size, _CHUNK_OVERLAP)
