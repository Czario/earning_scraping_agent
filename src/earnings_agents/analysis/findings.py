"""Structured ``Finding`` model + deterministic checkers.

A ``Finding`` is the unit of communication between the analysis node and
downstream consumers (``cleanup_metrics``, the re-extract loop). Each finding
carries enough metadata for a consumer to act without re-running heuristics.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from earnings_agents.analysis.metric_patterns import (
    BS_ASSETS_RE as _BS_ASSETS_RE,
    BS_EQUITY_RE as _BS_EQUITY_RE,
    BS_LIAB_RE as _BS_LIAB_RE,
    COGS_RE as _COGS_RE,
    COMPOSITE_COMMA_RE as _COMPOSITE_COMMA_RE,
    COMPOSITE_SLASH_RE as _COMPOSITE_SLASH_RE,
    EPS_BASIC_RE as _EPS_BASIC_RE,
    EPS_DILUTED_RE as _EPS_DILUTED_RE,
    GAAP_NONGAAP_RE as _GAAP_NONGAAP_RE,
    GROSS_PROFIT_RE as _GROSS_PROFIT_RE,
    NEVER_ROUND_RE as _NEVER_ROUND_RE,
    OPINC_RE as _OPINC_RE,
    OPEX_SUBTOTAL_RE as _OPEX_SUBTOTAL_RE,
    OPEX_TOTAL_RE as _OPEX_TOTAL_RE,
    REVENUE_RE as _REVENUE_RE,
    ROUND_MAX as _ROUND_MAX,
    ROUND_MIN as _ROUND_MIN,
    ROUND_UNIT as _ROUND_UNIT,
)

# Severity drives routing decisions in analyze_metrics_node:
#   "high"   → triggers a re-extract loop (if attempts remain)
#   "medium" → adds a hint to extraction_notes only when a "high" already triggers
#   "low"    → never triggers re-extract; consumed by cleanup or logged
Severity = Literal["high", "medium", "low"]

# Finding types. Keep this set closed so consumers can switch on it safely.
FindingType = Literal[
    "missing_critical",       # tier-1 metric absent
    "missing_expected",       # tier-2 metric absent
    "case_duplicate",         # two keys differ only by case/whitespace, same value
    "identity_violation",     # accounting identity does not reconcile
    "sign_anomaly",           # value carries the wrong sign for its concept
    "suspect_round",          # implausibly round number (likely narrative prose)
    "suspect_value",          # value matches a different metric — likely mis-assigned row
    "gaap_nongaap_leakage",   # key leaked from a GAAP/Non-GAAP reconciliation table
    "composite_key",          # key is a comma/slash list of synonyms, not a real label
    "auto_corrected",         # value was provably wrong and deterministically fixed
    "source_unverified",      # value not grounded in any verbatim source snippet
    # Reserved for later steps (not emitted yet):
    "section_mismatch",
]


@dataclass(frozen=True)
class Finding:
    """A single deterministic observation about extracted metrics."""

    type: FindingType
    severity: Severity
    message: str                      # human-readable one-liner
    keys: tuple[str, ...] = ()        # affected metric keys
    suggested_action: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Checkers
# -----------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize_key(k: str) -> str:
    """Case- and whitespace-insensitive canonical form."""
    return _WS_RE.sub(" ", k.strip()).casefold()


def _values_close(a: Any, b: Any, rel_tol: float = 0.001) -> bool:
    """True when two numeric values agree within *rel_tol* (0.1 %% default)."""
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return a == b
    af, bf = float(a), float(b)
    if af == bf == 0:
        return True
    return abs(af - bf) <= rel_tol * max(abs(af), abs(bf))


def check_case_duplicates(metrics: dict[str, Any]) -> list[Finding]:
    """Find key groups that normalize to the same form *and* hold the same value.

    For each collision the finding lists ALL colliding keys; the cleanup
    consumer decides which to keep (preferring Title-Case → first-seen).
    """
    groups: dict[str, list[str]] = {}
    for k in metrics:
        groups.setdefault(_normalize_key(k), []).append(k)

    findings: list[Finding] = []
    for variants in groups.values():
        if len(variants) < 2:
            continue
        # Only flag when ALL variants carry equivalent values; otherwise they
        # may legitimately be distinct metrics that happen to look similar.
        ref = metrics[variants[0]]
        if all(_values_close(metrics[v], ref) for v in variants[1:]):
            findings.append(
                Finding(
                    type="case_duplicate",
                    severity="low",
                    message=(
                        f"{len(variants)} keys collide on normalized form "
                        f"with identical values: {variants}"
                    ),
                    keys=tuple(variants),
                    suggested_action="keep one variant; drop the others",
                    evidence={"normalized": _normalize_key(variants[0]), "value": ref},
                )
            )
    return findings


def check_presence(metrics: dict[str, Any], presence: dict[str, list[str]]) -> list[Finding]:
    """Convert the tiered presence result into Finding objects."""
    out: list[Finding] = []
    for name in presence.get("tier1_missing", []):
        out.append(
            Finding(
                type="missing_critical",
                severity="high",
                message=f"Tier-1 metric not found: {name}",
                suggested_action=f"Re-extract; locate '{name}' in the source.",
                evidence={"tier": 1, "metric": name},
            )
        )
    for name in presence.get("tier2_missing", []):
        out.append(
            Finding(
                type="missing_expected",
                severity="medium",
                message=f"Tier-2 metric not found: {name}",
                suggested_action=f"Confirm whether the company reports '{name}'.",
                evidence={"tier": 2, "metric": name},
            )
        )
    return out


# -----------------------------------------------------------------------------
# Lookup helper
# -----------------------------------------------------------------------------

def _find_metric(
    metrics: dict[str, Any], pattern: str | re.Pattern
) -> tuple[str | None, Any]:
    """Return ``(matching_key, numeric_value)`` for the first key matching pattern.

    Only numeric (int/float, non-bool) values are returned. ``(None, None)``
    when nothing matches.
    """
    pat = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, re.IGNORECASE)
    for k, v in metrics.items():
        if pat.search(k) and isinstance(v, (int, float)) and not isinstance(v, bool):
            return k, float(v)
    return None, None


# -----------------------------------------------------------------------------
# Balance-sheet identity:  Total Assets ≈ Total Liabilities + Total Equity
# -----------------------------------------------------------------------------

# Balance-sheet patterns imported from metric_patterns; alias kept for reader context.
# Tolerance: 1 % accommodates rounding when components are reported in
# millions but totals in thousands (rare but documented).
_BS_IDENTITY_TOLERANCE = 0.01


def check_balance_sheet_identity(metrics: dict[str, Any]) -> list[Finding]:
    """Flag when Total Assets ≠ Total Liabilities + Total Equity (within 1 %).

    Catches the NVDA-style defect where a sibling component was mis-extracted
    and Total Liabilities ends up far off the sum of its parts.
    """
    a_key, a = _find_metric(metrics, _BS_ASSETS_RE)
    l_key, lia = _find_metric(metrics, _BS_LIAB_RE)
    e_key, eq = _find_metric(metrics, _BS_EQUITY_RE)
    if None in (a, lia, eq):
        return []

    lhs = float(a)
    rhs = float(lia) + float(eq)
    if lhs == 0:
        return []
    rel_err = abs(lhs - rhs) / abs(lhs)
    if rel_err <= _BS_IDENTITY_TOLERANCE:
        return []

    return [
        Finding(
            type="identity_violation",
            severity="high",
            message=(
                f"Balance-sheet identity broken: Total Assets ({lhs:,.0f}) ≠ "
                f"Total Liabilities + Equity ({rhs:,.0f}); diff "
                f"{lhs - rhs:,.0f} ({rel_err * 100:.2f}%)"
            ),
            keys=tuple(k for k in (a_key, l_key, e_key) if k),
            suggested_action="Re-extract balance-sheet rows; verify component sums.",
            evidence={
                "total_assets": lhs,
                "total_liabilities": float(lia),
                "total_equity": float(eq),
                "relative_error": rel_err,
            },
        )
    ]


# -----------------------------------------------------------------------------
# Sign anomalies — balance-sheet line items that should be positive.
# -----------------------------------------------------------------------------

# Each entry: (display_name, regex). Listed items are stock balances; they
# can never be legitimately negative on the balance sheet. (Contra-asset
# items like "accumulated depreciation" or cash-flow uses of cash are
# intentionally excluded.)
_POSITIVE_BS_ITEMS: list[tuple[str, re.Pattern]] = [
    ("Inventories",                re.compile(r"^\s*inventor(y|ies)\s*$", re.IGNORECASE)),
    ("Cash and equivalents",       re.compile(r"^\s*cash and (cash )?equivalents\s*$", re.IGNORECASE)),
    ("Accounts receivable",        re.compile(r"^\s*accounts receivable", re.IGNORECASE)),
    ("Total assets",               re.compile(r"^\s*total assets\b", re.IGNORECASE)),
    ("Total current assets",       re.compile(r"^\s*total current assets\b", re.IGNORECASE)),
    ("Total liabilities",          re.compile(r"^\s*total liabilities\s*$", re.IGNORECASE)),
    ("Total current liabilities",  re.compile(r"^\s*total current liabilities\b", re.IGNORECASE)),
    ("Long-term debt",             re.compile(r"^\s*long-?term debt\s*$", re.IGNORECASE)),
    ("Goodwill",                   re.compile(r"^\s*goodwill\s*$", re.IGNORECASE)),
]


def check_sign_anomalies(metrics: dict[str, Any]) -> list[Finding]:
    """Flag balance-sheet stock items that came back with a negative value.

    A negative ``Inventories`` almost always means the LLM picked up a
    cash-flow row (where Δinventory can be negative) instead of the
    balance-sheet row.
    """
    out: list[Finding] = []
    for label, pat in _POSITIVE_BS_ITEMS:
        key, val = _find_metric(metrics, pat)
        if val is None or val >= 0:
            continue
        out.append(
            Finding(
                type="sign_anomaly",
                severity="medium",
                message=(
                    f"{label} reported as negative ({val:,.0f}); the balance-sheet "
                    f"value is expected to be ≥ 0. The negative value may be a "
                    f"cash-flow delta mis-keyed as a balance."
                ),
                keys=(key,) if key else (),
                suggested_action="Re-extract; confirm whether this is a balance or a delta.",
                evidence={"metric": label, "value": val},
            )
        )
    return out


# -----------------------------------------------------------------------------
# Suspect round-number heuristic.
# -----------------------------------------------------------------------------

# Numbers that are integer multiples of 100M with no smaller digits and a
# magnitude < $100B are commonly produced when the LLM lifted a value from
# narrative prose ("debt of about $1 billion") rather than a financial
# table. Larger values (e.g. "1.0 trillion") and small values (EPS,
# percentages, share counts) are exempt.
# Suspect-round constants + guard imported from metric_patterns.


def check_suspect_round(metrics: dict[str, Any]) -> list[Finding]:
    """Flag values that look like narrative-prose pickups, not table rows."""
    out: list[Finding] = []
    for k, v in metrics.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if _NEVER_ROUND_RE.search(k):
            continue
        av = abs(float(v))
        if av < _ROUND_MIN or av >= _ROUND_MAX:
            continue
        if av % _ROUND_UNIT != 0:
            continue
        out.append(
            Finding(
                type="suspect_round",
                severity="low",
                message=(
                    f"{k!r} = {v:,.0f} is an exact multiple of $100M in a range "
                    f"typical of narrative prose rather than a reported table value."
                ),
                keys=(k,),
                suggested_action="Verify against the source statement before relying on it.",
                evidence={"value": float(v)},
            )
        )
    return out


# -----------------------------------------------------------------------------
# GAAP / Non-GAAP reconciliation table leakage.
# -----------------------------------------------------------------------------

# GAAP/Non-GAAP pattern imported from metric_patterns.


def check_gaap_nongaap_leakage(metrics: dict[str, Any]) -> list[Finding]:
    """Flag keys that leaked from a GAAP-to-Non-GAAP reconciliation table.

    These keys should be dropped in favour of the plain GAAP income statement
    values. Severity is ``"low"`` — no re-extract is needed; cleanup removes
    them deterministically.
    """
    leaking = [k for k in metrics if _GAAP_NONGAAP_RE.search(k)]
    if not leaking:
        return []
    return [
        Finding(
            type="gaap_nongaap_leakage",
            severity="low",
            message=(
                f"{len(leaking)} key(s) appear to originate from a GAAP/Non-GAAP "
                f"reconciliation table and will be dropped: "
                + ", ".join(repr(k) for k in leaking)
            ),
            keys=tuple(leaking),
            suggested_action=(
                "Drop these keys; use the plain GAAP income statement values instead."
            ),
            evidence={"leaked_keys": leaking},
        )
    ]


# -----------------------------------------------------------------------------
# Composite / synonym-list key detection.
# -----------------------------------------------------------------------------

# Detect keys that are LLM-generated synonym lists rather than real document labels.
# Two patterns — kept separate because the rules differ:
#
# SLASH pattern: " / " never appears in legitimate GAAP labels, so any key that
# matches "<finance_word> ... / ... <finance_word>" is always a composite.
# Example: "Cost of revenue / Cost of goods sold"
#
# COMMA pattern: commas DO appear in real GAAP labels ("Earnings Per Share, Basic",
# "Income (Loss) from Continuing Operations, Net of Tax, Attributable to Parent").
# We only flag a comma-joined key when the SAME financial keyword appears both
# before AND after the comma — the hallmark of an LLM copying a synonym bullet.
# Example: "Diluted earnings per share, Diluted EPS" → "diluted" on both sides.
# Composite-key patterns imported from metric_patterns.


def check_composite_keys(metrics: dict[str, Any]) -> list[Finding]:
    """Flag keys that are lists of synonyms rather than real document labels.

    These are produced when the LLM copies SCOPE instruction text verbatim
    as a key name (e.g. ``"Diluted earnings per share, Diluted EPS, ..."``).
    Severity is ``"low"``; cleanup drops them deterministically.
    """
    bad = [
        k for k in metrics
        if _COMPOSITE_SLASH_RE.search(k) or _COMPOSITE_COMMA_RE.search(k)
    ]
    if not bad:
        return []
    return [
        Finding(
            type="composite_key",
            severity="low",
            message=(
                f"{len(bad)} key(s) appear to be comma/slash-joined synonym lists "
                f"rather than real document labels: "
                + ", ".join(repr(k) for k in bad)
            ),
            keys=tuple(bad),
            suggested_action="Drop these keys; the real label was captured under a proper key.",
            evidence={"composite_keys": bad},
        )
    ]


# -----------------------------------------------------------------------------
# Opex-label collision: Total operating expenses ≡ Operating income / Revenue.
# -----------------------------------------------------------------------------

# Income-statement patterns imported from metric_patterns.


# -----------------------------------------------------------------------------
# Income-statement identity:  Revenue − Cost of revenue ≈ Gross profit
# -----------------------------------------------------------------------------

# Tolerance: 1.5 % of revenue. Wide enough for legitimate rounding when
# components are reported at different scales, tight enough to catch a
# mis-extracted column (typically tens of percent off).
_GP_IDENTITY_TOLERANCE = 0.015


def check_gross_profit_identity(metrics: dict[str, Any]) -> list[Finding]:
    """Flag when Revenue − Cost of revenue ≠ Gross profit (within tolerance).

    This is the single most diagnostic income-statement invariant. When it
    fails, at least one of the three values was extracted from the wrong
    column (e.g. a prior-year comparison) or the wrong row. Emitted at
    ``"high"`` severity so it triggers a targeted re-extract while attempts
    remain — the model is told exactly which relationship is broken.

    Two distinct failure modes are reported:
      * ``cost_exceeds_revenue`` — Cost of revenue ≥ Revenue: arithmetically
        impossible for a going concern; almost always a stale/footnote value.
      * ``reconciliation_off``  — all three present but they do not sum.
    """
    rev_key, rev = _find_metric(metrics, _REVENUE_RE)
    cogs_key, cogs = _find_metric(metrics, _COGS_RE)
    gp_key, gp = _find_metric(metrics, _GROSS_PROFIT_RE)

    # Mode 1: Cost of revenue ≥ Revenue — needs only those two values.
    if rev is not None and cogs is not None and rev > 0 and cogs >= rev:
        return [
            Finding(
                type="identity_violation",
                severity="high",
                message=(
                    f"Cost of revenue ({cogs:,.0f}) ≥ Revenue ({rev:,.0f}); "
                    f"gross profit would be negative. The value almost certainly "
                    f"came from a prior-year column or a footnote sub-table."
                ),
                keys=tuple(k for k in (rev_key, cogs_key) if k),
                suggested_action=(
                    "Re-extract 'Cost of revenue' from the CURRENT-period column "
                    "of the consolidated income statement only. It must be less "
                    "than total revenue and satisfy "
                    "Revenue − Cost of revenue = Gross profit."
                ),
                evidence={"revenue": rev, "cost_of_revenue": cogs},
            )
        ]

    # Mode 2: full reconciliation — needs all three.
    if None in (rev, cogs, gp) or rev is None or rev <= 0:
        return []

    implied_gp = float(rev) - float(cogs)
    rel_err = abs(implied_gp - float(gp)) / float(rev)
    if rel_err <= _GP_IDENTITY_TOLERANCE:
        return []

    return [
        Finding(
            type="identity_violation",
            severity="high",
            message=(
                f"Income-statement identity broken: Revenue ({rev:,.0f}) − "
                f"Cost of revenue ({cogs:,.0f}) = {implied_gp:,.0f}, but reported "
                f"Gross profit is {gp:,.0f} (off by {rel_err * 100:.1f}% of revenue)."
            ),
            keys=tuple(k for k in (rev_key, cogs_key, gp_key) if k),
            suggested_action=(
                "Re-extract Revenue, Cost of revenue, and Gross profit from the "
                "SAME current-period column of the consolidated income statement. "
                "They must satisfy Revenue − Cost of revenue = Gross profit."
            ),
            evidence={
                "revenue": float(rev),
                "cost_of_revenue": float(cogs),
                "gross_profit": float(gp),
                "implied_gross_profit": implied_gp,
                "relative_error": rel_err,
            },
        )
    ]


def check_opex_label_collision(metrics: dict[str, Any]) -> list[Finding]:
    """Flag when 'Total operating expenses' carries the same value as operating income or revenue.

    Total operating expenses (COGS + opex line items) can never equal Operating
    income (gross profit − opex) unless gross margin is zero. When they match,
    the LLM almost certainly assigned the value of a neighbouring row to the
    wrong label. Severity is ``"medium"`` — appended to extraction notes on the
    next pass but does not trigger a re-extract on its own.
    """
    opex_key, opex = _find_metric(metrics, _OPEX_TOTAL_RE)
    if opex is None:
        return []

    collisions: list[tuple[str, float]] = []
    for pat in (_OPINC_RE, _REVENUE_RE):
        other_key, other = _find_metric(metrics, pat)
        if other is not None and other_key is not None and _values_close(opex, other):
            collisions.append((other_key, other))

    if not collisions:
        return []

    collision_desc = "; ".join(f"{k}={v:,.0f}" for k, v in collisions)
    all_keys = tuple(k for k in (opex_key, *(k for k, _ in collisions)) if k is not None)
    return [
        Finding(
            type="suspect_value",
            severity="medium",
            message=(
                f"'Total operating expenses' ({opex:,.0f}) equals {collision_desc}. "
                f"These metrics cannot be equal; the LLM likely assigned the wrong row value."
            ),
            keys=all_keys,
            suggested_action=(
                "Re-extract 'Total operating expenses' from the income statement. "
                "It equals Cost of revenue + all operating expense line items "
                "(R&D, S&M, G&A), NOT Operating income."
            ),
            evidence={
                "total_operating_expenses": opex,
                "colliding_metrics": {k: v for k, v in collisions},
            },
        )
    ]


# -----------------------------------------------------------------------------
# Income-statement ordering invariants.
# -----------------------------------------------------------------------------

# Tolerance: 1 % — wide enough for legitimate rounding when components are
# reported at different scales, tight enough to catch a value lifted from the
# wrong row/column (typically tens of percent off).
_ORDERING_TOLERANCE = 0.01


def check_operating_vs_gross(metrics: dict[str, Any]) -> list[Finding]:
    """Flag when Operating income exceeds Gross profit.

    Operating income = Gross profit − Operating expenses, and operating
    expenses are never negative, so Operating income can never exceed Gross
    profit. When it does, one of the two values was lifted from the wrong row
    or column. Only checked when both values are positive — a gross or
    operating *loss* makes the ordering meaningless. Severity is ``"medium"``:
    it sharpens the next re-extract hint but never triggers a loop on its own.
    """
    gp_key, gp = _find_metric(metrics, _GROSS_PROFIT_RE)
    oi_key, oi = _find_metric(metrics, _OPINC_RE)
    if gp is None or oi is None or gp <= 0 or oi <= 0:
        return []
    if oi <= gp * (1 + _ORDERING_TOLERANCE):
        return []
    return [
        Finding(
            type="suspect_value",
            severity="medium",
            message=(
                f"Operating income ({oi:,.0f}) exceeds Gross profit ({gp:,.0f}); "
                f"operating income cannot exceed gross profit when operating "
                f"expenses are non-negative — one value was likely taken from the "
                f"wrong row or period column."
            ),
            keys=tuple(k for k in (oi_key, gp_key) if k),
            suggested_action=(
                "Re-extract Operating income and Gross profit from the SAME "
                "current-period column; Operating income = Gross profit − "
                "Operating expenses, so it must be ≤ Gross profit."
            ),
            evidence={"operating_income": oi, "gross_profit": gp},
        )
    ]


def check_eps_dilution_ordering(metrics: dict[str, Any]) -> list[Finding]:
    """Flag when Diluted EPS exceeds Basic EPS.

    Diluted EPS divides the same earnings by a *larger* share count (it adds
    dilutive securities), so for positive earnings Diluted EPS is always ≤
    Basic EPS. When Diluted > Basic the two were almost certainly swapped or
    one was read from the wrong row. Only checked when both are positive — for
    a net loss the figures are anti-dilutive and reported equal. Severity is
    ``"medium"``.
    """
    basic_key, basic = _find_metric(metrics, _EPS_BASIC_RE)
    diluted_key, diluted = _find_metric(metrics, _EPS_DILUTED_RE)
    if basic is None or diluted is None or basic <= 0 or diluted <= 0:
        return []
    if diluted <= basic * (1 + _ORDERING_TOLERANCE):
        return []
    return [
        Finding(
            type="suspect_value",
            severity="medium",
            message=(
                f"Diluted EPS ({diluted:g}) exceeds Basic EPS ({basic:g}); "
                f"diluted EPS divides earnings by a larger share count and can "
                f"never exceed basic EPS for positive earnings — the values were "
                f"likely swapped or mis-extracted."
            ),
            keys=tuple(k for k in (diluted_key, basic_key) if k),
            suggested_action=(
                "Re-extract Basic and Diluted EPS from the income statement; "
                "Diluted EPS must be ≤ Basic EPS when earnings are positive."
            ),
            evidence={"basic_eps": basic, "diluted_eps": diluted},
        )
    ]


# -----------------------------------------------------------------------------
# Deterministic correction: derive Total operating expenses from components.
# -----------------------------------------------------------------------------

def derive_corrected_total_opex(
    metrics: dict[str, Any],
) -> tuple[str | None, float | None]:
    """Return ``(opex_key, corrected_value)`` when *Total operating expenses*
    can be derived deterministically as ``Cost of revenue + Operating expenses``.

    Returns ``(None, None)`` when the opex key is absent or either component
    is missing.  The caller is responsible for updating the metrics dict and
    emitting an ``"auto_corrected"`` Finding.
    """
    opex_key, _ = _find_metric(metrics, _OPEX_TOTAL_RE)
    if opex_key is None:
        return None, None
    _, cogs = _find_metric(metrics, _COGS_RE)
    _, opex_sub = _find_metric(metrics, _OPEX_SUBTOTAL_RE)
    if cogs is None or opex_sub is None:
        return None, None
    return opex_key, cogs + opex_sub


# ---------------------------------------------------------------------------
# Source grounding ("show me") verification
# ---------------------------------------------------------------------------

# Strip markdown table/emphasis artifacts so a snippet read from a rendered
# table chunk still matches the plain source text.
_GROUNDING_STRIP_RE = re.compile(r"[|*`]")
_GROUNDING_WS_RE = re.compile(r"\s+")
# Matches numeric tokens as printed in financial statements, e.g. "82,886",
# "(1,234)", "$4.27", "-12.5".
_GROUNDING_NUM_RE = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?")


def _normalize_for_grounding(text: str) -> str:
    """Whitespace/case-normalise text for verbatim substring comparison."""
    return _GROUNDING_WS_RE.sub(" ", _GROUNDING_STRIP_RE.sub(" ", text)).strip().casefold()


def _grounding_numeric_tokens(text: str) -> list[str]:
    """Extract comma-free digit tokens from *text* (e.g. "82,886" → "82886")."""
    tokens: list[str] = []
    for raw in _GROUNDING_NUM_RE.findall(text):
        digits = re.sub(r"[^\d.]", "", raw).strip(".")
        if digits and any(c.isdigit() for c in digits):
            tokens.append(digits)
    return tokens


def check_source_grounding(
    metrics: dict[str, Any],
    source_snippets: dict[str, Any] | None,
    source_text: str,
) -> list[Finding]:
    """Verify each extracted value is grounded in a verbatim source snippet.

    Implements the "show me" verification step: the extraction LLM returns, per
    metric, the exact text snippet it read the value from (``__sources__``). A
    value is considered grounded when **either** the cited snippet appears
    (whitespace/case-insensitively) in the source document **or** every numeric
    token in the snippet appears in the source. A value whose claimed source
    text *and* numbers are both absent from the document is almost certainly
    hallucinated and yields a ``high``-severity finding that triggers a
    re-extract.

    This checker is additive and degrades gracefully: when no snippets are
    supplied (e.g. an older prompt or a test mock) it returns no findings, so
    existing behaviour is unchanged.
    """
    findings: list[Finding] = []
    if not source_snippets or not source_text:
        return findings

    norm_corpus = _normalize_for_grounding(source_text)
    digit_corpus = norm_corpus.replace(",", "")

    for key, snippet in source_snippets.items():
        if key.startswith("__"):
            continue
        value = metrics.get(key)
        if value is None or not isinstance(snippet, str) or not snippet.strip():
            continue

        norm_snip = _normalize_for_grounding(snippet)
        if norm_snip and norm_snip in norm_corpus:
            continue  # cited snippet found verbatim

        nums = _grounding_numeric_tokens(snippet)
        if not nums:
            continue  # nothing numeric to verify — stay conservative
        if all(n in digit_corpus for n in nums):
            continue  # numbers present (snippet merely paraphrased)

        findings.append(
            Finding(
                type="source_unverified",
                severity="high",
                message=(
                    f"Value for '{key}' could not be verified against the source "
                    f"document — the cited snippet and its numbers are absent."
                ),
                keys=(key,),
                suggested_action=(
                    "Re-read the statement and report only values that appear "
                    "verbatim in the document; do not infer or compute values."
                ),
                evidence={"value": value, "cited_snippet": snippet[:200]},
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Checker registry
#
# The ordered list of every pure-observer checker.  ``analyze_metrics_node``
# iterates this list rather than hard-coding a call sequence, so adding a new
# checker only requires writing the function and appending it here.
#
# Rules for registry members:
#   - Signature: (metrics: dict[str, Any]) -> list[Finding]
#   - Must not mutate *metrics*.
#   - ``check_presence`` is NOT in this list because it requires the pre-computed
#     *presence* summary from ``critical_metrics.check_presence`` as a second
#     argument.  It is called explicitly before the registry loop.
#
# Correctors (``derive_corrected_total_opex``) are also excluded — they return
# ``(key, value)`` not ``list[Finding]`` and are handled in a separate post-pass
# inside the node (ADR-0003).
# ---------------------------------------------------------------------------
CHECKER_REGISTRY: list = [
    check_case_duplicates,
    check_composite_keys,
    check_gaap_nongaap_leakage,
    check_balance_sheet_identity,
    check_gross_profit_identity,
    check_operating_vs_gross,
    check_eps_dilution_ordering,
    check_sign_anomalies,
    check_suspect_round,
    check_opex_label_collision,
]
