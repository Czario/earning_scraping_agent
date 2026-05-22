"""Structured ``Finding`` model + deterministic checkers.

A ``Finding`` is the unit of communication between the analysis node and
downstream consumers (``cleanup_metrics``, the re-extract loop). Each finding
carries enough metadata for a consumer to act without re-running heuristics.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

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
    "gaap_nongaap_leakage",   # key leaked from a GAAP/Non-GAAP reconciliation table
    "composite_key",          # key is a comma/slash list of synonyms, not a real label
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

_BS_ASSETS_RE = re.compile(r"^\s*total assets\b", re.IGNORECASE)
_BS_LIAB_RE = re.compile(r"^\s*total liabilities\s*$", re.IGNORECASE)
_BS_EQUITY_RE = re.compile(
    r"total\s+(stockholders'?|shareholders'?)\s*equity", re.IGNORECASE
)
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
_ROUND_UNIT = 100_000_000   # $100M
_ROUND_MIN = 100_000_000    # ignore tiny numbers
_ROUND_MAX = 100_000_000_000  # ignore megacaps where round totals are common

# Concept guards — never flag these as suspect-round even when the value
# happens to be an exact multiple.
_NEVER_ROUND_RE = re.compile(
    r"per share|eps|margin|ratio|rate|shares|weighted|%|percent",
    re.IGNORECASE,
)


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

# Keys matching any of these patterns came from a reconciliation or Non-GAAP
# table, not the primary GAAP income statement. They should be dropped.
_GAAP_NONGAAP_RE = re.compile(
    r"^(GAAP|Non-GAAP)\s+"         # prefixed with "GAAP " or "Non-GAAP "
    r"|non-gaap"                    # mid-key occurrence
    r"|\badjusted\b"               # "Adjusted operating income", etc.
    r"|\breconciliation\b"         # reconciliation table headers
    r"|impact of\b"                # "Total impact of non-GAAP adjustments"
    r"|\btax impact\b",
    re.IGNORECASE,
)


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

# A legitimate metric key never contains " / " or ", " between two recognisable
# financial concepts. When the LLM copies the synonym list from the prompt
# verbatim it produces keys like
# "Diluted earnings per share, Diluted EPS, Diluted net income per share"
# or "Cost of revenue / Cost of goods sold".
_COMPOSITE_KEY_RE = re.compile(
    # Comma- or slash-separated synonyms that each look like a metric label.
    # Heuristic: the key contains " / " or matches two comma-separated
    # phrase chunks that both contain finance-ish words.
    r"(?:diluted|basic|revenue|income|cost|expense|profit|earnings|shares|loss)"
    r".{1,60}"
    r"(?:,\s+|\s+/\s+)"
    r"(?:diluted|basic|revenue|income|cost|expense|profit|earnings|shares|loss|eps|net|per\s+share)",
    re.IGNORECASE,
)


def check_composite_keys(metrics: dict[str, Any]) -> list[Finding]:
    """Flag keys that are lists of synonyms rather than real document labels.

    These are produced when the LLM copies SCOPE instruction text verbatim
    as a key name (e.g. ``"Diluted earnings per share, Diluted EPS, ..."``).
    Severity is ``"low"``; cleanup drops them deterministically.
    """
    bad = [k for k in metrics if _COMPOSITE_KEY_RE.search(k)]
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
