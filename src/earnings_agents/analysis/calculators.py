"""Deterministic income-statement derivation for calculated/system concepts.

After LLM extraction + two-tier concept mapping, some target concepts may
remain unmapped because their values do not appear verbatim in the filing —
they are *computed* metrics (``system:``-prefixed or ``calculated: True`` in
the normalize_data DB).  This module encodes the accounting identities that
derive those values from already-mapped metrics.

Usage
-----
    from earnings_agents.analysis.calculators import derive_missing_concept_metrics

    concept_metrics = derive_missing_concept_metrics(
        concept_metrics,   # {concept_id → float} already populated from filing
        all_concepts,      # combined list: target_concepts + calculated_concepts
    )

Derivation rules (applied in dependency order)
----------------------------------------------
1.  Gross Profit          = Revenue − Cost of Revenue
2.  Operating Income      = Gross Profit − Total Operating Expenses
    (fallback)            = Gross Profit − Σ(R&D + S&M + G&A)
3.  Pre-tax Income        = Operating Income
    (only when no below-the-line interest / other-income items are present —
    their sign is ambiguous in press releases)
4.  Net Income            = Pre-tax Income − Income Tax Expense
5.  EPS Basic             = Net Income / Weighted-Avg Basic Shares
6.  EPS Diluted           = Net Income / Weighted-Avg Diluted Shares
7.  Gross Margin %        = (Gross Profit  / Revenue) × 100
8.  Operating Margin %    = (Operating Income / Revenue) × 100
9.  Net Margin %          = (Net Income / Revenue) × 100

Safety gates
------------
* All required operands must be non-null numeric values.
* Result must be finite and not NaN.
* Margin percentages are accepted only in [−200, 200].
* EPS is accepted only in [−1 000, 1 000].
* Derivation never overwrites a value already present in concept_metrics.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

_R = lambda s: re.compile(s, re.IGNORECASE)  # noqa: E731

# ── Role patterns ─────────────────────────────────────────────────────────────
# List order matters: first match wins when a label could satisfy multiple roles.
# More-specific patterns are listed before broad catch-alls.
_ROLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("cost_of_revenue",      _R(r"cost\s+of\s+(?:revenue|sales|goods)")),
    ("gross_profit",         _R(r"gross\s+profit(?!\s*margin)")),
    ("rd_expense",           _R(r"research\s+(?:and|&)\s+development|r&d\s+expense")),
    ("sm_expense",           _R(r"(?:sales|selling)\s+(?:and|&)\s+marketing")),
    ("ga_expense",           _R(r"general\s+(?:and|&)\s+administrative")),
    ("total_opex",           _R(r"total\s+operating\s+(?:expenses?|costs?)")),
    ("operating_income",     _R(r"operating\s+(?:income|profit|loss)|income\s+from\s+operations")),
    ("interest_income",      _R(r"^interest\s+(?:and\s+other\s+)?income")),
    ("interest_expense",     _R(r"interest\s+expense")),
    ("other_income_net",     _R(r"other\s+(?:income|expense)|non.?operating\s+income")),
    ("pretax_income",        _R(r"income\s+before\s+(?:income\s+)?tax|pre.?tax")),
    ("tax_expense",          _R(r"income\s+tax\s+(?:expense|provision)|provision\s+for\s+(?:income\s+)?tax")),
    # EPS patterns must appear before net_income — "Diluted net income per share"
    # contains "net income" but should map to eps_diluted, not net_income.
    ("eps_basic",            _R(r"basic.*(?:per\s+share|eps)|(?:per\s+share|eps).*basic|per\s+basic\s+share")),
    ("eps_diluted",          _R(r"diluted.*(?:per\s+share|eps)|(?:per\s+share|eps).*diluted|per\s+diluted\s+share")),
    ("net_income",           _R(r"net\s+(?:income|earnings|loss)")),
    ("shares_basic",         _R(r"(?:weighted.{0,20}average\s+)?basic\s+shares|shares.{0,10}basic|basic.{0,10}shares\s+outstanding")),
    ("shares_diluted",       _R(r"(?:weighted.{0,20}average\s+)?diluted\s+shares|shares.{0,10}diluted|diluted.{0,10}shares\s+outstanding")),
    ("gross_margin_pct",     _R(r"gross\s+(?:profit\s+)?margin\s*%?")),
    ("operating_margin_pct", _R(r"operating\s+(?:income\s+)?margin\s*%?")),
    ("net_margin_pct",       _R(r"net\s+(?:income\s+|profit\s+)?margin\s*%?")),
    # Broad revenue pattern last to avoid false matches on "Cost of revenue".
    ("revenue",              _R(r"(?:total\s+)?(?:net\s+)?(?:revenue|sales)")),
]


def _identify_role(label: str) -> str | None:
    """Return the first matching semantic role for *label*, or ``None``."""
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(label):
            return role
    return None


# ── Formula functions ─────────────────────────────────────────────────────────
# Each accepts a ``role_values`` dict and returns float | None.

def _f_gross_profit(r: dict) -> float | None:
    rev = r.get("revenue")
    cogs = r.get("cost_of_revenue")
    if rev is None or cogs is None:
        return None
    return rev - cogs


def _f_operating_income_via_total_opex(r: dict) -> float | None:
    gp = r.get("gross_profit")
    opex = r.get("total_opex")
    if gp is None or opex is None:
        return None
    return gp - opex


def _f_operating_income_via_items(r: dict) -> float | None:
    """Fallback: GP − Σ individual opex line items (when total_opex absent)."""
    gp = r.get("gross_profit")
    if gp is None:
        return None
    items = [r[k] for k in ("rd_expense", "sm_expense", "ga_expense") if r.get(k) is not None]
    if not items:
        return None
    return gp - sum(items)


def _f_pretax_income(r: dict) -> float | None:
    """Pre-tax ≈ Operating Income only when no below-the-line items are known.

    If interest income, interest expense, or other income (net) are mapped,
    skip derivation: their sign conventions in press releases are ambiguous and
    mixing them with operating income risks a wrong result.
    """
    if any(r.get(k) is not None for k in ("interest_income", "interest_expense", "other_income_net")):
        return None
    return r.get("operating_income")


def _f_net_income(r: dict) -> float | None:
    pt = r.get("pretax_income")
    tax = r.get("tax_expense")
    if pt is None or tax is None:
        return None
    return pt - tax


def _f_eps_basic(r: dict) -> float | None:
    ni = r.get("net_income")
    shares = r.get("shares_basic")
    if ni is None or not shares or shares <= 0:
        return None
    eps = ni / shares
    return eps if -1_000.0 <= eps <= 1_000.0 else None


def _f_eps_diluted(r: dict) -> float | None:
    ni = r.get("net_income")
    shares = r.get("shares_diluted")
    if ni is None or not shares or shares <= 0:
        return None
    eps = ni / shares
    return eps if -1_000.0 <= eps <= 1_000.0 else None


def _f_gross_margin_pct(r: dict) -> float | None:
    gp = r.get("gross_profit")
    rev = r.get("revenue")
    if gp is None or not rev:
        return None
    return (gp / rev) * 100.0


def _f_operating_margin_pct(r: dict) -> float | None:
    oi = r.get("operating_income")
    rev = r.get("revenue")
    if oi is None or not rev:
        return None
    return (oi / rev) * 100.0


def _f_net_margin_pct(r: dict) -> float | None:
    ni = r.get("net_income")
    rev = r.get("revenue")
    if ni is None or not rev:
        return None
    return (ni / rev) * 100.0


# ── Rules ─────────────────────────────────────────────────────────────────────
# (target_role, formula_fn) — processed in order so earlier derived values
# (e.g. gross_profit) are available as operands for later rules.
_RULES: list[tuple[str, Callable[[dict], float | None]]] = [
    ("gross_profit",         _f_gross_profit),
    ("operating_income",     _f_operating_income_via_total_opex),
    ("operating_income",     _f_operating_income_via_items),    # fallback
    ("pretax_income",        _f_pretax_income),
    ("net_income",           _f_net_income),
    ("eps_basic",            _f_eps_basic),
    ("eps_diluted",          _f_eps_diluted),
    ("gross_margin_pct",     _f_gross_margin_pct),
    ("operating_margin_pct", _f_operating_margin_pct),
    ("net_margin_pct",       _f_net_margin_pct),
]

_PCT_ROLES = frozenset({"gross_margin_pct", "operating_margin_pct", "net_margin_pct"})


def _is_valid(value: float | None, role: str) -> bool:
    if value is None or not isinstance(value, (int, float)):
        return False
    if math.isnan(value) or math.isinf(value):
        return False
    if role in _PCT_ROLES and not (-200.0 <= value <= 200.0):
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def derive_missing_concept_metrics(
    concept_metrics: dict[str, float],
    all_concepts: list[dict[str, Any]],
) -> dict[str, float]:
    """Fill in calculated concept values from already-mapped metrics.

    Parameters
    ----------
    concept_metrics:
        Current ``{concept_id → float}`` mapping produced by the two-tier
        extraction from the filing.  Must not be ``None``; may be empty.
    all_concepts:
        Combined list of regular + calculated concept dicts.  Each dict must
        have ``_id`` (str) and ``label`` (str) fields.

    Returns
    -------
    Updated ``concept_metrics`` with derived values inserted for any previously
    unmapped concept whose value can be computed.  The original dict is not
    mutated.
    """
    if not concept_metrics or not all_concepts:
        return concept_metrics

    result = dict(concept_metrics)

    # Build concept_id → label lookup for role identification.
    id_to_label: dict[str, str] = {c["_id"]: c.get("label", "") for c in all_concepts}

    # Step 1 — populate role_values from already-mapped concepts.
    role_values: dict[str, float] = {}
    for cid, value in result.items():
        if not isinstance(value, (int, float)):
            continue
        label = id_to_label.get(cid, "")
        role = _identify_role(label)
        if role and role not in role_values:
            role_values[role] = value

    # Step 2 — identify unmapped concepts (not yet in concept_metrics).
    unmapped = [c for c in all_concepts if c["_id"] not in result]
    if not unmapped:
        return result

    logger.debug(
        "derive_missing_concept_metrics: %d unmapped concept(s) — attempting derivation",
        len(unmapped),
    )

    # Step 3 — apply rules in order; new derived values enrich role_values for
    # subsequent rules (enabling chained derivation: GP → OI → NI).
    for target_role, formula in _RULES:
        # Find unmapped concepts that match this rule's target role.
        targets = [
            c for c in unmapped
            if c["_id"] not in result and _identify_role(c.get("label", "")) == target_role
        ]
        if not targets:
            continue

        # If the role value is already known (from filing or an earlier rule),
        # just copy it to any remaining unmapped concept that carries that role.
        if target_role in role_values:
            for concept in targets:
                result[concept["_id"]] = role_values[target_role]
                logger.info(
                    "derive_missing_concept_metrics: filled '%s' (%s) = %s "
                    "from existing role '%s'",
                    concept.get("label", ""), concept["_id"],
                    role_values[target_role], target_role,
                )
            continue

        # Try to compute the value for this role.
        computed = formula(role_values)
        if not _is_valid(computed, target_role):
            continue

        # Store the computed value and make it available to later rules.
        role_values[target_role] = computed  # type: ignore[assignment]
        for concept in targets:
            result[concept["_id"]] = computed  # type: ignore[assignment]
            logger.info(
                "derive_missing_concept_metrics: derived '%s' (%s) = %s via rule '%s'",
                concept.get("label", ""), concept["_id"], computed, target_role,
            )

    return result
