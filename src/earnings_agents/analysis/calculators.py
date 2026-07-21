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

# ── Exclusion guard ───────────────────────────────────────────────────────────
# Labels that match any of these patterns should NEVER be assigned a P&L role,
# even if they incidentally contain a P&L keyword.
#
# Examples of false-positive matches without this guard:
#   • "Pre-tax restructuring charges"  →  would claim  pretax_income  role
#   • "OCI, Reclassification Adjustment … Included in Net Income, Net of Tax"
#                                       →  would claim  net_income  role
_ROLE_EXCLUSION_RX = _R(
    r"comprehensive\s+(?:income|loss)"
    r"|reclassification\s+adjust"
    r"|accumulated\s+other\s+comprehensive"
    r"|restructuring\s+charges?"
    r"|included\s+in\s+net\s+income"
    r"|other\s+comprehensive\s+income"
)

# ── Role patterns ─────────────────────────────────────────────────────────────
# List order matters: first match wins when a label could satisfy multiple roles.
# More-specific patterns are listed before broad catch-alls.
_ROLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("cost_of_revenue",      _R(r"cost\s+of\s+(?:revenue|sales|goods)")),
    ("gross_profit",         _R(r"gross\s+profit(?!\s*margin)")),
    ("rd_expense",           _R(r"research\s+(?:and|&)\s+development|r&d\s+expense"
                                r"|technology\s+(?:and\s+product\s+development|costs?|expense)")),
    ("sm_expense",           _R(r"(?:sales|selling)\s+(?:and|&)\s+marketing")),
    ("ga_expense",           _R(r"general\s*,?\s*(?:and|&)\s+administrative")),  # comma variant: "Selling, general, and administrative"
    ("total_opex",           _R(r"total\s+operating\s+(?:expenses?|costs?)|\boperating\s+(?:expenses?|costs?)\b")),
    ("operating_income",     _R(r"operating\s+(?:income|profit|loss)|income\s+from\s+operations")),
    ("interest_income",      _R(r"^interest\s+(?:and\s+other\s+)?income")),
    # Banking/fintech — net/total interest income (label does not start with
    # "interest" so the pattern above misses it).
    ("interest_income",      _R(r"(?:total|net)\s+interest\s+(?:income|revenue)")),
    ("interest_expense",     _R(r"interest\s+expense")),
    ("other_income_net",     _R(r"other\s+(?:income|expense)|non.?operating\s+income")),
    # Banking/fintech — noninterest income variants (also catch "non-interest").
    ("other_income_net",     _R(r"(?:total\s+)?non.?interest\s+(?:income|revenue)")),
    # Banking/fintech — noninterest expense = total operating expenses for banks.
    ("total_opex",           _R(r"(?:total\s+)?non.?interest\s+expense")),
    ("pretax_income",        _R(r"income\s+before\s+(?:income\s+)?tax|pre.?tax\s+income")),
    ("tax_expense",          _R(r"income\s+tax\s+(?:expense|provision)|provision\s+for\s+(?:income\s+)?tax|\bincome\s+taxes?\b")),
    # EPS patterns must appear before net_income — "Diluted net income per share"
    # contains "net income" but should map to eps_diluted, not net_income.
    ("eps_basic",            _R(r"basic.*(?:per\s+share|eps)|(?:per\s+share|eps).*basic|per\s+basic\s+share")),
    ("eps_diluted",          _R(r"diluted.*(?:per\s+share|eps)|(?:per\s+share|eps).*diluted|per\s+diluted\s+share")),
    ("net_income",           _R(r"net\s+(?:income|earnings|loss)")),
    ("shares_basic",         _R(r"(?:weighted.{0,20}average\s+)?basic\s+shares|shares.{0,25}basic|basic.{0,10}shares\s+outstanding")),
    ("shares_diluted",       _R(r"(?:weighted.{0,20}average\s+)?diluted\s+shares|shares.{0,25}diluted|diluted.{0,10}shares\s+outstanding")),
    ("gross_margin_pct",     _R(r"gross\s+(?:profit\s+)?margin\s*%?")),
    ("operating_margin_pct", _R(r"operating\s+(?:income\s+)?margin\s*%?")),
    ("net_margin_pct",       _R(r"net\s+(?:income\s+|profit\s+)?margin\s*%?")),
    # Broad revenue pattern last to avoid false matches on "Cost of revenue".
    ("revenue",              _R(r"(?:total\s+)?(?:net\s+)?(?:revenue|sales)")),
]


def _identify_role(label: str) -> str | None:
    """Return the first matching semantic role for *label*, or ``None``."""
    if _ROLE_EXCLUSION_RX.search(label):
        return None
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(label):
            return role
    return None


# ── Taxonomy-key role patterns ────────────────────────────────────────────────
# XBRL taxonomy keys encode the accounting concept in CamelCase (e.g.
# ── XBRL taxonomy-key → role lookup ─────────────────────────────────────────
# Exact match is always better than regex: the XBRL name IS the semantics.
# Priority: XBRL exact lookup  >  label regex patterns  >  LLM override.
#
# Every entry maps one canonical XBRL key to a derivation role.  Add new keys
# here when a new taxonomy variant is encountered; no regex needed.
#
# Notes:
# - EarningsPerShareBasicAndDiluted maps to eps_diluted so it acts as the
#   "diluted" operand for EPS derivation (the more conservative value).
# - RevenuesNetOfInterestExpense is SoFi's/banks' top-line revenue key.
# - Dimensional keys (containing "|") are excluded at the call site.
_XBRL_TO_ROLE: dict[str, str] = {
    # Revenue
    "us-gaap:Revenues":                                                        "revenue",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax":             "revenue",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax":             "revenue",
    "us-gaap:RevenuesNetOfInterestExpense":                                    "revenue",  # banks
    "us-gaap:SalesRevenueNet":                                                 "revenue",
    "us-gaap:SalesRevenueGoodsNet":                                            "revenue",
    # Cost of revenue
    "us-gaap:CostOfRevenue":                                                   "cost_of_revenue",
    "us-gaap:CostOfGoodsAndServicesSold":                                      "cost_of_revenue",
    "us-gaap:CostOfGoodsSold":                                                 "cost_of_revenue",
    # Gross profit
    "us-gaap:GrossProfit":                                                     "gross_profit",
    # R&D
    "us-gaap:ResearchAndDevelopmentExpense":                                   "rd_expense",
    "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost":     "rd_expense",
    # Sales & marketing
    "us-gaap:SellingAndMarketingExpense":                                      "sm_expense",
    "us-gaap:MarketingExpense":                                                "sm_expense",
    # G&A
    "us-gaap:GeneralAndAdministrativeExpense":                                 "ga_expense",
    # Total opex
    "us-gaap:OperatingExpenses":                                               "total_opex",
    "us-gaap:CostsAndExpenses":                                                "total_opex",
    "us-gaap:NoninterestExpense":                                              "total_opex",  # banks
    # Operating income
    "us-gaap:OperatingIncomeLoss":                                             "operating_income",
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "pretax_income",
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": "pretax_income",
    # Interest income / expense (banks)
    "us-gaap:InterestIncomeExpenseNet":                                        "interest_income",
    "us-gaap:InterestIncomeOperating":                                         "interest_income",
    "us-gaap:InterestAndDividendIncomeOperating":                              "interest_income",
    "us-gaap:InterestExpense":                                                 "interest_expense",
    "us-gaap:InterestExpenseOperating":                                        "interest_expense",
    # Non-interest / other income
    "us-gaap:NoninterestIncome":                                               "other_income_net",
    "us-gaap:OtherNonoperatingIncomeExpense":                                  "other_income_net",
    # Tax
    "us-gaap:IncomeTaxExpenseBenefit":                                         "tax_expense",
    # Net income
    "us-gaap:NetIncomeLoss":                                                   "net_income",
    "us-gaap:ProfitLoss":                                                      "net_income",
    "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic":                 "net_income",
    # EPS
    "us-gaap:EarningsPerShareBasic":                                           "eps_basic",
    "us-gaap:EarningsPerShareDiluted":                                         "eps_diluted",
    "us-gaap:EarningsPerShareBasicAndDiluted":                                 "eps_diluted",
    # Shares
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic":                   "shares_basic",
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding":                 "shares_diluted",
    "us-gaap:WeightedAverageNumberOfSharesOutstandingDiluted":                 "shares_diluted",
    # Margins
    "us-gaap:GrossProfit":                                                     "gross_profit",  # already above; explicit for clarity
}


def _role_from_xbrl(taxonomy_key: str) -> str | None:
    """Return the derivation role for an exact XBRL taxonomy key.

    Strips dimensional member suffixes (e.g. ``us-gaap:Revenues|aapl:Member``)
    before lookup.  Returns ``None`` for unknown or company-specific keys.
    """
    if not taxonomy_key:
        return None
    base = taxonomy_key.split("|")[0]
    return _XBRL_TO_ROLE.get(base)


def identify_role(label: str, taxonomy_key: str = "") -> str | None:
    """Return the semantic role for a concept, or ``None``.

    Lookup priority:
    1. Exact XBRL taxonomy-key match (``_XBRL_TO_ROLE``) — zero ambiguity,
       handles all standard US-GAAP variants explicitly.
    2. Label regex patterns (``_ROLE_PATTERNS``) — catches display-name
       conventions and company-specific labels.

    The LLM override (``role_overrides`` in ``derive_missing_concept_metrics``)
    is applied by the caller as a final fallback.
    """
    role = _role_from_xbrl(taxonomy_key)
    if role is not None:
        return role
    return _identify_role(label)


# All roles this module can derive or use as operands — exposed so callers can
# build constrained LLM prompts without duplicating the role list.
ALL_ROLES: frozenset[str] = frozenset(role for role, _ in _ROLE_PATTERNS)


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


def _f_total_opex_from_gp_oi(r: dict) -> float | None:
    """Total opex = Gross Profit − Operating Income (reverse derivation)."""
    gp = r.get("gross_profit")
    oi = r.get("operating_income")
    if gp is None or oi is None:
        return None
    return gp - oi


def _f_total_opex_from_items(r: dict) -> float | None:
    """Total opex = Σ individual line items (R&D + S&M + G&A)."""
    items = [r[k] for k in ("rd_expense", "sm_expense", "ga_expense") if r.get(k) is not None]
    if not items:
        return None
    return sum(items)


# ── Rules ─────────────────────────────────────────────────────────────────────
# (target_role, formula_fn) — processed in order so earlier derived values
# (e.g. gross_profit) are available as operands for later rules.
_RULES: list[tuple[str, Callable[[dict], float | None]]] = [
    ("gross_profit",         _f_gross_profit),
    ("operating_income",     _f_operating_income_via_total_opex),
    ("operating_income",     _f_operating_income_via_items),    # fallback
    # total_opex runs AFTER operating_income so it can use a known OI value
    # (either extracted from the filing or derived above).
    ("total_opex",           _f_total_opex_from_gp_oi),
    ("total_opex",           _f_total_opex_from_items),         # fallback
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


# ── Parent/child hierarchy calculation ──────────────────────────────────────
# Applied AFTER all extraction and derivation is complete.  Replaces extracted
# parent values with calculated sums when ALL direct children have values.


def _path_depth(path: str) -> int:
    """Return the number of dot-separated segments in *path*."""
    return path.count(".") + 1 if path else 0


def _is_direct_child(child_path: str, parent_path: str) -> bool:
    """Return True when *child_path* is exactly one level below *parent_path*."""
    if not child_path.startswith(parent_path + "."):
        return False
    return _path_depth(child_path) == _path_depth(parent_path) + 1


def _resolve_parent_child_hierarchy(
    all_concepts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build a mapping of parent concept_id → list of direct child concept dicts.

    A concept is a *parent* when at least one other concept's ``path`` is
    exactly one level below it (direct child).  Grandchildren (two levels
    below) are not included — only immediate children.
    """
    # Index concepts by path for fast lookup
    id_to_concept: dict[str, dict] = {c["_id"]: c for c in all_concepts}
    parent_to_children: dict[str, list[dict[str, Any]]] = {}

    for cid, concept in id_to_concept.items():
        parent_path = concept.get("path") or ""
        if not parent_path:
            continue
        for other_cid, other in id_to_concept.items():
            if other_cid == cid:
                continue
            other_path = other.get("path") or ""
            if _is_direct_child(other_path, parent_path):
                parent_to_children.setdefault(cid, []).append(other)

    return parent_to_children


def _find_value_for_role(
    concept_metrics: dict[str, float],
    all_concepts: list[dict[str, Any]],
    role: str,
    role_overrides: dict[str, str] | None = None,
) -> tuple[float | None, str | None]:
    """Return ``(value, concept_id)`` for the first mapped concept with *role*."""
    _overrides = role_overrides or {}
    for c in all_concepts:
        cid = c["_id"]
        if cid not in concept_metrics:
            continue
        c_role = identify_role(c.get("label", ""), c.get("taxonomy_key", "")) or _overrides.get(cid)
        if c_role == role:
            return concept_metrics[cid], cid
    return None, None


def _find_parent_concept_id(
    all_concepts: list[dict[str, Any]],
    role: str,
    parent_to_children: dict[str, list[dict[str, Any]]],
    role_overrides: dict[str, str] | None = None,
) -> str | None:
    """Return the concept_id of the first concept with *role* that HAS children."""
    _overrides = role_overrides or {}
    for c in all_concepts:
        cid = c["_id"]
        if cid not in parent_to_children:
            continue
        c_role = identify_role(c.get("label", ""), c.get("taxonomy_key", "")) or _overrides.get(cid)
        if c_role == role:
            return cid
    return None


def apply_parent_child_calculations(
    concept_metrics: dict[str, float],
    all_concepts: list[dict[str, Any]],
    role_overrides: dict[str, str] | None = None,
) -> dict[str, float]:
    """Apply parent/child calculation rules to *concept_metrics*.

    Rules (applied in dependency order):

    1. **Cost of Revenue** — if ALL direct children have values, replace the
       parent value with ``sum(children)`` and remove children from the output.
       Otherwise, keep the extracted parent value as-is.

    2. **Gross Profit** — if an extracted value exists, keep it.  Otherwise
       calculate ``Revenue − Cost of Revenue`` using the FINAL Cost of Revenue
       from step 1.

    3. **Operating Expenses** — if ALL direct children have values, replace
       the parent with ``sum(children)``.  Otherwise, keep the extracted parent
       value.  If no extracted parent value exists, calculate
       ``Gross Profit − Operating Income`` when both are available.

    4. **Operating Income** — always keep the extracted value as-is; never
       recalculate.

    Priority (for each parent):
      1. Sum of all children (every child must have a value)
      2. Extracted parent value
      3. Calculation (OpEx = GP − OI only, never a partial children sum)

    Only ONE final value is stored for each parent concept: either the
    calculated sum of children or the extracted value, never both.
    Children always remain in the output as individual line-item metrics.
    """
    if not concept_metrics or not all_concepts:
        return concept_metrics

    result = dict(concept_metrics)
    _overrides = role_overrides or {}
    parent_to_children = _resolve_parent_child_hierarchy(all_concepts)
    if not parent_to_children:
        return result

    # ── Step 1: Cost of Revenue ────────────────────────────────────────────
    _cor_parent_id = _find_parent_concept_id(
        all_concepts, "cost_of_revenue", parent_to_children, _overrides,
    )
    _cor_children: list[dict[str, Any]] = (
        parent_to_children.get(_cor_parent_id or "", []) if _cor_parent_id else []
    )
    _cor_all_children_present = False
    _cor_calculated: float | None = None

    if _cor_parent_id and _cor_children:
        _child_vals: list[float] = []
        _missing_children: list[str] = []
        for child in _cor_children:
            child_val = result.get(child["_id"])
            if isinstance(child_val, (int, float)):
                _child_vals.append(float(child_val))
            else:
                _missing_children.append(child.get("label", child["_id"]))

        if not _missing_children and _child_vals:
            # All children present → sum them, replace parent value.
            # Children remain in the output as individual metrics.
            _cor_calculated = sum(_child_vals)
            result[_cor_parent_id] = _cor_calculated
            _cor_all_children_present = True
            logger.info(
                "Cost of Revenue = sum(%d children) = %s",
                len(_child_vals), f"{_cor_calculated:,.0f}",
            )
        else:
            # Not all children present → keep extracted parent as-is
            _parent_val = result.get(_cor_parent_id)
            if isinstance(_parent_val, (int, float)):
                _cor_calculated = float(_parent_val)
                logger.info(
                    "Cost of Revenue = extracted parent = %s (missing %d/%d child(ren))",
                    f"{_cor_calculated:,.0f}", len(_missing_children), len(_cor_children),
                )

    # ── Step 2: Gross Profit ───────────────────────────────────────────────
    _gp_val: float | None = None
    _gp_parent_id = _find_parent_concept_id(
        all_concepts, "gross_profit", parent_to_children, _overrides,
    )
    if _gp_parent_id:
        _gp_extracted = result.get(_gp_parent_id)
        if isinstance(_gp_extracted, (int, float)):
            _gp_val = float(_gp_extracted)
        elif _cor_calculated is not None:
            # No extracted Gross Profit → calculate from Revenue − CoR
            _rev_val, _ = _find_value_for_role(
                result, all_concepts, "revenue", _overrides,
            )
            if _rev_val is not None:
                _gp_val = _rev_val - _cor_calculated
                result[_gp_parent_id] = _gp_val
                logger.info(
                    "Gross Profit = Revenue − Cost of Revenue = %s − %s = %s",
                    f"{_rev_val:,.0f}", f"{_cor_calculated:,.0f}", f"{_gp_val:,.0f}",
                )

    # ── Step 3: Operating Expenses ──────────────────────────────────────────
    _opex_parent_id = _find_parent_concept_id(
        all_concepts, "total_opex", parent_to_children, _overrides,
    )
    _opex_children: list[dict[str, Any]] = (
        parent_to_children.get(_opex_parent_id or "", []) if _opex_parent_id else []
    )

    if _opex_parent_id and _opex_children:
        _child_vals_opex: list[float] = []
        _missing_opex: list[str] = []
        for child in _opex_children:
            child_val = result.get(child["_id"])
            if isinstance(child_val, (int, float)):
                _child_vals_opex.append(float(child_val))
            else:
                _missing_opex.append(child.get("label", child["_id"]))

        if not _missing_opex and _child_vals_opex:
            # All children present → sum them, replace parent value.
            # Children remain in the output as individual metrics.
            _opex_calculated = sum(_child_vals_opex)
            result[_opex_parent_id] = _opex_calculated
            logger.info(
                "Operating Expenses = sum(%d children) = %s",
                len(_child_vals_opex), f"{_opex_calculated:,.0f}",
            )
        else:
            _parent_opex = result.get(_opex_parent_id)
            if isinstance(_parent_opex, (int, float)):
                # Keep extracted parent value
                logger.info(
                    "Operating Expenses = extracted parent = %s (missing %d/%d child(ren))",
                    f"{float(_parent_opex):,.0f}", len(_missing_opex), len(_opex_children),
                )
            elif _gp_val is not None:
                # No extracted parent → calculate GP − OI
                _oi_val, _ = _find_value_for_role(
                    result, all_concepts, "operating_income", _overrides,
                )
                if _oi_val is not None:
                    _opex_calculated = _gp_val - _oi_val
                    result[_opex_parent_id] = _opex_calculated
                    logger.info(
                        "Operating Expenses = Gross Profit − Operating Income = %s − %s = %s",
                        f"{_gp_val:,.0f}", f"{_oi_val:,.0f}", f"{_opex_calculated:,.0f}",
                    )

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def derive_missing_concept_metrics(
    concept_metrics: dict[str, float],
    all_concepts: list[dict[str, Any]],
    role_overrides: dict[str, str] | None = None,
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
    role_overrides:
        Optional ``{concept_id → role}`` mapping from LLM-based role
        identification for concepts whose labels did not match any regex
        pattern in ``_ROLE_PATTERNS``.  Used as a fallback when
        ``_identify_role(label)`` returns ``None``.

    Returns
    -------
    Updated ``concept_metrics`` with derived values inserted for any previously
    unmapped concept whose value can be computed.  The original dict is not
    mutated.
    """
    if not concept_metrics or not all_concepts:
        return concept_metrics

    result = dict(concept_metrics)

    # Build concept_id → label + taxonomy_key lookups for role identification.
    id_to_label: dict[str, str] = {c["_id"]: c.get("label", "") for c in all_concepts}
    id_to_tkey: dict[str, str] = {c["_id"]: c.get("taxonomy_key", "") for c in all_concepts}

    # Step 1 — populate role_values from already-mapped concepts.
    # Use both label patterns AND taxonomy-key patterns so banking/fintech
    # concepts with company-specific labels (e.g. sofi:InterestIncomeServicing)
    # are correctly identified without an LLM call.
    _overrides = role_overrides or {}
    role_values: dict[str, float] = {}
    for cid, value in result.items():
        if not isinstance(value, (int, float)):
            continue
        label = id_to_label.get(cid, "")
        tkey  = id_to_tkey.get(cid, "")
        role = identify_role(label, tkey) or _overrides.get(cid)
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
        # Find unmapped concepts that match this rule's target role — try regex
        # first, fall back to LLM-supplied role_overrides for unrecognized labels.
        targets = [
            c for c in unmapped
            if c["_id"] not in result
            and (
                identify_role(c.get("label", ""), c.get("taxonomy_key", ""))
                or _overrides.get(c["_id"])
            ) == target_role
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

    _audit_derivation_gaps(role_values, [c for c in unmapped if c["_id"] not in result], _overrides)
    return result


# ── Operand requirements per derivation rule ──────────────────────────────────
_RULE_OPERANDS: dict[str, list[str]] = {
    "gross_profit":         ["revenue", "cost_of_revenue"],
    "operating_income":     ["gross_profit", "total_opex"],  # primary path
    "total_opex":           ["gross_profit", "operating_income"],  # reverse derivation
    "pretax_income":        ["operating_income"],
    "net_income":           ["pretax_income", "tax_expense"],
    "eps_basic":            ["net_income", "shares_basic"],
    "eps_diluted":          ["net_income", "shares_diluted"],
    "gross_margin_pct":     ["gross_profit", "revenue"],
    "operating_margin_pct": ["operating_income", "revenue"],
    "net_margin_pct":       ["net_income", "revenue"],
}

_DERIVABLE_ROLES = frozenset(_RULE_OPERANDS)


def _audit_derivation_gaps(
    role_values: dict[str, float],
    still_unmapped: list[dict],
    role_overrides: dict[str, str] | None = None,
) -> None:
    """Log a warning for each unmapped concept whose derivation was blocked."""
    _overrides = role_overrides or {}
    for concept in still_unmapped:
        cid = concept.get("_id", "")
        label = concept.get("label", cid or "?")
        role = _identify_role(label) or _overrides.get(cid)
        if role not in _DERIVABLE_ROLES:
            continue
        missing = [op for op in _RULE_OPERANDS[role] if op not in role_values]
        if missing:
            logger.warning(
                "derive_missing_concept_metrics: could not derive '%s' (%s) — "
                "missing operand(s): %s",
                label, role, ", ".join(missing),
            )
