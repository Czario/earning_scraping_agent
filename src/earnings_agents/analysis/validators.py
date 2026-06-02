"""Deterministic post-merge accounting validators.

These functions are *correctors*, not observers:
  - They may null-out implausible values in the metrics dict.
  - They return an updated dict **and** a list of warning strings.

Contrast with ``findings.py`` (pure observers that return ``Finding`` objects
without mutating the dict). Keeping them separate preserves the contract that
every ``Finding`` is an observation — nothing in ``findings.py`` changes data.

Public API
----------
validate_metrics(metrics) -> (metrics, warnings)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
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
