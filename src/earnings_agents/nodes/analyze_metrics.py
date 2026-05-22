"""``analyze_metrics`` — single point of post-extraction quality control.

Runs pure-Python checkers from ``earnings_agents.analysis`` and produces:

  * ``state["findings"]``  — list of ``Finding.to_dict()`` for downstream nodes
                              and persistence.
  * ``state["extraction_notes"]`` — structured hint string consumed by
                              ``extract_financial_metrics`` on re-extract.
  * ``state["status"]``    — flipped to ``"text_extracted"`` to trigger a
                              re-extract loop ONLY when a ``high``-severity
                              finding exists AND ``extraction_attempts``
                              remain below ``MAX_EXTRACTION_ATTEMPTS``.

This node *replaces* ``reflect_metrics`` in the graph. The old node remains
on disk for now (its ``MAX_EXTRACTION_ATTEMPTS`` constant is re-used).
"""
from __future__ import annotations

import logging
from typing import Any

from earnings_agents.analysis.critical_metrics import check_presence as presence_summary
from earnings_agents.analysis.findings import (
    Finding,
    check_balance_sheet_identity,
    check_case_duplicates,
    check_composite_keys,
    check_gaap_nongaap_leakage,
    check_presence,
    check_sign_anomalies,
    check_suspect_round,
)
from earnings_agents.nodes.reflect_metrics import MAX_EXTRACTION_ATTEMPTS
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)


def _build_extraction_notes(findings: list[Finding]) -> str:
    """Build a focused hint block for the next extraction pass.

    Only ``high`` and ``medium`` findings are surfaced — ``low`` findings
    (e.g. case duplicates) are handled deterministically downstream.
    """
    lines: list[str] = [
        "Previous extraction was incomplete. On this pass, prioritise finding:",
    ]
    high = [f for f in findings if f.severity == "high"]
    medium = [f for f in findings if f.severity == "medium"]
    for f in high:
        lines.append(f"  - [REQUIRED] {f.message}")
    if medium:
        lines.append("Also locate these if reported:")
        for f in medium:
            lines.append(f"  - {f.message}")
    lines.append(
        "Search the source text exhaustively (income statement, balance sheet, "
        "cash-flow statement, and supplementary tables). Preserve the exact "
        "wording each metric uses in the document."
    )
    return "\n".join(lines)


def analyze_metrics_node(state: EarningsAgentState) -> EarningsAgentState:
    """Run all deterministic analysis checkers and decide on re-extract."""
    if state.get("status") == "failed":
        return state

    metrics: dict[str, Any] = state.get("metrics") or {}
    ticker = state.get("ticker", "?")

    if not metrics:
        logger.warning("analyze_metrics: no metrics on state for %s", ticker)
        return state

    presence = presence_summary(metrics.keys())
    findings: list[Finding] = []
    findings.extend(check_presence(metrics, presence))
    findings.extend(check_case_duplicates(metrics))
    findings.extend(check_composite_keys(metrics))
    findings.extend(check_gaap_nongaap_leakage(metrics))
    findings.extend(check_balance_sheet_identity(metrics))
    findings.extend(check_sign_anomalies(metrics))
    findings.extend(check_suspect_round(metrics))

    # Log a compact summary.
    by_type: dict[str, int] = {}
    for f in findings:
        by_type[f.type] = by_type.get(f.type, 0) + 1
    logger.info(
        "analyze_metrics %s: tier1_missing=%d tier2_missing=%d tier3_present=%d findings=%s",
        ticker,
        len(presence["tier1_missing"]),
        len(presence["tier2_missing"]),
        len(presence["tier3_present"]),
        by_type or "{}",
    )

    out: dict[str, Any] = {
        **state,
        "findings": [f.to_dict() for f in findings],
    }

    high = [f for f in findings if f.severity == "high"]
    attempts = state.get("extraction_attempts", 0)
    if high and attempts < MAX_EXTRACTION_ATTEMPTS:
        out["extraction_notes"] = _build_extraction_notes(findings)
        out["status"] = "text_extracted"
        logger.info(
            "analyze_metrics %s: %d critical metric(s) missing — looping back "
            "(attempt %d/%d)",
            ticker, len(high), attempts + 1, MAX_EXTRACTION_ATTEMPTS,
        )
    elif high:
        logger.warning(
            "analyze_metrics %s: %d critical metric(s) still missing after "
            "%d attempts — proceeding to cleanup/save",
            ticker, len(high), attempts,
        )

    return out  # type: ignore[return-value]
