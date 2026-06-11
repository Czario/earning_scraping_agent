"""``analyze_metrics`` — single point of post-extraction quality control.

Runs pure-Python checkers from ``earnings_agents.analysis`` and produces:

  * ``state["findings"]``  — list of ``Finding.to_dict()`` for downstream nodes
                              and persistence.
  * ``state["extraction_notes"]`` — structured hint string consumed by
                              ``extract_financial_metrics`` on re-extract.
  * ``state["needs_reextract"]`` — True when a high-severity finding exists
                              AND ``extraction_attempts`` remain below
                              ``MAX_EXTRACTION_ATTEMPTS`` AND new findings
                              differ from the previous pass (progress detected).

This node *replaces* ``reflect_metrics`` in the graph.
"""
from __future__ import annotations

import logging
from typing import Any

from earnings_agents.analysis.critical_metrics import check_presence as presence_summary
from earnings_agents.analysis.findings import (
    CHECKER_REGISTRY,
    Finding,
    derive_corrected_total_opex,
    check_presence,
    check_source_grounding,
)
from earnings_agents.config import MAX_EXTRACTION_ATTEMPTS
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)


def _build_extraction_notes(
    findings: list[Finding],
    attempt_num: int,
    prev_notes: str,
) -> str:
    """Build a focused hint block for the next extraction pass.

    Only ``high`` and ``medium`` findings are surfaced — ``low`` findings
    (e.g. case duplicates) are handled deterministically downstream.

    The previous pass's notes are prepended so the model has cumulative
    context on what has already been attempted and still failed.
    """
    high = [f for f in findings if f.severity == "high"]
    medium = [f for f in findings if f.severity == "medium"]

    lines: list[str] = []
    if prev_notes and attempt_num > 1:
        lines.append(
            f"[Attempt {attempt_num - 1} history] {prev_notes.strip()}"
        )
        lines.append("")

    lines.append(
        f"Attempt {attempt_num} still incomplete. Prioritise finding these metrics "
        f"(search income statement, balance sheet, cash-flow, and supplementary tables):"
    )
    for f in high:
        lines.append(f"  - [REQUIRED] {f.message}")
        if f.suggested_action:
            lines.append(f"      → {f.suggested_action}")
        # For income-statement identity violations, tell the LLM exactly
        # which value it got wrong and what the implied correct value is.
        # This turns a vague "identity broken" message into a concrete hint:
        # "look for a Cost of revenue row close to X, not Y".
        if f.type == "identity_violation":
            ev = f.evidence or {}
            rev = ev.get("revenue")
            cor = ev.get("cost_of_revenue")
            gp  = ev.get("gross_profit")
            if (
                isinstance(rev, (int, float)) and rev
                and isinstance(gp, (int, float))
            ):
                implied_cor = rev - gp
                lines.append(f"      DIAGNOSIS — your previous extraction returned:")
                lines.append(f"        Revenue         = {rev:>22,.0f}  (reference)")
                if isinstance(cor, (int, float)):
                    lines.append(
                        f"        Cost of revenue = {cor:>22,.0f}  ← WRONG — taken from wrong column/row"
                    )
                lines.append(f"        Gross profit    = {gp:>22,.0f}  (reference)")
                lines.append(
                    f"      Revenue − Gross profit = {rev:,.0f} − {gp:,.0f} = {implied_cor:,.0f}"
                )
                lines.append(
                    f"      That means Cost of revenue in the current-period column should be"
                )
                lines.append(
                    f"      approximately {implied_cor:,.0f} — locate that row and use it."
                )
    if medium:
        lines.append("Also locate these if reported:")
        for f in medium:
            lines.append(f"  - {f.message}")
    lines.append(
        "Preserve the exact wording each metric uses in the document."
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
        return {**state, "needs_reextract": False}

    presence = presence_summary(metrics.keys())
    findings: list[Finding] = []
    findings.extend(check_presence(metrics, presence))

    # Registry loop — pure observers only (ADR-0003). check_presence is called
    # separately above because it requires a pre-computed presence summary.
    for checker in CHECKER_REGISTRY:
        findings.extend(checker(metrics))

    # Source-grounding ("show me") verification. Called explicitly (not via the
    # registry) because it needs the per-metric source snippets and the source
    # text in addition to the metrics dict. Degrades to a no-op when no
    # snippets were captured.
    findings.extend(
        check_source_grounding(
            metrics,
            state.get("metric_source_snippets"),
            state.get("raw_text") or "",
        )
    )

    # Corrector post-pass — kept explicitly separate from the observer loop so
    # it is easy to audit: only derive_corrected_total_opex mutates metrics.
    opex_collision = any(
        f.type == "suspect_value"
        and any("total operating expenses" in k.lower() for k in (f.keys or ()))
        for f in findings
    )
    if opex_collision:
        corrected_key, corrected_value = derive_corrected_total_opex(metrics)
        if corrected_key is not None and corrected_value is not None:
            old_value = metrics[corrected_key]
            metrics = {**metrics, corrected_key: corrected_value}
            findings.append(
                Finding(
                    type="auto_corrected",
                    severity="low",
                    message=(
                        f"Auto-corrected '{corrected_key}' from {old_value:,.0f} "
                        f"to {corrected_value:,.0f} "
                        f"(Cost of revenue + Operating expenses)."
                    ),
                    keys=(corrected_key,),
                    evidence={
                        "old_value": old_value,
                        "corrected_value": corrected_value,
                        "method": "Cost_of_revenue + Operating_expenses",
                    },
                )
            )
            logger.info(
                "analyze_metrics %s: auto-corrected '%s' %s → %s",
                ticker, corrected_key, old_value, corrected_value,
            )

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
        "metrics": metrics,
        "findings": [f.to_dict() for f in findings],
        "needs_reextract": False,
    }

    # In targeted mode (normalize_data) the truth set is ``target_concepts``,
    # not the hardcoded TIER1 registry.  Looping based on TIER1 misses is
    # futile when the company simply doesn't report that metric (e.g. BJ
    # Wholesale Club never reports a standalone Gross Profit line) AND when
    # the targeted prompt was only asked about a subset of statements.
    # Demote ``missing_critical`` findings to medium severity so they no
    # longer trigger re-extract; keep them in ``findings`` for visibility.
    if state.get("target_concepts"):
        from dataclasses import replace as _dc_replace
        findings = [
            _dc_replace(f, severity="medium")
            if f.severity == "high" and f.type == "missing_critical"
            else f
            for f in findings
        ]
        out["findings"] = [f.to_dict() for f in findings]

    high = [f for f in findings if f.severity == "high"]
    attempts = state.get("extraction_attempts", 0)

    if not high or attempts >= MAX_EXTRACTION_ATTEMPTS:
        if high:
            logger.warning(
                "analyze_metrics %s: %d critical metric(s) still missing after "
                "%d attempt(s) — proceeding to cleanup/save",
                ticker, len(high), attempts,
            )
        return out  # type: ignore[return-value]

    # No-progress detection: if the set of high-severity messages is identical
    # to the previous pass, the LLM gained nothing — break the loop early.
    current_high_keys = sorted(f.message for f in high)
    prev_high_keys: list[str] = state.get("previous_high_finding_keys") or []
    if attempts > 0 and current_high_keys == prev_high_keys:
        logger.warning(
            "analyze_metrics %s: same %d high finding(s) as previous pass — "
            "no progress detected, skipping re-extract",
            ticker, len(high),
        )
        return out  # type: ignore[return-value]

    # Loop back: build cumulative extraction notes.
    prev_notes = state.get("extraction_notes") or ""
    out["extraction_notes"] = _build_extraction_notes(findings, attempts + 1, prev_notes)
    out["needs_reextract"] = True
    out["previous_high_finding_keys"] = current_high_keys
    logger.info(
        "analyze_metrics %s: %d critical metric(s) missing — looping back "
        "(attempt %d/%d)",
        ticker, len(high), attempts + 1, MAX_EXTRACTION_ATTEMPTS,
    )
    return out  # type: ignore[return-value]

