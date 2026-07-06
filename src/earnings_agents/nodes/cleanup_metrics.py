"""Cleanup node — drops duplicate / scale-broken metric keys before save.

A single **constrained-LLM pass** identifies keys to remove. Three
deterministic guardrails make it physically impossible to corrupt data:

  a. The returned ``remove`` set must be a SUBSET of the original keys.
  b. Every surviving key keeps its original value byte-for-byte.
  c. Identity / sanity warnings are re-computed on the cleaned dict;
     if a new blocking warning appears, the cleanup is rejected.

If the LLM call fails or returns malformed JSON the node logs a warning
and returns the metrics unchanged. The cleanup pass cannot corrupt
data — at worst it does nothing.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from earnings_agents.config import CLEANUP_METRICS, LLM_PROVIDER as _LLM_PROVIDER
from earnings_agents.llm_factory import build_llm
from earnings_agents.analysis.validators import validate_metrics
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Token-efficient encoding for the LLM payload.
#
# Raw integers like 694228000000 tokenize per-digit (~7 tokens each); across
# 80 metrics that costs ~600 extra tokens per cleanup call. Compact forms like
# "694B" tokenize as 2 tokens and still carry enough information for the LLM
# to spot Rule A (identical value) and Rule B (impossible scale).
# -----------------------------------------------------------------------------

_PER_SHARE_RX = re.compile(r"\b(per\s*share|eps)\b", re.IGNORECASE)


def _compact_value(v: Any) -> Any:
    """Return a short token-efficient string for large numbers; pass through small ones."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    av = abs(v)
    if av >= 1_000_000_000:
        return f"{v / 1_000_000_000:.3g}B"
    if av >= 1_000_000:
        return f"{v / 1_000_000:.3g}M"
    if av >= 1_000:
        return f"{v / 1_000:.3g}K"
    return v  # small floats (EPS, percentages) stay numeric


def _build_compact_metrics(metrics: dict) -> dict:
    """Build a token-efficient mirror of metrics for the LLM cleanup prompt."""
    return {k: _compact_value(v) for k, v in metrics.items()}


def needs_cleanup(metrics: dict, protected_keys: set[str] | None = None) -> bool:
    """Return True when *metrics* contains any plausible cleanup candidate.

    A cheap Python pre-check that avoids an LLM call on already-clean dicts.
    Heuristics (any one triggers a call):

      - Two keys whose numeric values are equal within 0.1 %.
      - Two keys that differ only by case.
      - Any ``per share`` / ``EPS`` value above $10,000 (scale Rule B).

    ``protected_keys`` is the set of concept-mapped metric keys (from
    ``state["mapped_metric_keys"]``).  The cleanup guardrail blocks removal of
    any protected key, so if ALL candidates in a heuristic involve only
    protected keys the LLM call can never produce a removal — skip it.
    """
    _protected = protected_keys or set()
    numeric_items = [
        (k, float(v)) for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]

    # Rule B candidate: implausible per-share value.
    for k, v in numeric_items:
        if _PER_SHARE_RX.search(k) and abs(v) > 10_000:
            if k not in _protected:          # protected key → guardrail blocks → skip
                return True

    # Case-only duplicates ("Net income" vs "Net Income").
    lower_keys: dict[str, list[str]] = {}
    for k in metrics:
        lower_keys.setdefault(k.lower(), []).append(k)
    for variants in lower_keys.values():
        if len(variants) > 1:
            # If every variant is protected the guardrail blocks all removals → skip.
            if not all(v in _protected for v in variants):
                return True

    # Rule A candidate: any two non-zero values within 0.1%.
    # Skip pairs where BOTH keys are protected — the guardrail would block any
    # removal from such a pair, so the LLM call can never act on them.
    nonzero = [(k, v) for k, v in numeric_items if v != 0]
    nonzero.sort(key=lambda kv: kv[1])
    for i in range(1, len(nonzero)):
        k_a, a = nonzero[i - 1]
        k_b, b = nonzero[i]
        if abs(b - a) <= 0.001 * max(abs(a), abs(b)):
            if _protected and k_a in _protected and k_b in _protected:
                continue  # both protected → guardrail blocks → not actionable
            return True

    return False


# ---------------------------------------------------------------------------
# Public guardrail helpers
#
# Named so they can be imported and unit-tested independently.  The node uses
# them sequentially; each returning False causes the LLM pass to be rejected.
# They mirror the three contracts stated in the module docstring (a, b, c).
# ---------------------------------------------------------------------------

def guardrail_keys_are_subset(remove: list[str], metrics: dict) -> bool:
    """Guardrail a: every proposed removal key exists in *metrics*."""
    return all(k in metrics for k in remove)


def guardrail_values_unchanged(cleaned: dict, original: dict) -> bool:
    """Guardrail b: every surviving value is byte-identical to the original."""
    return all(_values_equal(v, original[k]) for k, v in cleaned.items())


def guardrail_no_new_warnings(
    cleaned: dict, before_warnings: list[str]
) -> tuple[bool, list[str]]:
    """Guardrail c: cleanup must not introduce new identity/sanity warnings.

    Returns ``(passes, after_warnings)`` where *after_warnings* is the full
    list from ``validate_metrics`` (used to update ``identity_warnings`` in
    state when the pass is accepted).
    """
    _, after_warnings = validate_metrics(cleaned)
    new_failures = [w for w in after_warnings if w not in before_warnings]
    return not new_failures, after_warnings


def _build_prompt(metrics: dict) -> str:
    # Use a token-efficient encoding: large numbers become "82.9B" / "315M" / "24.3K".
    # The LLM proposes removals using the ORIGINAL keys (unchanged); only the
    # value column is compacted to save tokens.
    metrics_json = json.dumps(_build_compact_metrics(metrics), indent=2, default=str)
    return f"""You are a deterministic financial-data cleanup assistant. Your only job is to identify metric keys that should be REMOVED from the JSON object below.

NOTE on value formatting: large numeric values are shown in compact form — "82.9B" means 82,900,000,000; "315M" means 315,000,000; "24.3K" means 24,300. Small floats (EPS, percentages, ratios) are shown as-is. Compare magnitudes accordingly.

You may ONLY return keys to remove. You CANNOT rename keys, change values, or add keys. Violations are discarded.

Remove a key only when one of these is clearly true:

RULE A — IDENTICAL-VALUE DUPLICATE. Two keys refer to the same number and have the same value within 0.1%. Examples (drop the second in each pair):
  • "Diluted EPS" : 2.39  AND  "Diluted earnings per share" : 2.39   → drop the longer one
  • "Diluted EPS" : 2.39  AND  "Diluted net income per share GAAP" : 2.39   → drop the longer one
  • "Weighted average shares used in per share computation (Diluted)" : "24.4B"  AND  "Weighted average shares used in diluted net income per share computation" : "24.4B"   → drop the longer one
  KEEP both when values differ (GAAP vs Non-GAAP gap, before/after tax, etc.).
  CRITICAL: NEVER drop a key just because its value sums with siblings to equal another key.
  Sub-components are NOT duplicates of their totals, even though they "add up":
    • "Revenue: Product" : "15.1B"  AND  "Revenue: Service and other" : "67.8B"  AND  "Total revenue" : "82.9B"   → KEEP ALL THREE. The product / service breakdown is distinct data.
    • "Cost of revenue: Product" : "2.73B"  AND  "Cost of revenue: Service" : "24.1B"  AND  "Total cost of revenue" : "26.8B"   → KEEP ALL THREE.
  Rule A applies ONLY when two keys hold the SAME number with the same meaning.

RULE B — SCALE ERROR. The value is impossible for the concept named. Examples (drop these):
  • "Gross margin" : "749M"   → that's a percentage (74.9%) mis-scaled by 1M
  • "Operating margin" : "34B"   → that's a percentage mis-scaled
  • A "per share" value above $10,000

RULE C — MALFORMED. Plainly a parse artifact. Examples (drop these):
  • A label-only fragment with no sensible numeric meaning.
  • "Proceeds related to employee stock plans" : 515  when every other cash-flow item is in the hundreds of millions or billions — 515 is a mis-scaled value (the table said "In millions"; the LLM failed to multiply). Drop it; a wrong value is worse than a missing one.
  • "Payments related to employee stock plan taxes" : -2129  — same issue; -2129 in a billions-scale statement is plainly "$-2,129" not "$-2,129,000,000".

Heuristic for choosing which of two duplicates to drop:
  1. The LONGER key.
  2. The one with a redundant qualifier ("GAAP", "Non-GAAP", "net income per share", "used in ... computation", etc.).

Do NOT remove a key just because you don't recognize it. When in doubt, KEEP.

Metrics to clean:
{metrics_json}

Respond with a SINGLE JSON object, no other text, no markdown fences:

{{"remove": ["<key>", "<key>", ...], "reasons": {{"<key>": "<one-line reason citing rule A/B/C>", ...}}}}

If nothing should be removed, return {{"remove": [], "reasons": {{}}}}.
"""


def _parse_response(response: str) -> dict[str, Any] | None:
    """Extract the first JSON object from the LLM response."""
    if not response:
        return None
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("Cleanup LLM returned invalid JSON: %s", exc)
        return None


def _values_equal(a: Any, b: Any) -> bool:
    """Strict equality for cleanup guardrails."""
    if type(a) is not type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
        return False
    return a == b


def _preferred_variant(variants: list[str], metric_order: list[str]) -> str:
    """Pick which case-duplicate variant survives deterministic dedup.

    Preference order:
      1. The variant that appears FIRST in the original metrics dict
         (preserves source ordering, stable across runs).
    Tie-breaking on identical insertion order is impossible, since dicts
    preserve insertion order in Python 3.7+.
    """
    return min(variants, key=metric_order.index)


def _apply_case_duplicate_findings(
    metrics: dict[str, Any],
    findings: list[dict[str, Any]],
    ticker: str,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically drop case-duplicate variants flagged by analyze_metrics.

    Returns (cleaned_metrics, removed_keys). When no case_duplicate findings
    are present, returns (metrics, []) unchanged.
    """
    case_dups = [f for f in findings if f.get("type") == "case_duplicate"]
    if not case_dups:
        return metrics, []

    order = list(metrics.keys())
    removed: list[str] = []
    cleaned = dict(metrics)
    for f in case_dups:
        variants = [k for k in f.get("keys", []) if k in cleaned]
        if len(variants) < 2:
            continue
        keep = _preferred_variant(variants, order)
        for v in variants:
            if v != keep:
                cleaned.pop(v, None)
                removed.append(v)
    if removed:
        logger.info(
            "Cleanup (deterministic) for %s dropped %d case-duplicate key(s): %s",
            ticker, len(removed), removed,
        )
    return cleaned, removed


def _apply_gaap_nongaap_findings(
    metrics: dict[str, Any],
    findings: list[dict[str, Any]],
    ticker: str,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically drop keys flagged as GAAP/Non-GAAP leakage or composite keys.

    Handles both ``gaap_nongaap_leakage`` and ``composite_key`` finding types —
    both indicate keys that should be removed rather than re-extracted.
    Returns (cleaned_metrics, removed_keys).
    """
    drop_types = {"gaap_nongaap_leakage", "composite_key"}
    relevant = [f for f in findings if f.get("type") in drop_types]
    if not relevant:
        return metrics, []

    to_drop: list[str] = []
    for f in relevant:
        to_drop.extend(k for k in f.get("keys", []) if k in metrics)

    if not to_drop:
        return metrics, []

    cleaned = {k: v for k, v in metrics.items() if k not in to_drop}
    logger.info(
        "Cleanup (deterministic) for %s dropped %d GAAP/Non-GAAP leakage and composite key(s): %s",
        ticker, len(to_drop), to_drop,
    )
    return cleaned, to_drop


def cleanup_metrics_node(state: EarningsAgentState) -> EarningsAgentState:
    """LLM-only cleanup: propose keys to drop, validate with three guardrails."""
    if not CLEANUP_METRICS:
        return state

    if state.get("status") == "failed":
        return state

    metrics = state.get("metrics") or {}
    if not metrics:
        return state

    ticker = state.get("ticker", "?")
    removed: list[str] = []

    # Step 0a — deterministic case-dedup driven by analyze_metrics findings.
    findings = state.get("findings") or []
    metrics, det_removed = _apply_case_duplicate_findings(metrics, findings, ticker)
    removed.extend(det_removed)

    # Step 0b — deterministic GAAP/Non-GAAP leakage + composite-key removal.
    metrics, leak_removed = _apply_gaap_nongaap_findings(metrics, findings, ticker)
    removed.extend(leak_removed)

    # Fix 2: skip the LLM call entirely when there are no plausible cleanup
    # candidates (no near-equal values, no case dupes, no implausible per-share).
    # Protected (concept-mapped) keys are passed so pairs/candidates that the
    # guardrail would block anyway don't needlessly trigger the LLM call.
    protected_keys = set(state.get("mapped_metric_keys") or [])
    if not needs_cleanup(metrics, protected_keys):
        logger.info(
            "Cleanup skipped for %s — no duplicate / scale candidates detected", ticker,
        )
        return _finalize(state, metrics, removed)

    prompt = _build_prompt(metrics)
    try:
        from earnings_agents.hooks import report_call
        report_call(f"  [llm]  cleanup  → calling llm  ({_LLM_PROVIDER or 'llm'})")
        llm = build_llm(format_json=True, request_timeout=60)
        response: str = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.warning("Cleanup LLM call failed for %s: %s — keeping metrics", ticker, exc)
        report_call(f"  cleanup  ✗  {exc}")
        return _finalize(state, metrics, [])

    parsed = _parse_response(response)
    if parsed is None:
        logger.warning("Cleanup parse failed for %s — keeping current metrics", ticker)
        return _finalize(state, metrics, removed)

    remove = parsed.get("remove") or []
    reasons = parsed.get("reasons") or {}
    if not isinstance(remove, list):
        logger.warning("Cleanup 'remove' is not a list for %s — keeping current metrics", ticker)
        return _finalize(state, metrics, removed)

    # Guardrail a: every requested removal must reference an existing key.
    unknown = [k for k in remove if k not in metrics]
    if unknown:
        logger.warning(
            "Cleanup LLM proposed removal of unknown keys for %s: %s — skipping LLM pass",
            ticker, unknown,
        )
        return _finalize(state, metrics, removed)

    # Guardrail 1b: every removal must come with a non-empty reason.
    # Filters out lazy "no reason" deletions that the LLM can't justify.
    unjustified = [k for k in remove if not str(reasons.get(k, "")).strip()]
    if unjustified:
        logger.warning(
            "Cleanup LLM proposed unjustified removals for %s: %s — ignoring those",
            ticker, unjustified,
        )
        remove = [k for k in remove if k not in unjustified]

    # Guardrail 1c: never remove a key that was confirmed mapped to a concept_id.
    # These keys are verified real document metrics (Tier 1 label match or Tier 2
    # LLM semantic match against the company's XBRL taxonomy).  The cleanup LLM
    # has no access to that mapping and cannot be trusted to override it.
    # (protected_keys already computed above for the needs_cleanup() pre-check)
    if protected_keys:
        concept_blocked = [k for k in remove if k in protected_keys]
        if concept_blocked:
            logger.info(
                "Cleanup guardrail: blocked removal of %d concept-mapped key(s) for %s: %s",
                len(concept_blocked), ticker, concept_blocked,
            )
            report_call(
                f"  [guardrail]  blocked {len(concept_blocked)} concept-mapped removal(s)"
                f"  — {', '.join(concept_blocked)}"
            )
            remove = [k for k in remove if k not in protected_keys]

    cleaned = {k: v for k, v in metrics.items() if k not in remove}

    # Guardrail b: surviving keys must be byte-identical to inputs.
    if not guardrail_values_unchanged(cleaned, metrics):
        logger.warning("Cleanup mutated value — skipping LLM pass")
        return _finalize(state, metrics, removed)

    # Guardrail c: cleanup must not introduce NEW identity / sanity failures.
    before_warnings = state.get("identity_warnings") or []
    passes, after_warnings = guardrail_no_new_warnings(cleaned, before_warnings)
    if not passes:
        new_failures = [w for w in after_warnings if w not in before_warnings]
        logger.warning(
            "Cleanup LLM introduced %d new identity warning(s) for %s — skipping LLM pass: %s",
            len(new_failures), ticker, new_failures,
        )
        return _finalize(state, metrics, removed)

    if remove:
        logger.info(
            "Cleanup (LLM) for %s dropped %d key(s): %s",
            ticker, len(remove), [f"{k} ({reasons.get(k, 'no reason')})" for k in remove],
        )
        removed.extend(remove)
    else:
        logger.info("Cleanup (LLM) for %s proposed no removals", ticker)

    return _finalize(state, cleaned, removed, identity_warnings=after_warnings)


def _finalize(
    state: EarningsAgentState,
    metrics: dict,
    removed: list[str],
    identity_warnings: list | None = None,
) -> EarningsAgentState:
    """Return updated state with the (possibly partially-)cleaned metrics."""
    out: dict = {**state, "metrics": metrics, "cleanup_removed": removed}
    if identity_warnings is not None:
        out["identity_warnings"] = identity_warnings
    return out  # type: ignore[return-value]
