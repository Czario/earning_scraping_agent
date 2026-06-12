"""Failure-mode **skill catalog** for the analysis layer.

A *skill* is a self-contained, discoverable unit of analysis knowledge about
one extraction failure mode. Each skill bundles three things that previously
lived scattered across the codebase:

  1. **Metadata** — a stable ``id``, a human ``title``, and the
     ``finding_types`` the skill can emit.
  2. **A detector** — the pure-observer checker from ``findings.py`` that spots
     the failure mode. Detectors keep the observer contract
     (``(metrics) -> list[Finding]``); they never mutate metrics (ADR-0003).
  3. **Remediation** — curated, generic guidance describing how to fix the
     failure mode. This text feeds both the re-extract hint and the
     company-hint drafter, so the system's "how to correct this" knowledge is
     written down once and reused everywhere.

This is deliberately *not* a runtime skill engine. Skills are a static,
code-reviewed catalog: adding a failure mode is a one-entry change here plus a
checker in ``findings.py`` and a regression test. Determinism, verbatim metric
keys, and the corrector/observer seam (ADR-0003) are all preserved — only the
*organisation* of knowledge changes, never the execution model.

Correctors (which mutate values) intentionally stay in ``analysis/validators.py``;
skills catalog observer detectors and remediation knowledge only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from earnings_agents.analysis.findings import (
    Finding,
    FindingType,
    check_case_duplicates,
    check_composite_keys,
    check_eps_dilution_ordering,
    check_gaap_nongaap_leakage,
    check_gross_profit_identity,
    check_opex_label_collision,
    check_operating_vs_gross,
    check_suspect_round,
)

Detector = Callable[[dict[str, Any]], list[Finding]]


@dataclass(frozen=True)
class Skill:
    """One self-contained analysis skill for a single failure mode.

    Attributes
    ----------
    id:
        Stable, lower-snake-case identifier (e.g. ``"gross_profit_identity"``).
    title:
        Human-readable one-liner shown in discovery listings.
    finding_types:
        The ``FindingType`` values this skill's detector may emit. Several
        skills legitimately share a type (e.g. ``suspect_value``), so this is a
        many-to-one relationship, not a unique key.
    remediation:
        Curated, company-agnostic guidance describing how to correct the
        failure mode. Reused verbatim in the re-extract hint block and the
        company-hint drafter prompt.
    detector:
        The pure-observer checker. ``None`` for skills that are invoked
        specially by the node because they need extra arguments (presence and
        source-grounding), which keeps the catalog complete for discovery while
        leaving their call sites untouched.
    """

    id: str
    title: str
    finding_types: tuple[FindingType, ...]
    remediation: str
    detector: Detector | None = None
    notes: str = field(default="")

    def detect(self, metrics: dict[str, Any]) -> list[Finding]:
        """Run the skill's detector, or return ``[]`` when it has none."""
        if self.detector is None:
            return []
        return self.detector(metrics)


# ---------------------------------------------------------------------------
# The catalog. Order matters: detectors run in this sequence in
# ``analyze_metrics_node`` (it mirrors the previous CHECKER_REGISTRY order).
# ---------------------------------------------------------------------------
SKILL_REGISTRY: list[Skill] = [
    Skill(
        id="case_duplicate",
        title="Detect case/whitespace-only duplicate keys with equal values",
        finding_types=("case_duplicate",),
        detector=check_case_duplicates,
        remediation=(
            "Emit each metric under a single key. Do not repeat the same line "
            "with only case or whitespace differences."
        ),
    ),
    Skill(
        id="composite_key",
        title="Detect synonym-list keys (comma/slash-joined labels)",
        finding_types=("composite_key",),
        detector=check_composite_keys,
        remediation=(
            "Use the exact label as printed in the document. Never join several "
            "synonyms into one key (e.g. 'Diluted EPS / Diluted net income per "
            "share')."
        ),
    ),
    Skill(
        id="gaap_nongaap_leakage",
        title="Detect GAAP/Non-GAAP reconciliation-table leakage",
        finding_types=("gaap_nongaap_leakage",),
        detector=check_gaap_nongaap_leakage,
        remediation=(
            "Extract values from the plain GAAP consolidated statements only. "
            "Skip 'Adjusted', 'Non-GAAP', and reconciliation/impact rows."
        ),
    ),
    Skill(
        id="gross_profit_identity",
        title="Verify Revenue − Cost of revenue = Gross profit",
        finding_types=("identity_violation",),
        detector=check_gross_profit_identity,
        remediation=(
            "Revenue, Cost of revenue, and Gross profit must come from the SAME "
            "current-period column and satisfy Revenue − Cost of revenue = "
            "Gross profit. Cost of revenue must be less than Revenue."
        ),
    ),
    Skill(
        id="operating_vs_gross",
        title="Verify Operating income does not exceed Gross profit",
        finding_types=("suspect_value",),
        detector=check_operating_vs_gross,
        remediation=(
            "Operating income = Gross profit − Operating expenses, so it must be "
            "≤ Gross profit. If Operating income looks larger, one value was "
            "taken from the wrong row or period column."
        ),
    ),
    Skill(
        id="eps_dilution_ordering",
        title="Verify Diluted EPS does not exceed Basic EPS",
        finding_types=("suspect_value",),
        detector=check_eps_dilution_ordering,
        remediation=(
            "For positive earnings, Diluted EPS must be ≤ Basic EPS (diluted "
            "divides by a larger share count). If Diluted > Basic, the two were "
            "likely swapped."
        ),
    ),
    Skill(
        id="suspect_round",
        title="Flag implausibly round values likely lifted from prose",
        finding_types=("suspect_round",),
        detector=check_suspect_round,
        remediation=(
            "Report values from the financial tables, not from narrative prose "
            "('debt of about $1 billion'). Use the exact figure printed in the "
            "statement."
        ),
    ),
    Skill(
        id="opex_label_collision",
        title="Detect Total operating expenses colliding with another row",
        finding_types=("suspect_value",),
        detector=check_opex_label_collision,
        remediation=(
            "Total operating expenses = Cost of revenue + all operating-expense "
            "line items (R&D, S&M, G&A). It can never equal Operating income or "
            "Revenue — re-extract it from the correct row."
        ),
    ),
    # ── Skills invoked specially by the node (extra arguments). Catalogued for
    # discovery; their call sites in analyze_metrics_node are unchanged. ──
    Skill(
        id="presence",
        title="Tiered presence check (Tier-1/2/3 metric registries)",
        finding_types=("missing_critical", "missing_expected"),
        detector=None,
        remediation=(
            "Ensure every Tier-1 income-statement line is present: Total "
            "Revenue, Gross Profit, Operating Income, Net Income, Diluted EPS."
        ),
        notes="Invoked via critical_metrics.check_presence with a precomputed summary.",
    ),
    Skill(
        id="source_grounding",
        title="Verify each value is grounded in a verbatim source snippet",
        finding_types=("source_unverified",),
        detector=None,
        remediation=(
            "Report only values that appear verbatim in the document; do not "
            "infer or compute values. Cite the exact snippet each value came "
            "from."
        ),
        notes="Invoked via findings.check_source_grounding with source text + snippets.",
    ),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def iter_detectors() -> list[Detector]:
    """Return the ordered observer detectors run by ``analyze_metrics_node``.

    Replaces the former ``findings.CHECKER_REGISTRY``: only skills that own a
    detector with the ``(metrics) -> list[Finding]`` contract are included.
    """
    return [s.detector for s in SKILL_REGISTRY if s.detector is not None]


def skill_by_id(skill_id: str) -> Skill | None:
    """Return the skill with *skill_id*, or ``None``."""
    return next((s for s in SKILL_REGISTRY if s.id == skill_id), None)


def skills_for_finding_types(finding_types: Iterable[str]) -> list[Skill]:
    """Return catalog skills whose ``finding_types`` intersect *finding_types*.

    Order follows ``SKILL_REGISTRY``. A skill is returned at most once even when
    it matches several of the requested types.
    """
    wanted = set(finding_types)
    return [s for s in SKILL_REGISTRY if wanted.intersection(s.finding_types)]


def remediation_block(finding_types: Iterable[str]) -> str:
    """Build a bullet list of curated remediations for the given finding types.

    Used by the company-hint drafter so generated hints are grounded in the
    system's known failure-mode fixes rather than improvised from scratch.
    Returns ``""`` when no catalogued skill matches.
    """
    skills = skills_for_finding_types(finding_types)
    if not skills:
        return ""
    return "\n".join(f"- ({s.id}) {s.remediation}" for s in skills)


# ---------------------------------------------------------------------------
# Skill-effectiveness tracking (observability)
# ---------------------------------------------------------------------------
#
# These are pure, deterministic functions used by ``analyze_metrics_node`` to
# record which skills' findings actually got *resolved* between re-extract
# passes — so a human can later see which detectors earn their keep. This is
# pure observation: it never mutates metrics, never selects skills at runtime,
# and never feeds back into routing (ADR-0001/0003/0006 preserved).


def finding_signature(finding: dict[str, Any]) -> str:
    """Return a stable identity for *finding*, used to match it across passes.

    Two findings from different passes are "the same finding" when their
    signatures match. The signature is ``"{type}::{subject}"`` where *subject*
    is the sorted metric keys when present, else the ``evidence.metric`` name
    (presence checks carry no ``keys``), else the message text.
    """
    ftype = finding.get("type") or ""
    keys = finding.get("keys") or ()
    if keys:
        subject = "|".join(sorted(str(k) for k in keys))
    else:
        evidence = finding.get("evidence") or {}
        subject = str(evidence.get("metric") or finding.get("message") or "")
    return f"{ftype}::{subject}"


def compute_skill_effectiveness(
    prev_findings: list[dict[str, Any]],
    curr_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare two consecutive analysis passes and roll up per finding type.

    Returns one record per finding type that appeared in either pass::

        {"finding_type": str, "skills": [skill_id, ...],
         "resolved": int, "persisted": int, "new": int}

    where *resolved* findings were present last pass and gone this pass,
    *persisted* findings appeared in both, and *new* findings appeared only
    this pass. ``skills`` lists the catalog skill ids that own the type. The
    result is sorted by ``finding_type`` for deterministic output; types with
    no movement at all are omitted.
    """
    def _by_type(findings: list[dict[str, Any]]) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for f in findings:
            if not isinstance(f, dict):
                continue
            ftype = f.get("type") or ""
            out.setdefault(ftype, set()).add(finding_signature(f))
        return out

    prev_by_type = _by_type(prev_findings)
    curr_by_type = _by_type(curr_findings)

    records: list[dict[str, Any]] = []
    for ftype in sorted(set(prev_by_type) | set(curr_by_type)):
        prev_sigs = prev_by_type.get(ftype, set())
        curr_sigs = curr_by_type.get(ftype, set())
        resolved = len(prev_sigs - curr_sigs)
        persisted = len(prev_sigs & curr_sigs)
        new = len(curr_sigs - prev_sigs)
        if not (resolved or persisted or new):
            continue
        records.append(
            {
                "finding_type": ftype,
                "skills": [s.id for s in skills_for_finding_types([ftype])],
                "resolved": resolved,
                "persisted": persisted,
                "new": new,
            }
        )
    return records
