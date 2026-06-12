# ADR 0006 — Failure-mode **skill catalog** (`analysis/skills.py`)

**Status:** Accepted

---

## Context

The analysis layer's knowledge about each extraction failure mode was scattered
across three places:

1. The **detector** — a pure-observer checker in `analysis/findings.py`.
2. The **ordering** — a hand-maintained `CHECKER_REGISTRY` list (also in
   `findings.py`) that `analyze_metrics_node` iterated.
3. The **remediation** — "how to correct this" guidance, which existed only as
   ad-hoc strings inside `_build_extraction_notes` and improvised text in the
   company-hint drafter prompt.

Adding or understanding a failure mode meant touching disconnected spots, and
the remediation knowledge was duplicated/improvised rather than written down
once. We evaluated the Hermes-agent "skills" concept — self-contained,
discoverable units that bundle metadata + behaviour + guidance — and wanted its
*organisational* benefit without adopting a runtime skill-selection engine
(which would break determinism and ADR-0001 routing).

## Decision

Introduce `analysis/skills.py`: a **static, code-reviewed catalog** where each
failure mode is one `Skill` entry bundling:

- **Metadata** — stable `id`, human `title`, and the `finding_types` it emits.
- **A detector** — the existing pure-observer checker from `findings.py`
  (`(metrics) → list[Finding]`), referenced, not reimplemented. `None` for the
  special-invocation checks (`check_presence`, `check_source_grounding`) that
  need extra arguments and are called directly by the node.
- **Remediation** — curated, company-agnostic guidance, reused verbatim by the
  re-extract hint block and the company-hint drafter.

`SKILL_REGISTRY` is the single ordered source of truth. The former
`findings.CHECKER_REGISTRY` is **removed**; `analyze_metrics_node` now iterates
`skills.iter_detectors()` (same order, detector-bearing skills only). Lookup
helpers: `skill_by_id`, `skills_for_finding_types`, `remediation_block`.

Dependency direction is one-way: `skills.py` imports from `findings.py`;
`findings.py` has no knowledge of `skills.py`.

Discoverability CLI: `uv run earnings-skills` (`earnings_agents.cli.skills`).

## Consequences

- **Positive:** Adding a failure mode is a one-entry change in `skills.py` plus
  a checker in `findings.py` and a regression test — detector, ordering, and
  remediation live together.
- **Positive:** Remediation knowledge is written once and reused by both the
  re-extract loop and the hint drafter (grounded, not improvised).
- **Positive:** The catalog is browsable via `earnings-skills` without reading
  source.
- **Preserves ADR-0003:** Correctors (which mutate values) stay in
  `analysis/validators.py`. Skills catalog *observer detectors and remediation
  knowledge only* — a skill referencing a pure detector does not merge the
  observer/corrector seam.
- **Preserves ADR-0001:** Routing still keys off `needs_reextract`; severity →
  routing semantics are unchanged. The skill layer reorganises knowledge, not
  the execution model.
- **Not a runtime engine:** Skills are not dynamically selected, scored, or
  loaded. Execution remains a deterministic ordered detector sweep.
- **Do not re-open unless:** We need dynamic/runtime skill selection, at which
  point the determinism guarantees in this ADR and ADR-0001 must be revisited.
