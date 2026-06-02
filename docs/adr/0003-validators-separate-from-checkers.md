# ADR 0003 — Validators (`analysis/validators.py`) are separate from checkers (`analysis/findings.py`)

**Status:** Accepted

---

## Context

After LLM extraction the pipeline needs two kinds of post-merge analysis:

1. **Checkers (observers)** — inspect the metrics dict and return structured
   `Finding` objects that describe what they found. They never mutate the dict.
   Contract: `(metrics: dict) → list[Finding]`.

2. **Validators (correctors)** — apply accounting identity constraints,
   **null out** implausible values (e.g. FCF > Operating Cash Flow), and
   return both the corrected dict and a list of warning strings that the save
   node uses to decide whether to abort.
   Contract: `(metrics: dict) → tuple[dict, list[str]]`.

Two placement options were considered:
- **Option A:** Append `validate_metrics` and its helpers to `findings.py`.
- **Option B:** Create a separate `analysis/validators.py` module.

Option A mixes two fundamentally different contracts in one module. Readers
must distinguish observer functions (return `list[Finding]`) from corrector
functions (return `tuple[dict, list[str]]`) with no structural boundary.
The `cleanup_metrics` node importing a private corrector from a sibling node
(`nodes/extract_financial_metrics._validate_metrics`) was exactly the symptom
of the uncleaned seam that triggered this decision.

Option B places the corrector behind a clean seam. Both callers
(`extract_financial_metrics_node` and `cleanup_metrics_node`) import from
`analysis/validators`, not from each other.

## Decision

`analysis/validators.py` is the home for deterministic post-merge correctors:
- `_find_first(metrics, pattern)` — private regex key-lookup helper.
- `_check_identity(name, lhs, rhs, *, tolerance)` — private tolerance check.
- `validate_metrics(metrics)` — **public** corrector; the only exported symbol.
  Returns `(corrected_dict, warnings)`.

`analysis/findings.py` contains only observer checkers that return
`list[Finding]`. It has no knowledge of `validators.py`.

## Consequences

- **Positive:** `cleanup_metrics_node` no longer imports from a sibling node's
  private internals. The cross-node coupling that prevented isolated testing
  of cleanup is eliminated.
- **Positive:** The two contracts (`list[Finding]` vs `tuple[dict, list[str]]`)
  are enforced by module boundaries, not comments.
- **Positive:** `validate_metrics` can be unit-tested directly without
  constructing any `Finding` objects or running `analyze_metrics_node`.
- **Constraint:** Validators may not produce `Finding` objects. If a correction
  needs to be surfaced as a `Finding` (e.g. for the re-extract loop), the
  corrected value should be emitted by the validator and a separate checker in
  `findings.py` should observe the result.
- **Do not re-open unless:** The validator and checker seams need to be unified
  under a single typed protocol (e.g. a `Corrector` dataclass), which would
  warrant a new ADR.
