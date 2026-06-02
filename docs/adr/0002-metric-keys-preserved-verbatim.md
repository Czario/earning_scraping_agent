# ADR 0002 — Metric keys are preserved exactly as they appear in source documents

**Status:** Accepted

---

## Context

When an LLM extracts financial metrics from an earnings release, it assigns
key names to each extracted value. Three approaches were considered:

1. **Free-form keys** — keep whatever label the LLM returned, verbatim.
2. **Normalised keys** — map all keys to a canonical vocabulary (e.g.
   `"net_income"`, `"revenue"`) at extraction time.
3. **Post-hoc mapping** — keep free-form keys through extraction and analysis;
   perform concept mapping as a separate, optional step only when the target
   save system (`normalize_data`) requires it.

Approach 2 makes extraction lossy: the company's exact wording is discarded,
the LLM must learn the canonical vocabulary, and any mismatch between what
the LLM names a metric and the canonical name silently drops data. It also
makes checkers in `analysis/findings.py` fragile — they rely on regex-matching
company document labels, not an internal vocabulary.

Approach 3 separates concerns cleanly: free-form extraction maximises recall;
semantic concept mapping (`load_company_concepts_node` + the Tier 2 LLM match
inside `extract_financial_metrics_node`) is an optional enrichment layer that
does not touch the primary metrics dict.

## Decision

Metric keys are **never renamed, normalised, or edited** after extraction.
The `metrics` dict in `EarningsAgentState` always holds the company's own
wording. Checkers match against these keys using regex patterns.

The `concept_metrics` field (`concept_id → float`) is the only normalised
representation; it is populated as an optional side-channel by
`extract_financial_metrics_node` and is never used by analysis checkers.

`cleanup_metrics_node` may **remove** duplicate keys (case variants,
Rule-A/B/C duplicates) but must not rename or modify the values of retained keys.

## Consequences

- **Positive:** Analysis checkers (`findings.py`) are robust to vocabulary
  changes — they match the company's own phrasing.
- **Positive:** MongoDB documents contain the original labels, making
  post-hoc debugging straightforward.
- **Positive:** Concept mapping failures do not corrupt the metrics dict.
- **Constraint:** Every checker that needs to find a key by meaning must use
  a regex pattern (or the `_find_first` helper in `analysis/validators.py`).
  Callers cannot assume key names.
- **Do not re-open unless:** A new save target requires normalised keys at
  extraction time AND the concept-mapping side-channel cannot satisfy it.
