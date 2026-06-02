# ADR 0005 — `MAX_EXTRACTION_ATTEMPTS` as a hard cap with no infinite retry paths

**Status:** Accepted

---

## Context

`analyze_metrics_node` can set `needs_reextract = True` when high-severity
findings remain after an extraction pass. Without a cap, a document that
consistently produces high-severity findings (e.g. a filing format the LLM
cannot parse) would loop indefinitely, consuming tokens and blocking the
queue.

Three approaches were considered:

1. **No cap** — loop until all findings resolve. Risks infinite loops on
   documents that can never fully satisfy the analysis checkers.

2. **Fixed hard cap** — stop looping after *N* attempts regardless. Simple,
   predictable, prevents runaway costs. The document is saved with
   `degraded` MongoDB document status to signal incomplete data.

3. **Adaptive cap** — increase the cap if the LLM escalates to a larger
   model or a different provider on each pass. This was removed because it
   triggered rate-limit errors on Groq's free tier (`429`) and added
   escalation logic that obscured the retry semantics.

Option 3 was explicitly tried and removed. Option 2 was adopted.

## Decision

`MAX_EXTRACTION_ATTEMPTS` (default `3`, env-overridable via
`MAX_EXTRACTION_ATTEMPTS`) is the hard cap. When `extraction_attempts`
reaches this value, `analyze_metrics_node` sets `needs_reextract = False`
regardless of finding severity.

A no-progress guard also breaks early: if the set of high-severity finding
messages is identical to the previous pass (`previous_high_finding_keys`),
the loop breaks before reaching the cap.

Documents saved with unresolved Tier-1 findings receive `degraded` MongoDB
document status rather than `failed`, so downstream consumers know data is
present but incomplete.

## Consequences

- **Positive:** Every pipeline run terminates in bounded time.
- **Positive:** The cap is environment-overridable for development/testing
  without code changes.
- **Positive:** The no-progress guard avoids burning attempts on a stuck LLM.
- **Constraint:** No code path may introduce a retry loop that does not check
  against `MAX_EXTRACTION_ATTEMPTS`. A new retry mechanism (e.g. cleanup
  triggering re-extraction) must use the same counter or a sibling counter
  with its own cap.
- **Constraint:** LLM provider escalation (e.g. switching from Ollama to
  Groq on retry) is not permitted inside the extraction loop. Provider
  selection is decided once at pipeline startup via `LLM_PROVIDER`.
- **Do not re-open unless:** A use case requires more than 3 passes with
  evidence that additional passes improve yield on a class of documents.
