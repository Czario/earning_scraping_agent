# ADR 0004 — MongoDB document ID format: `{TICKER}_{YEAR}_latest`

**Status:** Accepted

---

## Context

The pipeline upserts extraction results into MongoDB. Each document represents
the most recent earnings result for a given company and fiscal year.

Several ID schemes were considered:

1. **Auto-generated ObjectId** — MongoDB's default. Re-runs create new
   documents; there is no natural way to find "the current result for MSFT
   FY2024" without a secondary index.

2. **`{TICKER}_{YEAR}_{RUN_TIMESTAMP}`** — each run creates a separate
   document; history is preserved but the "current" result is ambiguous
   without sorting.

3. **`{TICKER}_{YEAR}_latest`** — a deterministic, human-readable key. Re-runs
   overwrite the previous document for the same ticker/year. The `_latest`
   suffix makes clear this is a live slot, not an append-only log.

The pipeline's primary use case is to maintain a current view of each
company's reported metrics. Historical versioning is out of scope and adds
query complexity. The suffix `_latest` was chosen over a plain
`{TICKER}_{YEAR}` key to leave the namespace open for future archival
documents (e.g. `{TICKER}_{YEAR}_v1`) without changing existing consumers.

## Decision

The MongoDB `_id` for every upserted document is `{TICKER}_{YEAR}_latest`
where:
- `TICKER` is the uppercased ticker symbol (e.g. `MSFT`, `AAPL`).
- `YEAR` is the 4-digit fiscal year from the source document.

Re-runs for the same ticker/year **upsert** (overwrite) the existing document.

## Consequences

- **Positive:** "Get the latest MSFT 2024 result" is a point lookup, not a
  query.
- **Positive:** Re-runs are idempotent from the caller's perspective.
- **Constraint:** There is no built-in audit trail. If run history is needed
  later, a separate `earnings_history` collection with timestamped documents
  should be introduced rather than changing the `_id` format.
- **Constraint:** `YEAR` is the company's fiscal year from the document, not
  the calendar year of the run or the report date.
- **Do not re-open unless:** The platform requires immutable audit records or
  multiple "snapshots" per fiscal year to coexist in the same collection.
