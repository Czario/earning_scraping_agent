# ADR 0001 — `needs_reextract` as the sole routing signal for the extract↔analyze loop

**Status:** Accepted

---

## Context

The pipeline runs an agentic loop between `extract_financial_metrics` and
`analyze_metrics`. After analysis, the graph must decide whether to loop back
for another extraction pass or proceed to cleanup and save.

Two mechanisms were available:
1. **Overload `status`** — set `status` to a new value such as `"needs_retry"` inside `analyze_metrics_node`, then branch in `_route_after_analysis` on that value.
2. **Dedicated boolean field** — add `needs_reextract: bool` to `EarningsAgentState` that `analyze_metrics_node` sets explicitly.

Option 1 collapses two orthogonal concerns: *pipeline lifecycle stage* (what
step was last completed) and *routing intent* (should we retry?). This makes
`status` ambiguous and breaks the invariant that status progression is strictly
linear (`pending → discovered → fetched → text_extracted → extracted →
saved | failed`).

Option 2 keeps each field single-purpose. Routing helpers can read
`needs_reextract` without knowing anything about the status lifecycle. The
status field continues to record the completed stage only.

## Decision

`needs_reextract: bool` is a dedicated field in `EarningsAgentState`. It is:
- Written exclusively by `analyze_metrics_node`.
- Read exclusively by `_route_after_analysis` in `workflow.py`.
- Reset to `False` on every entry into `analyze_metrics_node` before the
  loop decision is made.

`status` is never used as a routing signal for the extract↔analyze loop.

## Consequences

- **Positive:** `status` retains its single meaning; status-based failure
  short-circuits (`state.get("status") == "failed" → "__end__"`) are
  unambiguous and apply uniformly.
- **Positive:** `analyze_metrics_node` can be tested in isolation — its output
  contract is `needs_reextract` + `findings`, not a `status` string that
  routing also depends on.
- **Constraint:** Any future loop-control mechanism (e.g. a cleanup-rerun
  signal) must follow the same pattern — a dedicated boolean/enum field, not
  an overloaded `status` value.
- **Do not re-open unless:** routing complexity grows to a point where
  a named `RetryIntent` enum produces clearer routing helpers than multiple
  boolean flags.
