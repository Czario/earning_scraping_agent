---
applyTo: "**"
---

# Earnings Agents Scoped Instructions

Use these instructions when editing the LangGraph earnings extraction pipeline.

## Primary workflow
- Prefer `uv` commands.
- Install/sync deps with `uv sync`.
- Run tests with `uv run pytest -q`.
- When changing a node or tool module, run only the matching tests: `uv run pytest -q tests/test_<module>.py` or `uv run pytest -q -k <module_name>`.

## State and graph rules
- Preserve `EarningsAgentState` shape from `workflow_state.py`.
- Preserve status progression: `pending -> discovered -> fetched -> text_extracted -> extracted -> saved|failed`.
- On node failure, set `status = "failed"` and populate a clear `error` string.
- Keep routing behavior in `workflow.py` consistent with failure short-circuit logic.

## Extraction and analysis loop
- Keep loop behavior between extraction and analysis nodes deterministic.
- Respect extraction retry cap from existing constants/imports.
- Do not introduce infinite retry paths.
- Follow shared harness rules in `.github/instructions/agent-harness.instructions.md`.

## Metrics handling
- Keep extracted metric keys exactly as found in source documents.
- Do not normalize or rename user-facing metric keys.
- Preserve `__scale__` handling and post-merge scale application behavior.

## Persistence
- Preserve MongoDB `_id` format: `{TICKER}_{YEAR}_latest`.
- Re-runs for same ticker/year should upsert/overwrite latest document.

## Testing expectations
- Add or update tests in `tests/` with behavior changes.
- Keep tests deterministic and isolated.
- Avoid network dependence in unit tests unless intentionally integration-scoped.

## Repo caveats
- `tools/edgar_client.py` has a single canonical definition of `normalize_cik` and `get_latest_earnings_url`.
- `MAX_EXTRACTION_ATTEMPTS` is defined in `config.py` (env-overridable). `nodes/reflect_metrics.py` has been deleted; do not re-create it.
- `needs_reextract: bool` in state is the routing signal for the extract→analyze loop; do not revert to overloading `status` for routing.
- MongoDB document `status` can be `"success"`, `"degraded"`, or `"failed"`. `"degraded"` means unresolved Tier-1 findings after all attempts.
