# Copilot Instructions for Earnings Agents

This repository implements a LangGraph-based earnings extraction pipeline.

Shared harness principles live in `.github/instructions/agent-harness.instructions.md` and should be treated as baseline agent behavior guidance.

## Environment and commands
- Use the `uv` workflow.
- Install/sync dependencies: `uv sync`
- Run SEC mode: `uv run earnings --source sec --ticker MSFT`
- Run IR mode: `uv run earnings --source ir --ticker MSFT --ir-url "<url>"`
- Run tests: `uv run pytest -q`
- When changing a node or tool module, run only the matching tests: `uv run pytest -q tests/test_<module>.py` or `uv run pytest -q -k <module_name>`.

## Architecture guardrails
- Workflow entrypoint is `src/earnings_agents/workflow.py`.
- State model is `src/earnings_agents/workflow_state.py`.
- Node functions are in `src/earnings_agents/nodes/`.
- Deterministic analysis logic is in `src/earnings_agents/analysis/`.
- External integrations are in `src/earnings_agents/tools/`.

## Coding conventions
- Preserve `EarningsAgentState` shape and status progression:
  `pending -> discovered -> fetched -> text_extracted -> extracted -> saved|failed`
- On failures in nodes, set:
  - `state["status"] = "failed"`
  - `state["error"] = "<message>"`
- Metrics cleanup is limited to duplicate removal and structural cleanup; for retained metrics, do not change key text or values.
- Respect `__scale__` handling (`millions`, `thousands`, `billions`, `as-is`).

## Agentic loop behavior
- `extract_financial_metrics` and `analyze_metrics` can loop when high-severity findings exist.
- High-severity findings are those with severity field equal to `error` or `critical` in the analysis output.
- Keep `MAX_EXTRACTION_ATTEMPTS` limit behavior consistent.
- `cleanup_metrics` only removes duplicate entries and structural artifacts (e.g. empty containers). It never renames, normalizes, or edits the text of retained metric keys or their values. Run the deterministic pass first; any optional LLM-assisted pass is bound by the same rule.

## Agent Development Principles
- See `.github/instructions/agent-harness.instructions.md` for the full harness-engineering principles applied in this repo.

## Data and persistence
- MongoDB upsert id format must remain: `{TICKER}_{YEAR}_latest`.
- `YEAR` is the 4-digit fiscal year from the source document and `TICKER` is uppercase (example: `MSFT_2024_latest`).
- Re-runs should overwrite the previous latest document for the same ticker/year.

## Testing expectations
- Add or update tests in `tests/` for behavioral changes.
- For node/tool modifications, run only the matching tests: `uv run pytest -q tests/test_<module>.py` or `uv run pytest -q -k <module_name>`.
- Keep tests deterministic; avoid network calls unless explicitly integration-scoped.

## Repository notes
- `src/earnings_agents/tools/edgar_client.py` has a single canonical definition of `normalize_cik` and `get_latest_earnings_url`.
- `MAX_EXTRACTION_ATTEMPTS` is defined in `src/earnings_agents/config.py` (env-overridable via `MAX_EXTRACTION_ATTEMPTS`). `src/earnings_agents/nodes/reflect_metrics.py` has been deleted.