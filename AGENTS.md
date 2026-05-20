# Earnings Agents — AI Agent Instructions

LangGraph-based pipeline that scrapes earnings releases (SEC EDGAR or IR pages), extracts financial metrics via a local Ollama LLM, and upserts results into MongoDB.

See [README.md](README.md) for full setup and CLI usage. See [project_flow.md](project_flow.md) for a step-by-step trace of the SEC flow.

---

## Build & Run

```bash
uv sync                                          # install / sync deps
uv run earnings --source sec --ticker MSFT       # SEC EDGAR mode
uv run earnings --source ir --ticker MSFT --ir-url "<url>"  # IR mode
uv run earnings-scheduler                        # run the APScheduler daemon
uv run pytest -q                                 # full test suite
uv run pytest tests/test_extract_financial_metrics.py -q    # focused test
```

Runtime requires **Ollama** and **MongoDB** running locally. Copy `.env.example` → `.env` and adjust before first run.

---

## Architecture

```
CLI (cli/earnings.py)
  └─ builds initial EarningsAgentState
       └─ workflow.build_graph() → LangGraph StateGraph
            nodes/discover_earnings_release.py  (short-circuits if URL already set)
            nodes/detect_document_type.py       (extension-first, HEAD fallback)
            nodes/extract_html_text.py          (SEC-aware: SGML stripping, no JS)
            nodes/extract_pdf_text.py
            nodes/extract_financial_metrics.py  (chunked Ollama calls; per-chunk retry)
            nodes/reflect_metrics.py            (Observe+Decide: loops back if critical metrics missing)
            workflow.mongodb_save_node          (upsert as TICKER_YEAR_latest)
```

The pipeline runs an **agentic loop** between `extract_financial_metrics` and `reflect_metrics`:
- **Perceive & Plan** (`extract_financial_metrics`): reads `extraction_notes` from state for focused hints, chunks text, builds prompts.
- **Act/Execute** (`extract_financial_metrics`): calls Ollama per chunk; retries up to 2× per chunk on JSON parse failure.
- **Observe** (`reflect_metrics`): asks the LLM to compare merged metrics against the source text excerpt.
- **Reflect & Decide** (`reflect_metrics`): if critical metrics (Revenue, Net Income, EPS, Operating Income) are missing and `extraction_attempts < MAX_EXTRACTION_ATTEMPTS` (3), injects `extraction_notes` and loops back. Otherwise routes to `mongodb_save`.

External integrations live in `tools/`: `edgar_client.py`, `mongodb_client.py`, `playwright_scraper.py`, `static_scraper.py`.

Configuration is in `config.py` (reads `.env` via `python-dotenv`).

---

## Key Conventions

### State machine
`EarningsAgentState` (defined in `workflow_state.py`) is a `TypedDict`. Status progression:
`pending → discovered → fetched → text_extracted → extracted → saved | failed`

On failure a node sets `status = "failed"` and `error = "<message>"`. Routing helpers in `workflow.py` short-circuit to `END` on failure.

The agentic loop uses two additional fields: `extraction_attempts: int` (capped at `MAX_EXTRACTION_ATTEMPTS = 3`) and `extraction_notes: Optional[str]` (injected by `reflect_metrics`, consumed by `extract_financial_metrics`).

### Metric keys
Metric keys are preserved **exactly** as they appear in company documents — do not normalize or rename them. The special `__scale__` key (`"millions"` / `"thousands"` / `"billions"` / `"as-is"`) is set by the LLM per chunk and applied in Python after merging.

### MongoDB document IDs
Documents are upserted with `_id = "{TICKER}_{YEAR}_latest"` (e.g. `GOOGL_2026_latest`). Re-runs overwrite the previous result.

### Adding a new LangGraph node
1. Create `src/earnings_agents/nodes/<node_name>.py` with a function `<node_name>_node(state: EarningsAgentState) -> EarningsAgentState`.
2. Register it in `workflow.py` with `graph.add_node(...)` and wire edges/routing.

### SEC-specific behaviour (extract_html_text)
- Uses a SEC-compliant `User-Agent` header.
- Strips SGML wrappers from EDGAR archive responses.
- JavaScript fallback is **disabled** for `sec.gov` URLs.

---

## Known Issues

- `tools/edgar_client.py` contains duplicate definitions of `normalize_cik` and `get_latest_earnings_url`. The later definitions override the earlier ones at runtime; consolidate when editing that file.
