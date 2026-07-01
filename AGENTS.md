# Earnings Agents — AI Agent Instructions

LangGraph-based pipeline that scrapes earnings releases (SEC EDGAR or IR pages), extracts financial metrics via a local Ollama LLM (or Groq), and upserts results into MongoDB.

See [README.md](README.md) for full setup and CLI usage. See [project_flow.md](project_flow.md) for a step-by-step trace of the SEC flow.

---

## Build & Run

```bash
uv sync                                          # install / sync deps
uv run earnings --source sec --ticker MSFT       # SEC EDGAR mode
uv run earnings --source ir --ticker MSFT --ir-url "<url>"  # IR mode
uv run earnings --ticker AAPL MSFT GOOGL         # multi-ticker parallel run
uv run pytest -q                                 # full test suite
uv run pytest tests/test_extract_financial_metrics.py -q    # focused test
```

Runtime requires **Ollama** (or a Groq API key) and **MongoDB** running locally. Copy `.env.example` → `.env` and adjust before first run.

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
            nodes/extract_financial_metrics.py  (chunked LLM calls; per-chunk retry)
            nodes/analyze_metrics.py            (deterministic QC; loops back on critical gaps)
            nodes/cleanup_metrics.py            (LLM dedup pass; deterministic case-dedup first)
            workflow.mongodb_save_node          (upsert as TICKER_YEAR_latest)
```

The pipeline runs an **agentic loop** between `extract_financial_metrics` and `analyze_metrics`:
- **Extract** (`extract_financial_metrics`): reads `extraction_notes` hints, chunks raw text, calls the LLM per chunk, merges and validates metrics.
- **Analyse** (`analyze_metrics`): runs pure-Python checkers from `analysis/` — tiered presence checks, case-duplicate detection, balance-sheet identity, sign anomalies, suspect-round heuristic. Produces `state["findings"]`. If any `high`-severity finding exists and `extraction_attempts < MAX_EXTRACTION_ATTEMPTS` (3), sets `extraction_notes` and loops back.
- **Cleanup** (`cleanup_metrics`): applies deterministic case-dedup from `findings`, then an LLM pass to drop Rule-A/B/C duplicates.

Pure-Python analysis helpers live in `analysis/`: `critical_metrics.py` (tiered metric registries + `check_presence`), `findings.py` (the `Finding` dataclass and all checker functions), `skills.py` (the failure-mode skill catalog — bundles each checker with metadata and curated remediation; `iter_detectors()` replaces the former `CHECKER_REGISTRY`; browsable via `uv run earnings-skills`).

External integrations live in `tools/`: `edgar_client.py`, `mongodb_client.py`, `playwright_scraper.py`, `static_scraper.py`.

LLM provider selection is in `llm_factory.py` (reads `LLM_PROVIDER` env var: `"ollama"` | `"groq"`). Configuration is in `config.py` (reads `.env` via `python-dotenv`).

---

## Key Conventions

### State machine
`EarningsAgentState` (defined in [`workflow_state.py`](src/earnings_agents/workflow_state.py)) is a `TypedDict`. Status progression:
`pending → discovered → fetched → text_extracted → extracted → saved | failed`

On failure a node sets `status = "failed"` and `error = "<message>"`. Routing helpers in `workflow.py` short-circuit to `END` on failure.

Key agentic-loop fields:
| Field | Type | Purpose |
|---|---|---|
| `extraction_attempts` | `int` | Incremented each pass; capped at `MAX_EXTRACTION_ATTEMPTS = 3` |
| `extraction_notes` | `Optional[str]` | Hint block injected by `analyze_metrics`, consumed by `extract_financial_metrics` |
| `findings` | `Optional[list]` | `Finding.to_dict()` entries from `analyze_metrics`; consumed by `cleanup_metrics` |
| `identity_warnings` | `Optional[list]` | Accounting identity failures; blocks save when `STRICT_ACCURACY=true` |
| `cleanup_removed` | `Optional[list]` | Keys dropped by cleanup (informational) |

### Finding severity → routing
- `"high"` (e.g. missing Tier-1 metric, balance-sheet identity violation) → triggers re-extract loop.
- `"medium"` (e.g. missing Tier-2 metric, sign anomaly) → appended to `extraction_notes` but does NOT loop alone.
- `"low"` (e.g. case-duplicate, suspect-round number) → deterministic fix or observation only; never loops.

### Tiered metric registry (`analysis/critical_metrics.py`)
- **Tier 1** (8 items, `TIER1_REGISTRY`) — must be present; missing = re-extract.
- **Tier 2** (13 items, `TIER2_REGISTRY`) — should be present; missing = informational hint.
- **Tier 3** (7 items, `TIER3_REGISTRY`) — optional; tracked when present.

### Metric keys
Metric keys are preserved **exactly** as they appear in company documents — do not normalize or rename them. The special `__scale__` key (`"millions"` / `"thousands"` / `"billions"` / `"as-is"`) is set by the LLM per chunk and applied in Python after merging.

### MongoDB document IDs
Documents are upserted with `_id = "{TICKER}_{YEAR}_latest"` (e.g. `GOOGL_2026_latest`). Re-runs overwrite the previous result.

### Adding a new LangGraph node
1. Create `src/earnings_agents/nodes/<node_name>.py` with a function `<node_name>_node(state: EarningsAgentState) -> EarningsAgentState`.
2. Register in `workflow.py` with `graph.add_node(...)` and wire edges/routing.
3. Add tests in `tests/test_<node_name>.py`.

### Adding a new deterministic checker
1. Add the checker function to `src/earnings_agents/analysis/findings.py` returning `list[Finding]`. It must not mutate the dict.
2. Add the new `FindingType` literal to the union in `findings.py`.
3. Add a `Skill` entry to `SKILL_REGISTRY` in `src/earnings_agents/analysis/skills.py` — set `detector=<your_function>` and write `remediation` text. The node iterates `iter_detectors()` automatically; **do not edit `analyze_metrics_node`**.
4. Add a test in `tests/test_skills.py` (catalog integrity) and/or `tests/test_analyze_metrics.py` (end-to-end finding).

### SEC-specific behaviour (`extract_html_text`)
- Uses a SEC-compliant `User-Agent` header.
- Strips SGML wrappers from EDGAR archive responses.
- JavaScript fallback is **disabled** for `sec.gov` URLs.

---

## Known Issues

- No known structural issues. `tools/edgar_client.py` has a single canonical definition of `normalize_cik` and `get_latest_earnings_url`; `MAX_EXTRACTION_ATTEMPTS` lives in `config.py` (env-overridable); `nodes/reflect_metrics.py` has been deleted and all its consumers updated.
