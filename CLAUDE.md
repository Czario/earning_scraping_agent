# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install all dependencies
uv run pytest -q                 # Run all tests
uv run pytest -q tests/test_<module>.py    # Run a single test file
uv run pytest -q -k <pattern>              # Run tests matching a pattern

# CLI (SEC EDGAR path — default)
uv run earnings --ticker MSFT
uv run earnings --ticker AAPL MSFT GOOGL --max-workers 4
uv run earnings --ticker MSFT --dry-run    # Connectivity check, no LLM
uv run earnings --ticker MSFT -v           # Verbose (DEBUG logging)

# Worker (Redis consumer for 8-K filings)
uv run earnings-8k-worker --once           # Process one message, then exit
uv run earnings-skills                     # Browse the failure-mode skill catalog
uv run earnings-failures                   # Browse failed extraction records

# Docker
docker compose up -d --build               # Build + start the 8-K worker
docker compose restart worker-8k           # After code-only edits (src is mounted)
docker compose logs -f worker-8k
```

## Architecture

This is a **LangGraph-based earnings extraction pipeline** that ingests SEC 8-K Exhibit 99.1 press releases and extracts financial metrics into MongoDB.

### Graph Nodes (pipeline order)

```
load_company_concepts → detect_document_type → extract_html_text
  → extract_financial_metrics ⇄ analyze_metrics   (agentic loop, up to 3 passes)
    → cleanup_metrics → mongodb_save
```

Each node is a pure function `(EarningsAgentState) → EarningsAgentState`, wrapped by `with_hooks()` for structured logging, timing, and exception-to-failure conversion. Node failures set `status="failed"` and `error=<msg>`; routing helpers check `status` to short-circuit to `END`.

### State Machine

`EarningsAgentState` (`workflow_state.py`) is a `TypedDict` that carries all pipeline data. Status progression: `discovered → fetched → text_extracted → extracted → saved | failed`. Critical fields include:

- `target_concepts` — GAAP concept list loaded from normalize_data, which drives **targeted extraction** (there is no generic extraction path anymore)
- `metrics` — raw extracted key-value pairs (company-native labels)
- `concept_metrics` — `concept_id → float` mapping for normalize_data upsert
- `findings` — structured `Finding` objects from analysis (severity: high/medium/low)
- `needs_reextract` — routing signal that triggers the extract↔analyze loop
- `sec_report_date` — authoritative period-end date from SEC EDGAR submissions API

### Agentic Loop (extract ↔ analyze)

1. **extract_financial_metrics**: Splits raw text into chunks (GAAP-section-aware or character-based), runs one LLM call per chunk, merges results. On retry passes, uses `extraction_notes` from the prior analysis to focus the LLM on specific errors. Chunk-level provenance (`chunk_metric_sources`) enables scoped retries that only re-run affected chunks.

2. **analyze_metrics**: Runs deterministic checkers (presence, identity violations, suspect rounding, case duplicates, GAAP/Non-GAAP leakage, etc.) producing `Finding` objects. High-severity findings trigger `needs_reextract=True` when attempts remain and findings differ from the prior pass (progress detection prevents infinite loops). Medium findings add hints only when re-extract is already triggered.

The loop is capped by `MAX_EXTRACTION_ATTEMPTS` (default 3, configurable via env var).

### Three-Tier Concept Mapping (targeted extraction)

When `target_concepts` are loaded from normalize_data:
- **Tier 0**: Direct taxonomy_key match (LLM returns `[us-gaap:Revenue]` bracket key or bare key)
- **Tier 1**: Deterministic label matching (exact, then normalized)
- **Tier 2**: LLM semantic mapping — matches orphaned extracted keys to unmapped concepts, with filters excluding OCI/dimensional concepts
- **Tier 3**: Pure-Python derivation engine — computes missing values from mapped ones (e.g., Gross Profit = Revenue − COGS)

### Analysis Layer (`src/earnings_agents/analysis/`)

- **`findings.py`** — Pure observers: checkers that return `Finding` objects without mutating data (ADR-0003)
- **`validators.py`** — Correctors: deterministic functions that may null-out implausible values (identity violations, scale errors). Runs before analysis.
- **`skills.py`** — Failure-mode skill catalog: each skill bundles metadata, a detector, and curated remediation text. Used by `analyze_metrics` to generate focused re-extract hints.
- **`calculators.py`** — Tier-3 derivation engine and LLM role identification for unrecognized P&L labels

### Extraction Subsystem (`src/earnings_agents/extraction/`)

- **`chunker.py`** — Document prescan (scale, period, shares-in-thousands detection), GAAP-section-aware chunking with priority ordering (income statement first, then balance sheet, cash flow)
- **`merger.py`** — Cross-chunk metric merging with section provenance, scale resolution, duplicate handling, and flagged-chunk identification for scoped retries
- **`concept_mapper.py`** — Tier-2 LLM semantic concept mapping and concept prompt list builder

### LLM Provider System

Multi-provider via `llm_factory.py`: **Ollama** (default, local), **Groq**, **Gemini**, **DeepSeek**. Set `LLM_PROVIDER` in `.env`. All providers expose a uniform `llm.invoke(str) -> str` interface. JSON mode is enabled per-call via `build_llm(format_json=True)`.

Key config env vars: `LLM_PROVIDER`, `OLLAMA_MODEL`, `GROQ_API_KEY`/`GROQ_MODEL`, `GEMINI_API_KEY`/`GEMINI_MODEL`, `DEEPSEEK_API_KEY`/`DEEPSEEK_MODEL`, `OLLAMA_NUM_PARALLEL`.

### Two Deployment Modes

1. **CLI** (`cli/earnings.py`): Runs the graph directly via `ThreadPoolExecutor` for one or more tickers. Uses Rich for live progress with per-node step summaries and chunk-level status. Default `--max-workers 8`. Single-company runs skip the thread pool.

2. **Redis Worker** (`cli/worker_8k.py`): Long-running `BLPOP` consumer on the `sec:filings:8k` queue. Publishes real-time progress events to Redis pub/sub (`sec:worker:events`) for admin_backend SSE streaming. Includes heartbeat, dead-letter queue, and retry logic. Docker-based deployment via `docker-compose.yml`.

### Code Organization

```
src/earnings_agents/
  workflow.py           # LangGraph graph builder + routing helpers
  workflow_state.py     # EarningsAgentState TypedDict
  config.py             # All env-var-driven settings
  hooks.py              # Node lifecycle hooks (logging, timing, error handling)
  llm_factory.py        # Multi-provider LLM client builder
  company_registry.py   # CIK/ticker lookup from data/reference/tickers.json
  worker_progress.py    # Redis pub/sub progress streaming for worker mode
  nodes/                # LangGraph node functions (one per pipeline stage)
  analysis/             # Deterministic checkers, validators, calculators, skills
  extraction/           # Chunking, merging, concept mapping
  tools/                # External integrations (EDGAR, MongoDB, Redis, HTTP, Playwright, LLM extractor)
  cli/                  # CLI entry points
```

### Key Patterns and Guardrails

- **Metric keys are preserved verbatim** — no normalization at extraction time (ADR-0002)
- **Cleanup is append-only in spirit** — the cleanup node can only drop keys; it cannot rename, mutate values, or add keys. Guardrails reject any LLM output that violates this.
- **Save gate**: `STRICT_ACCURACY` (default on) refuses MongoDB upsert when accounting identity checks fail. Override with `--allow-inconsistent`.
- **MongoDB document ID format**: `{TICKER}_{YEAR}_latest` (ADR-0004)
- **EDGAR rate limiting**: Token-bucket at ≤10 req/s per SEC guidelines
- **Incremental guard**: Skips filings whose `sec_report_date` is ≤ the most recently stored period in normalize_data, preventing redundant reprocessing

### Tests

Tests use `pytest` with `pytest-asyncio` (auto mode). Golden fixture tests (`test_golden_fixtures.py`) validate extraction output against known-good results. Keep tests deterministic; mock network/LLM calls.
