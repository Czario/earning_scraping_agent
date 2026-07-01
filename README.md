# Earning Agents

LangGraph-based earnings scraping pipeline with two discovery modes:

- SEC EDGAR mode: finds latest 8-K Item 2.02 Exhibit 99.1
- IR mode: discovers earnings release link from a company IR page

Metrics are extracted dynamically with company-native labels and stored in MongoDB.

## Project Layout

- `src/earnings_agents/` core package
- `src/earnings_agents/cli/earnings.py` primary CLI implementation
- `data/reference/tickers.json` ticker and CIK lookup data
- `tests/` unit tests

## Prerequisites

- Python 3.12+
- uv
- Ollama running locally
- MongoDB running locally (or update URI in env)

pkill ollama
OLLAMA_NUM_PARALLEL=4 ollama serve

uv run earnings --ticker MSFT --dry-run
uv run earnings --ticker AAPL MSFT GOOGL --dry-run

uv run earnings --ticker AAPL MSFT GOOGL NVDA          # 4 parallel workers (default)
uv run earnings --ticker AAPL MSFT GOOGL --max-workers 2

uv run earnings --ticker MSFT -v

# or

uv run earnings --ticker MSFT --verbose

uv run earnings --ticker MSFT          # strict — refuses to save bad data
uv run earnings --ticker MSFT --allow-inconsistent   # save anyway, with audit

--max-workers N (default 4): bounds the ThreadPoolExecutor
Single-company runs skip the thread pool entirely (zero overhead)
Each worker buffers its own output; output is printed in submission order after all workers finish — no interleaved lines
Both live runs and --dry-run runs are parallelized the same way

## Environment

Create or update `.env` with:

```ini
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=earnings_db
MONGODB_COLLECTION=earnings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_NUM_CTX=4096
CHUNK_SIZE=6000
CHUNK_OVERLAP=300
```

## Install Dependencies

```bash
uv sync
```

## CLI Usage

Show help:

```bash
uv run earnings --help
```

Run with SEC source (default):

```bash
uv run earnings --source sec --ticker MSFT
uv run earnings --source sec --ticker AMD
uv run earnings --source sec --cik 0000320193
```

Run with IR source:

```bash
uv run earnings --source ir --ticker MSFT --ir-url "https://www.microsoft.com/en-us/investor/earnings/fy-2026-q3/press-release-webcast"
```

Multiple companies in one run:

```bash
uv run earnings --ticker AAPL MSFT GOOGL
uv run earnings --cik 0000320193 0000789019
```

## Tests

Run all tests:

```bash
uv run pytest -q
```

Run focused tests:

```bash
uv run pytest tests/test_extract_financial_metrics.py -q
uv run pytest tests/test_company_registry.py -q
```

## Notes

- Extracted metric keys are kept as-is from company documents.
- Results are upserted into MongoDB with keys like `TICKER_YEAR_latest`.



