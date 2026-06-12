# Project Flow

End-to-end trace of the earnings extraction pipeline as currently implemented
in [src/earnings_agents/workflow.py](src/earnings_agents/workflow.py).

The graph is the same for SEC and IR sources; only the *discovery* node behaves
differently. SEC mode pre-resolves the filing URL in the CLI and short-circuits
discovery; IR mode performs discovery inside the graph.

---

## Graph topology

```
discover_earnings_release
        │
        ▼
load_company_concepts        ← no-op unless EARNINGS_SAVE_TARGET=normalize_data
        │
        ▼
detect_document_type
        │
   ┌────┴────┐
   ▼         ▼
extract_pdf  extract_html         ← html path also classifies GAAP tables
   │         │
   └────┬────┘
        ▼
extract_financial_metrics  ◄─────┐
        │                        │
        ▼                        │
analyze_metrics  ────────────────┘   (loops while high-severity findings &
        │                              extraction_attempts < MAX_EXTRACTION_ATTEMPTS=3)
        ▼
cleanup_metrics            ← deterministic case-dedup + constrained LLM pass
        │
        ▼
mongodb_save               ← upsert as {TICKER}_{YEAR}_latest
```

All routing is implemented by the `_route_*` helpers in
[workflow.py](src/earnings_agents/workflow.py). Any node that sets
`status="failed"` short-circuits to `END`.

---

## 1. CLI input and company resolution

Entry point: [src/earnings_agents/cli/earnings.py](src/earnings_agents/cli/earnings.py).

- Accepts `--ticker`, `--cik`, `--source {sec|ir}`, optional `--ir-url`.
- Ticker/CIK resolution goes through
  [company_registry.py](src/earnings_agents/company_registry.py), backed by
  [data/reference/tickers.json](data/reference/tickers.json).
- Multi-ticker runs execute in parallel via `ThreadPoolExecutor`.
- A rich progress UI tracks per-company stage transitions through the
  hooks system ([hooks.py](src/earnings_agents/hooks.py)).

### SEC mode (default)
- The CLI calls `get_latest_earnings_url(cik)` from
  [tools/edgar_client.py](src/earnings_agents/tools/edgar_client.py) and
  pre-fills `discovered_file_url` on the initial state.
- Initial `status="discovered"` so the graph skips IR discovery.

### IR mode
- CLI leaves `discovered_file_url=None` and provides `ir_url`.
- The `discover_earnings_release` node does the LLM-driven URL search.

---

## 2. EDGAR filing URL discovery (SEC mode only)

[tools/edgar_client.py](src/earnings_agents/tools/edgar_client.py):

1. Fetch the SEC submissions JSON for the CIK.
2. Find the latest 8-K with Item 2.02 (Results of Operations).
3. Open the filing index and locate `EX-99.1` (the earnings release exhibit).
4. Fall back to the primary document when EX-99.1 is missing.

Uses the SEC-compliant `User-Agent` and never relies on JS rendering.

---

## 3. `discover_earnings_release_node`

[nodes/discover_earnings_release.py](src/earnings_agents/nodes/discover_earnings_release.py).

- Short-circuits when `discovered_file_url` is already set (SEC mode).
- Otherwise fetches the IR page via
  [tools/static_scraper.py](src/earnings_agents/tools/static_scraper.py),
  falls back to Playwright via
  [tools/playwright_scraper.py](src/earnings_agents/tools/playwright_scraper.py)
  if the static response is too small.
- Extracts anchor links and asks the LLM to identify the earnings release URL.
- On success: sets `discovered_file_url` and `status="discovered"`.

---

## 4. `load_company_concepts_node`

[nodes/load_company_concepts_node.py](src/earnings_agents/nodes/load_company_concepts_node.py).

- Targeted extraction requires stored concepts from `normalize_data`:
  1. Looks up the company by ticker in `normalize_data.companies`.
  2. Pulls income-statement concepts from
     `normalize_data.normalized_concepts_quarterly`.
  3. Populates `company_cik`, `target_concepts`,
     `fiscal_year_end_month`, `fiscal_year_end_code`.
- **Skips the run** (`status="skipped"` + clear `error`) when the company is
  absent from `normalize_data`, the DB lookup/concept query fails, or no
  income-statement concepts are stored — we don't have historical data for the
  company so we can't proceed. Generic extraction has been removed; there is no
  fallback path. The router `_route_after_concepts` ends the run on a skip.

---

## 5. `detect_document_type_node`

[nodes/detect_document_type.py](src/earnings_agents/nodes/detect_document_type.py).

- Extension-first (`.htm`/`.html`/`.pdf`), HEAD-request fallback otherwise.
- Sets `state["file_type"]` and routes to `extract_html_text` or
  `extract_pdf_text` via `_route_by_file_type`.

---

## 6. Raw text extraction

### `extract_html_text_node`
[nodes/extract_html_text.py](src/earnings_agents/nodes/extract_html_text.py).

- SEC-compliant `User-Agent` for `sec.gov` URLs.
- Strips SGML wrappers from EDGAR archive responses.
- Playwright JS fallback is **disabled** for `sec.gov` (filings are static HTML).
- **Table classification**: tables are sorted into
  `income_statement`, `balance_sheet`, `cash_flow`, `non_gaap`, `other`
  using regex matchers over each table's preceding-sibling context and body
  text. The non-GAAP probe runs first so reconciliation tables sharing GAAP
  metric names ("Net income as reported") are diverted correctly.
- The classified tables are placed on
  `state["raw_sections"]` as a dict of section → list of markdown tables.
  When this is present, `extract_financial_metrics` issues one LLM call per
  GAAP table instead of char-window chunking — eliminating numeric splits
  across chunk boundaries.

### `extract_pdf_text_node`
[nodes/extract_pdf_text.py](src/earnings_agents/nodes/extract_pdf_text.py).

- Downloads the PDF, extracts text, sets `state["raw_text"]`. PDFs always
  go through the char-window chunk path (no table classification).

---

## 7. `extract_financial_metrics_node`

[nodes/extract_financial_metrics.py](src/earnings_agents/nodes/extract_financial_metrics.py).

Targeted extraction prompt (`target_concepts` present): supplies the concrete
list of `"Label" (GAAP: LocalName)` strings and instructs the LLM to use those
label strings verbatim as keys, enabling lossless mapping to `concept_id`
for `normalize_data` upserts. (Generic extraction has been removed; a run that
reaches this node without `target_concepts` fails defensively — in production
`load_company_concepts` has already skipped it.)

The prompt requires two leading fields:

- `__scale__` ∈ `{"millions","thousands","billions","as-is"}`.
- `__period__` — the most recent period column label exactly as printed.

Per-call mechanics:

- Each pass increments `extraction_attempts`.
- The optional `extraction_notes` hint block (produced by `analyze_metrics`)
  is injected into the prompt to focus the next pass on missing metrics.
- Inputs are either (a) one chunk per classified GAAP table from
  `raw_sections`, or (b) char-window chunks of `raw_text` with overlap.
- A process-wide `Semaphore(OLLAMA_CONCURRENCY)` throttles concurrent
  Ollama calls.
- Per-chunk JSON is parsed, `__scale__` is applied in Python after merging,
  and `_validate_metrics` runs accounting sanity checks (populating
  `identity_warnings`).
- Special-cased keys: EPS / per-share / percentage / share-count / physical
  unit metrics are never multiplied by the table scale.

---

## 8. `analyze_metrics_node` (the loop controller)

[nodes/analyze_metrics.py](src/earnings_agents/nodes/analyze_metrics.py).

Runs every deterministic checker from the **skill catalog**
([analysis/skills.py](src/earnings_agents/analysis/skills.py) — `iter_detectors()`).
Each checker is defined in [analysis/findings.py](src/earnings_agents/analysis/findings.py)
and paired in the catalog with stable metadata and curated remediation guidance:

- `check_presence` — tiered coverage against `TIER1/2/3_REGISTRY` (special-invocation).
- `check_case_duplicates`
- `check_composite_keys`
- `check_gaap_nongaap_leakage`
- `check_gross_profit_identity`
- `check_operating_vs_gross`
- `check_eps_dilution_ordering`
- `check_suspect_round`
- `check_opex_label_collision`
- `check_source_grounding` (special-invocation)

Browse the catalog: `uv run earnings-skills`.

Deterministic auto-correction:

- When the OPEX/Operating-income collision is detected, the node calls
  `derive_corrected_total_opex` and replaces the bad value with
  `Cost of revenue + Operating expenses`.

Output:

- `state["findings"]` — list of `Finding.to_dict()` entries.
- `state["needs_reextract"]` — routing flag (replaces the deleted
  `reflect_metrics` node).
- `state["extraction_notes"]` — cumulative hint block for the next pass
  (prepends the previous attempt's notes so the LLM keeps context).
- `state["previous_high_finding_keys"]` — snapshot used to detect
  no-progress loops (identical high-severity messages between consecutive
  passes break the loop early).

Loop policy:

- Loops back to `extract_financial_metrics` only when **all** of the
  following hold: at least one `high`-severity finding,
  `extraction_attempts < MAX_EXTRACTION_ATTEMPTS (=3)`, and the new
  high-severity messages differ from the previous pass.
- **Targeted (normalize_data) mode tweak**: `missing_critical` findings are
  demoted from `high` to `medium` because the truth set is `target_concepts`,
  not the hard-coded TIER1 registry — a company that genuinely doesn't
  report a TIER1 metric (e.g. BJ Wholesale Club has no standalone Gross
  Profit) must not loop indefinitely.

---

## 9. `cleanup_metrics_node`

[nodes/cleanup_metrics.py](src/earnings_agents/nodes/cleanup_metrics.py).

Two-phase, constrained, **drop-only** cleanup:

1. **Deterministic case-dedup pass**: removes obvious case duplicates and
   structural artifacts. Driven by the `findings` produced upstream.
2. **Optional LLM pass** (`CLEANUP_METRICS=1`, default on): the LLM can
   only return a `remove` set, which is enforced to be a subset of the
   current keys. Surviving values are kept byte-for-byte. The cleanup is
   rejected if `_validate_metrics` raises a new blocking warning on the
   cleaned dict.
   - Metric values are encoded in a compact form (`"82.9B"`, `"315M"`,
     `"24.3K"`) for prompt efficiency; the LLM operates on original keys.
   - A `_needs_cleanup` pre-check skips the LLM call when the metrics look
     obviously clean (no near-equal value pairs, no case duplicates, no
     implausible per-share values).

Cleanup never renames keys, never edits values, never adds keys. Dropped
keys go to `state["cleanup_removed"]` for visibility.

---

## 10. `mongodb_save_node`

Defined in [workflow.py](src/earnings_agents/workflow.py).

- `_id = "{TICKER}_{YEAR}_latest"` where `YEAR` is parsed from the
  extracted `__period__` label (fallback: UTC current year).
- Refuses to save when `identity_warnings` is non-empty AND
  `STRICT_ACCURACY=1` (the default).
- Saves with `status="degraded"` instead of `"success"` when any
  unresolved `high`-severity finding remains after the loop. Saves
  `findings`, `identity_warnings`, and `unresolved_findings` as document
  fields when present.

### Optional `normalize_data` upsert
When `EARNINGS_SAVE_TARGET=normalize_data`, after the primary upsert the
node also calls `upsert_concept_values` from
[tools/normalize_data_client.py](src/earnings_agents/tools/normalize_data_client.py)
to record `concept_id → value` entries keyed by company CIK, period,
fiscal-year-end month, and fiscal-year-end code.

---

## State progression cheat-sheet

`pending → discovered → fetched → text_extracted → extracted → saved | failed`

Routing-specific state fields (not part of the linear status):

| Field | Set by | Consumed by |
|---|---|---|
| `needs_reextract` | `analyze_metrics` | `_route_after_analysis` |
| `extraction_notes` | `analyze_metrics` | `extract_financial_metrics` |
| `previous_high_finding_keys` | `analyze_metrics` | `analyze_metrics` (next pass) |
| `findings` | `analyze_metrics` | `cleanup_metrics`, save doc |
| `identity_warnings` | `extract_financial_metrics` | `mongodb_save` (gate) |
| `cleanup_removed` | `cleanup_metrics` | informational |
| `raw_sections` | `extract_html_text` | `extract_financial_metrics` |
| `target_concepts` / `company_cik` / `fiscal_year_end_*` | `load_company_concepts` | `extract_financial_metrics`, `mongodb_save` |

The document `status` field written to MongoDB is one of `"success"`,
`"degraded"` (high-severity findings unresolved after the loop), or
`"failed"` (only when the save itself fails or `STRICT_ACCURACY` blocks
the save).

---

## Key configuration knobs

From [config.py](src/earnings_agents/config.py):

- `LLM_PROVIDER` — `ollama` (default) | `openai` | `groq`.
- `OLLAMA_*`, `OPENAI_*`, `GROQ_*` — provider credentials, model, timeouts.
- `OLLAMA_CONCURRENCY` — global cap on concurrent LLM calls.
- `MONGODB_URI` / `MONGODB_DB` / `MONGODB_COLLECTION`.
- `EARNINGS_SAVE_TARGET` — `earnings_db` | `normalize_data`.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EXTRACTION_MAX_CHARS`.
- `MAX_EXTRACTION_ATTEMPTS` (default 3).
- `STRICT_ACCURACY` (default on) — blocks save on identity failures.
- `CLEANUP_METRICS` (default on) — enables the constrained LLM cleanup pass.
