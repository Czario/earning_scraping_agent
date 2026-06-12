# Earnings Agents

A LangGraph-based pipeline that scrapes earnings releases (SEC EDGAR or company IR pages), extracts financial metrics via an LLM, and upserts results into MongoDB.

---

## Language

**EarningsAgentState**:
The single shared dictionary (`TypedDict`) that flows through every node in the LangGraph graph. It is the only way nodes communicate — nodes read from it and return a partial dict to merge back.
_Avoid_: state dict, pipeline state, graph state

**Node**:
A Python function with signature `(state: EarningsAgentState) -> EarningsAgentState` that performs one bounded step of the pipeline. Registered in `workflow.py` via `graph.add_node`.
_Avoid_: step, stage, handler, processor

**Routing helper**:
A function in `workflow.py` that inspects `EarningsAgentState` and returns a string literal naming the next node (or `"__end__"`). It is the only mechanism that controls graph branching.
_Avoid_: condition, branch function, decision node

**Status**:
The `status` field in `EarningsAgentState`. Valid values: `pending`, `discovered`, `fetched`, `text_extracted`, `extracted`, `saved`, `failed`. Progression is strictly linear; no node may skip a stage or set a prior-stage value.
_Avoid_: state, pipeline status, run status

**MongoDB document status**:
The `status` field stored in the MongoDB document (distinct from pipeline `status`). Valid values: `success`, `degraded`, `failed`. `degraded` means unresolved Tier-1 findings remained after all extraction attempts.
_Avoid_: document state, save status

**Finding**:
A structured observation produced by `analyze_metrics_node` about extracted metrics. Has `type` (FindingType), `severity`, `message`, `keys`, and optional `evidence`. The set of findings drives the re-extract loop and cleanup decisions.
_Avoid_: issue, warning, error (use severity instead)

**Severity**:
The routing weight of a `Finding`. `high` triggers a re-extract loop. `medium` adds a hint to `extraction_notes` but does not loop alone. `low` is handled deterministically (cleanup) or logged only.
_Avoid_: priority, level, importance

**FindingType**:
The closed set of finding categories: `missing_critical`, `missing_expected`, `case_duplicate`, `identity_violation`, `sign_anomaly`, `suspect_round`, `suspect_value`, `gaap_nongaap_leakage`, `composite_key`, `auto_corrected`, `section_mismatch`. The `suspect_value` type also covers income-statement ordering violations (Operating income > Gross profit; Diluted EPS > Basic EPS).
_Avoid_: error type, warning type

**Tier-1 metric**:
A core income statement line that must appear in every earnings release. Missing Tier-1 metrics produce `missing_critical` findings with `high` severity, triggering re-extraction. Defined in `analysis/critical_metrics.py` → `TIER1_REGISTRY`. Current members: Total Revenue, Gross Profit, Operating Income, Net Income, Diluted EPS.
_Avoid_: required metric, critical field, mandatory metric

**Tier-2 metric**:
A supporting income statement line that should appear but whose absence does not force re-extraction. Missing Tier-2 metrics produce `missing_expected` findings with `medium` severity. Defined in `TIER2_REGISTRY`. Includes: Cost of Revenue, Total Operating Expenses, Pre-tax Income, Income Tax Expense, Basic EPS, Weighted Avg Shares Diluted, Weighted Avg Shares Basic, Interest Expense.
_Avoid_: secondary metric, optional metric (use Tier-3 for truly optional)

**Tier-3 metric**:
Supplemental income statement lines that are tracked when present but never trigger any action on absence. Defined in `TIER3_REGISTRY`. Includes R&D, Sales & Marketing, G&A, Comprehensive Income, Effective Tax Rate, Depreciation & Amortization, Stock-Based Compensation, EBITDA, Dividends per Share.
_Avoid_: optional metric, bonus metric

**`__scale__`**:
A special key the LLM sets per extraction chunk to indicate the unit of all numeric values: `millions`, `thousands`, `billions`, or `as-is`. Applied in Python after chunk merging. Must never be treated as a regular metric key.
_Avoid_: scale factor, unit key, magnitude

**needs_reextract**:
A boolean field in `EarningsAgentState` set by `analyze_metrics_node`. When `True`, the routing helper in `workflow.py` loops back to `extract_financial_metrics_node`. This is the sole routing signal for the extract↔analyze loop — `status` is never used for routing.
_Avoid_: retry flag, loop flag, rerun signal

**extraction_attempts**:
A counter in `EarningsAgentState` incremented before each extraction pass. Capped at `MAX_EXTRACTION_ATTEMPTS` (default 3, env-overridable). When the cap is reached, `analyze_metrics_node` sets `needs_reextract = False` regardless of finding severity.
_Avoid_: retry count, attempt number

**extraction_notes**:
A string field in `EarningsAgentState` written by `analyze_metrics_node` to carry hints for the next extraction pass. Consumed at the start of `extract_financial_metrics_node` and injected into the LLM prompt. Cleared between passes.
_Avoid_: hints, reflection output, feedback

**previous_high_finding_keys**:
A snapshot of high-severity finding messages from the previous analysis pass. Used by `analyze_metrics_node` to detect no-progress loops: if the same high findings recur across consecutive passes, the loop breaks early rather than burning remaining attempts.
_Avoid_: last findings, previous errors

**identity_warning**:
A message produced when an accounting identity check fails (e.g. Gross Profit ≠ Revenue − COGS). Stored in `identity_warnings` list in state. When `STRICT_ACCURACY=true`, the save node refuses to upsert a document with non-empty `identity_warnings`.
_Avoid_: balance check error, accounting error

**document ID**:
The MongoDB `_id` for every upserted document. Format: `{TICKER}_{YEAR}_latest` where `TICKER` is uppercase and `YEAR` is the 4-digit fiscal year (e.g. `MSFT_2024_latest`). Re-runs overwrite the previous document for the same ticker/year.
_Avoid_: record ID, MongoDB key, document key

**source**:
The data origin for a pipeline run: `sec` (SEC EDGAR — 8-K / Exhibit 99.1) or `ir` (company investor relations page). Determines which discovery path runs in `discover_earnings_release_node`.
_Avoid_: mode, provider, origin

**raw_sections**:
A dict populated by `extract_html_text_node` when GAAP tables are classified. Maps section type (`income_statement`, `balance_sheet`, `cash_flow`, `other`, `non_gaap`) to lists of markdown-rendered table entries. When present, `extract_financial_metrics_node` runs one LLM call per GAAP table instead of character-based chunking.
_Avoid_: table sections, parsed tables, HTML sections

**cleanup pass**:
The two-phase process in `cleanup_metrics_node`: first a deterministic case-deduplication using `findings`; then an optional LLM pass to remove Rule-A/B/C duplicates. Neither pass may rename or edit the text of retained metric keys.
_Avoid_: dedup, normalization, metric cleaning

**company hint**:
A Markdown file in `data/company_hints/<TICKER>.md` containing company-specific extraction guidance (e.g. known non-standard metric names, reporting quirks). Injected into the LLM prompt when present.
_Avoid_: company config, extraction config, ticker hints

**concept**:
A `normalize_data` platform entity representing a standardized financial metric. When `EARNINGS_SAVE_TARGET=normalize_data`, `load_company_concepts_node` fetches the company's target concept list and `extract_financial_metrics_node` attempts to map extracted keys to concept IDs.
_Avoid_: standard metric, normalized metric, concept ID (use concept_id for the ID specifically)

**degraded**:
A MongoDB document status (not pipeline status) meaning the pipeline completed all extraction attempts but Tier-1 findings remained unresolved. The document is saved with partial data rather than discarded.
_Avoid_: partial success, incomplete, failed (use `failed` only when the pipeline aborted with `status = "failed"`)
