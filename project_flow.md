# Earnings Pipeline — Project Flow

End-to-end trace of the SEC EDGAR path for one ticker.  
**Example throughout: `SOFI` Q1 2026 (period end 2026-03-31, "in thousands" filing).**

---

## Pipeline overview

```
CLI  →  load_company_concepts_node
     →  detect_document_type_node
     →  extract_html_text_node
     →  extract_financial_metrics_node   ← 3 LLM calls happen here
     →  analyze_metrics_node
     →  cleanup_metrics_node
     →  mongodb_save_node
```

---

## Node 1 — `load_company_concepts_node`

Queries `normalize_data.normalized_concepts_quarterly` for the ticker's
income-statement concepts.  No LLM call.

**Output stored in state:**
```python
state["target_concepts"] = [
    {"_id": "6a4b4f2ac5b985524d4e7111", "label": "Earnings Per Share, Basic",
     "taxonomy_key": "us-gaap:EarningsPerShareBasic",
     "concept": "us-gaap:EarningsPerShareBasic", "path": "009"},
    {"_id": "6a4b4f2ac5b985524d4e7118", "label": "Total net revenue",
     "taxonomy_key": "us-gaap:RevenuesNetOfInterestExpense",
     "concept": "us-gaap:RevenuesNetOfInterestExpense", "path": "001"},
    # ... 43 more concepts (45 total for SOFI)
]
```

---

## Node 2 — `detect_document_type_node`

Inspects the URL extension / HTTP Content-Type.  No LLM call.  
Sets `state["file_type"] = "html"` for SoFi's Exhibit 99.1.

---

## Node 3 — `extract_html_text_node`

Downloads and parses the HTML press release.  No LLM call.  
- Strips SGML wrappers (SEC EDGAR).  
- Identifies GAAP tables (income statement, balance sheet, cash flow).  
- Prepends scale captions like `"(in thousands)"` to each table chunk.  
- Populates `state["raw_text"]` and `state["raw_sections"]`.

**Output (abbreviated):**
```
state["raw_sections"] = {
    "income_statement": "=== GAAP INCOME STATEMENT ===\n"
                        "(in thousands)\n"
                        "Total net revenue     771,000\n"
                        "Net interest income   425,000\n"
                        "...",
    "balance_sheet":    "=== GAAP BALANCE SHEET ===\n...",
    ...
}
```

---

## Node 4 — `extract_financial_metrics_node`

This node makes **up to 3 LLM calls**. Two run sequentially on the critical
path; one runs in a background thread in parallel with the main extraction.

```
[roles LLM]  launched in background thread ─────────────────────────────────┐
[extraction LLM]  ~45s (critical path)                                       │
                 ↓                                                           │
[tier2 LLM]  ~25s sequential after extraction (critical path)               │
                 ↓                                                           │
[roles result collected]  instant — thread already finished ←───────────────┘
[derivation]  instant pure Python
```

### Step 4a — Document prescan (no LLM)

Scans `raw_text` for the scale caption before any LLM call.

```
Input:  "(in thousands, except per share data)"  ← found in the HTML
Output: doc_scale = "thousands"
        dollar_multiplier = 1_000
        shares_multiplier = 1_000   (no separate share scale declared)
```

---

### LLM Call 1 — `[roles]` (background thread)

**Purpose:** Assign a standard financial role (e.g., `revenue`, `net_income`,
`eps_basic`) to each concept label that the regex patterns in
`calculators.py` could not classify.  Launched as a background thread
immediately before the extraction call so its latency is hidden.

**When:** Concepts whose label has no regex match in `_ROLE_PATTERNS`.

**Prompt template** (`_LLM_ROLE_PROMPT` in `tools/llm_concept_mapper.py`):
```
You are a financial metric role classifier.

Map each label below to the closest standard financial metric role.
Only assign a role when you are confident — return null if unsure.

Known roles and their meaning:
  revenue              — total net revenue or sales
  cost_of_revenue      — cost of goods/products/services sold
  gross_profit         — gross profit (revenue minus cost of revenue)
  rd_expense           — research and development expense
  sm_expense           — sales and marketing expense
  ga_expense           — general and administrative expense
  total_opex           — total operating expenses or costs
  operating_income     — income/profit/loss from operations
  interest_income      — interest and other income
  interest_expense     — interest expense
  other_income_net     — other non-operating income or expense (net)
  pretax_income        — income before income taxes
  tax_expense          — income tax expense / provision for taxes
  net_income           — net income, net earnings, or net loss
  eps_basic            — basic earnings per share
  eps_diluted          — diluted earnings per share
  shares_basic         — basic weighted-average shares outstanding
  shares_diluted       — diluted weighted-average shares outstanding
  gross_margin_pct     — gross profit margin percentage
  operating_margin_pct — operating income margin percentage
  net_margin_pct       — net income margin percentage

Labels to classify:
  - "Loans and securitizations"
  - "Securitizations"
  - "Related party notes"
  - "Other interest income"
  - "Securitizations and warehouses"
  - "Deposits"
  - "Corporate borrowings"
  - "Other interest expense"
  - "Loan origination, sales, securitizations and servicing"
  ... (remaining unrecognized labels)

Return ONLY a JSON object mapping each label to its role (or null):
{"<label>": "<role_or_null>", ...}
```

**Real SoFi input — 24 unrecognized labels (after today's banking patterns fix,
labels like "Net interest income" and "Technology and product development" are
now matched by regex and no longer sent to the LLM):**
```json
Labels sent to roles LLM:
  "Loans and securitizations", "Securitizations",
  "Related party notes", "Securitizations and warehouses",
  "Deposits", "Corporate borrowings",
  "Loan origination, sales, securitizations and servicing",
  "Loan platform fees", "Net crypto transaction revenue",
  "Credit Card", "Commercial and consumer banking",
  ... (highly specific banking sub-items)
```

**LLM response (SoFi example):**
```json
{
  "Total interest income":              "interest_income",
  "Other interest income":              "interest_income",
  "Other noninterest income":           "other_income_net",
  "Technology and product development": "rd_expense",
  "Loans and securitizations":          null,
  "Securitizations":                    null,
  "Deposits":                           null,
  "Credit Card":                        null
}
```

**Post-processing:**
- Null values and unknown roles are discarded.
- Valid results stored as `role_overrides = {concept_id → role}`.
- Used in Step 4d (derivation) to compute EPS, margins, etc.

---

### LLM Call 2 — `[llm]` extraction (critical path, ~45s)

**Purpose:** Extract numeric values for all target concepts from the raw text.

**Prompt template** (`_TARGETED_PROMPT_TEMPLATE` in
`nodes/extract_financial_metrics.py`):
```
You are a financial data extraction assistant.

Extract ONLY the income statement metrics listed below from the text excerpt
for {company_name} ({ticker}).
This is chunk {chunk_num} of {total_chunks} of the full document.

CONFIRMED SCALE: document header says "(In thousands)" —
set __scale__ = "thousands" for this chunk.

CONFIRMED PERIOD: most-recent period is "Three Months Ended March 31, 2026" —
extract ONLY the column for that period.

SCOPE — extract ONLY the concepts listed below.
Use the bracketed key [ ] as your JSON key when one is shown; otherwise use
the quoted label exactly.

  • "Total net revenue"                       [us-gaap:RevenuesNetOfInterestExpense]
  • "Net interest income"                     [us-gaap:InterestIncomeExpenseNet]
  • "Total interest income"                   [us-gaap:InterestIncomeOperating]
  • "Loans and securitizations"               [us-gaap:InterestAndFeeIncomeLoansAndLeases]
  • "Securitizations"                         [sofi:InterestIncomeSecuritizations]
  • "Technology and product development"      [us-gaap:ResearchAndDevelopmentExpense]
  • "Sales and marketing"                     [us-gaap:SellingAndMarketingExpense]
  • "General and administrative"              [us-gaap:GeneralAndAdministrativeExpense]
  • "Net income (loss)"                       [us-gaap:NetIncomeLoss]
  • "Earnings Per Share, Basic"               [us-gaap:EarningsPerShareBasic]
  • "Earnings Per Share, Diluted"             [us-gaap:EarningsPerShareDiluted]
  • "Weighted Average Number of Shares Outstanding, Basic"
                                              [us-gaap:WeightedAverageNumberOfSharesOutstandingBasic]
  ... (all 45 concepts)

IGNORE — do NOT extract:
  • Balance sheet items, cash flow items, non-GAAP metrics, guidance / forecasts.
  • Values from FOOTNOTES or parenthetical sub-tables.

PERIOD RULE — extract ONLY from the most-recent column ("Three Months Ended
March 31, 2026"). NEVER take a value from the prior-year comparison column.

INTERNAL CONSISTENCY:
  Revenue − Cost of revenue = Gross profit (verify before returning).

Return ONLY a flat JSON object. First field: "__scale__". Second: "__period__".
Last field: "__sources__" mapping each key to its verbatim source snippet.

Text excerpt:
"""
(in thousands)
                                Three Months Ended    Three Months Ended
                                  March 31, 2026        March 31, 2025
Total net revenue                    771,000               621,000
Net interest income                  425,000               318,000
Total interest income                594,000               460,000
  Loans and securitizations          523,000               405,000
  Securitizations                     36,000                30,000
  ...
Technology and product development   197,584               165,000
Sales and marketing                  138,255               115,000
General and administrative            68,812                58,000
Net income (loss)                    121,593                71,000
Earnings Per Share, Basic               0.13                  0.08
Earnings Per Share, Diluted             0.12                  0.07
...
"""
```

**LLM response (SoFi Q1 2026):**
```json
{
  "__scale__":   "thousands",
  "__period__":  "Three Months Ended March 31, 2026",

  "Total net revenue":                                   771000,
  "Net interest income":                                 425000,
  "Total interest income":                               594000,
  "Loans and securitizations":                           523000,
  "Securitizations":                                      36000,
  "Technology and product development":                  197584,
  "Sales and marketing":                                 138255,
  "General and administrative":                           68812,
  "Net income (loss)":                                   121593,
  "Earnings Per Share, Basic":                             0.13,
  "Earnings Per Share, Diluted":                           0.12,
  "Weighted Average Number of Shares Outstanding, Basic": 932184,

  "__sources__": {
    "Total net revenue":          "Total net revenue 771,000",
    "Earnings Per Share, Basic":  "Earnings Per Share, Basic 0.13",
    ...
  }
}
```

**Post-processing (Python, no LLM):**

1. `__scale__` popped → `multiplier = 1_000` (confirmed by prescan → overrides LLM).
2. `__sources__` popped → stored in `state["metric_source_snippets"]`.
3. Scale applied to each key:
   - `_PCT_OR_PER_SHARE_PATTERNS` checked per key.
   - `"Total net revenue"` → scaled: `771000 × 1000 = 771,000,000` ✓
   - `"Earnings Per Share, Basic"` → **NOT scaled** (matches `per share` pattern): stays `0.13` ✓
   - `"[us-gaap:EarningsPerShareBasic]"` → **NOT scaled** (matches `per\w*share` pattern): stays as-is ✓
4. `_merge_metrics` picks the highest-authority section value per key (income_statement wins over other).

---

### Step 4b — Tier-0 / Tier-1 mapping (no LLM, instant)

Maps each extracted key to a `concept_id` deterministically.

```
Tier 0  exact taxonomy_key match:
        "us-gaap:RevenuesNetOfInterestExpense" in metrics
            → concept_id "6a4b4f2ac5b985524d4e7118"  ✓

Tier 1a  exact label match:
        "Total net revenue" == concept.label
            → concept_id "6a4b4f2ac5b985524d4e7118"  ✓

Tier 1b  normalised label match (lowercase + collapsed whitespace):
        "total net revenue" == normalize("Total net revenue")
            → concept_id "6a4b4f2ac5b985524d4e7118"  ✓
```

**SoFi result:** 31 of 45 concepts matched → `concept_metrics = {concept_id: value, ...}`.  
The remaining **13 concepts** could not be matched by any label.

---

### LLM Call 3 — `[tier2]` semantic mapping (critical path, ~25s)

**Purpose:** For concepts that tier0/tier1 couldn't match (label in the filing
differs from the stored label), ask the LLM to find the closest semantic match
from the pool of already-extracted keys.

**Pre-filter applied first (no LLM):**
Removes concepts that can never appear in earnings press releases:
- OCI / comprehensive-income items (`"Comprehensive income (loss)"`,
  `"Unrealized gains on AFS securities"`, `"Foreign currency translation"`)
- Dimensional segment concepts (`|` in `taxonomy_key`)

SoFi: 13 unmapped → 9 matchable candidates after filter.

**Prompt template** (`_LLM_MAP_PROMPT` in `tools/llm_concept_mapper.py`):
```
You are a financial concept mapper.

Below are metric keys extracted from an earnings press release and a list of
target concepts (XBRL tag + display label + concept_id) that we want to map to.

For each target concept, decide which extracted key best matches it (if any).

Rules:
  1. Each extracted key may be assigned to AT MOST ONE concept.
  2. Only assign a key when you are confident — do NOT guess.
  3. If no extracted key fits a concept, return null for that concept.
  4. Do not invent new keys; only use keys from the extracted list.

Extracted metric keys:
  - "Total net revenue"
  - "Net interest income"
  - "Total interest income"
  - "Loans and securitizations"
  - "Technology and product development"
  - "Sales and marketing"
  - "Net income (loss)"
  - "Earnings Per Share, Basic"
  ... (all 44 numeric keys)

Target concepts:
  - concept_id: "6a4b4f2ac5b985524d4e71aa"  GAAP: sofi:CreditCardLoanPortfolioSegmentMember  label: "Credit Card"
  - concept_id: "6a4b4f2ac5b985524d4e71bb"  GAAP: sofi:CommercialAndConsumerBankingPortfolioSegmentMember  label: "Commercial and consumer banking"
  - concept_id: "6a4b4f2ac5b985524d4e71cc"  GAAP: us-gaap:GoodwillImpairmentLoss  label: "Goodwill impairment"
  - concept_id: "6a4b4f2ac5b985524d4e71dd"  GAAP: us-gaap:NoninterestIncome  label: "Total noninterest income"
  ... (9 matchable concepts)

Return ONLY a flat JSON object mapping concept_id -> matched extracted key (or null):
{"<concept_id>": "<extracted_key_or_null>", ...}
```

**LLM response (SoFi Q1 2026):**
```json
{
  "6a4b4f2ac5b985524d4e71aa": null,
  "6a4b4f2ac5b985524d4e71bb": null,
  "6a4b4f2ac5b985524d4e71cc": null,
  "6a4b4f2ac5b985524d4e71dd": null
}
```
All null — the concepts genuinely don't appear in the Q1 2026 press release
(segment breakdown of credit loss provision not reported; no goodwill impairment).

**Post-processing:** Null entries discarded. `concept_metrics` unchanged.

---

### Step 4c — Roles result collected (instant)

The background roles thread (started at step 4a) has been running for ~40s
during extraction + tier2.  Its result is collected immediately.

```
role_overrides = {
    "6a4b4f2ac5b985524d4e7109": "interest_income",   # "Total interest income"
    "6a4b4f2ac5b985524d4e7107": "interest_income",   # "Other interest income"
    "6a4b4f2ac5b985524d4e7108": "other_income_net",  # "Other noninterest income"
    "6a4b4f2ac5b985524d4e710a": "rd_expense",        # "Technology and product development"
}
```

---

### Step 4d — Derivation (no LLM, instant)

`derive_missing_concept_metrics` in `analysis/calculators.py` computes values
using accounting identities for any concept still absent from `concept_metrics`.

**SoFi example — 1 value derived:**
```
concept:  "Earnings Per Share, Basic and Diluted"  (us-gaap:EarningsPerShareBasicAndDiluted)
formula:  eps = net_income / shares_basic
inputs:   net_income  = 121,593,000  (from concept_metrics, role: net_income)
          shares_basic = 932,184,000  (from concept_metrics, role: shares_basic)
result:   121,593,000 / 932,184,000 = 0.1304  → stored as 0.13
```

**Final `concept_metrics` for SoFi Q1 2026:**  32 values (31 tier1 + 1 derived).

---

## Node 5 — `analyze_metrics_node`

Runs pure-Python QC checkers. No LLM call.

- **Presence check:** are all Tier-1 XBRL concepts present? (Revenue, Net Income, EPS, etc.)
- **Gross profit identity:** Revenue − Cost of revenue = Gross profit (SoFi is a bank, so Cost of revenue check is skipped when that concept is absent)
- **EPS dilution ordering:** Diluted EPS ≤ Basic EPS (when both positive)
- **Source grounding:** each metric's `__sources__` snippet must be findable in `raw_text`

If any **high-severity** finding and `extraction_attempts < 3` → sets
`needs_reextract = True` → loops back to Node 4 with `extraction_notes`
containing the specific error.

**SoFi Q1 2026:** No high-severity findings → `needs_reextract = False`.

---

## Node 6 — `cleanup_metrics_node`

Deduplicates metric keys (case-insensitive). No LLM call in the deterministic
pass; optional LLM pass removes Rule-A/B/C duplicates.

---

## Node 7 — `mongodb_save_node`

Upserts each entry in `concept_metrics` into `normalize_data.concept_values_quarterly`:

```python
# One document per concept per company per period
{
    "_id":          ObjectId("..."),
    "company_cik":  "0001818874",
    "concept_id":   ObjectId("6a4b4f2ac5b985524d4e7118"),
    "value":        771000000,        # 771,000 thousands = $771M
    "reporting_period": {
        "end_date":    datetime(2026, 3, 31),
        "period_date": "2026-03-31",
        "form_type":   "10-Q",
        "quarter":     1,
        "fiscal_year": 2026,
    },
    "earning_data": True,
    "statement_type": "income_statement",
    "created_at":  datetime(2026, 7, 6, 3, 44, 27),
}
```

Document ID format for the earnings pipeline legacy path: `{TICKER}_{YEAR}_latest`
(e.g. `SOFI_2026_latest`).

---

## LLM call summary table

| Call | Prompt size | Typical time | Parallel? | SoFi result |
|---|---|---|---|---|
| `[roles]` identify roles | ~1K tokens | ~8s | ✅ background thread | 4/24 mapped |
| `[llm]` extraction | ~8K tokens | ~45s | ❌ critical path | 44 keys extracted |
| `[tier2]` semantic mapping | ~3K tokens | ~25s | ❌ critical path (after extraction) | 0/9 mapped |

**Total wall time:** ~70s (extraction 45s + tier2 25s; roles hidden behind extraction).

---

## State fields touched at each stage

| Field | Set by | Read by |
|---|---|---|
| `target_concepts` | `load_company_concepts_node` | `extract_financial_metrics_node` |
| `raw_text`, `raw_sections` | `extract_html_text_node` | `extract_financial_metrics_node`, `analyze_metrics_node` |
| `metrics` | `extract_financial_metrics_node` | `analyze_metrics_node`, `cleanup_metrics_node` |
| `concept_metrics` | `extract_financial_metrics_node` | `mongodb_save_node` |
| `metric_source_snippets` | `extract_financial_metrics_node` | `analyze_metrics_node` (grounding check) |
| `findings` | `analyze_metrics_node` | `extract_financial_metrics_node` (retry hints), `cleanup_metrics_node` |
| `needs_reextract` | `analyze_metrics_node` | `workflow.py` routing |
| `extraction_notes` | `analyze_metrics_node` | `extract_financial_metrics_node` (retry prompt) |
| `extraction_attempts` | `extract_financial_metrics_node` | routing guard (max 3) |
| `identity_warnings` | `extract_financial_metrics_node` | `mongodb_save_node` (STRICT_ACCURACY gate) |
