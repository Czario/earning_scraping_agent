from __future__ import annotations

from typing import Optional

from typing_extensions import NotRequired, TypedDict


class EarningsAgentState(TypedDict):
    ticker: str
    company_name: str
    ir_url: str
    discovered_file_url: Optional[str]
    file_type: Optional[str]   # "pdf" | "html"
    raw_text: Optional[str]
    metrics: Optional[dict]    # serialised EarningsMetrics
    error: Optional[str]
    # pending → discovered → fetched → text_extracted → extracted → saved | failed
    status: str
    # Agentic loop fields
    extraction_attempts: int          # incremented before each extraction pass; caps retries
    extraction_notes: Optional[str]   # reflection output: hints for the next extraction pass
    # Routing signal emitted by analyze_metrics_node. True = loop back to
    # extract_financial_metrics. Using a dedicated field avoids overloading the
    # status field as a routing signal.
    needs_reextract: bool
    # Snapshot of high-severity finding messages from the previous analysis pass.
    # Used by analyze_metrics_node to detect no-progress loops (same findings
    # across consecutive passes → break early rather than burn remaining attempts).
    previous_high_finding_keys: Optional[list]
    # Populated by _validate_metrics when accounting identities don't reconcile.
    # The save node refuses to upsert when this is non-empty and STRICT_ACCURACY is on.
    identity_warnings: Optional[list]
    # Keys dropped by the LLM cleanup pass (cleanup_metrics_node). Informational.
    cleanup_removed: Optional[list]
    # Structured Finding.to_dict() entries produced by analyze_metrics_node.
    # Drives the re-extract loop and is consumed by cleanup_metrics for
    # deterministic case-duplicate removal.
    findings: Optional[list]
    # ── normalize_data targeted extraction ──────────────────────────────────
    # Populated by load_company_concepts_node when EARNINGS_SAVE_TARGET=normalize_data.
    # Empty list (not None) means the node ran but the company was not found,
    # triggering the generic extraction path.
    company_cik: NotRequired[Optional[str]]
    target_concepts: NotRequired[Optional[list]]    # concept dicts from normalized_concepts_quarterly
    concept_metrics: NotRequired[Optional[dict]]    # concept_id → float for normalize_data upsert
    fiscal_year_end_month: NotRequired[Optional[int]]
    fiscal_year_end_code: NotRequired[Optional[str]]  # raw MMDD string, e.g. "0130" or "1231"
    # ── Table-aware HTML extraction ─────────────────────────────────────────
    # Populated by extract_html_text_node when GAAP tables are classified.
    # Maps section type ('income_statement', 'balance_sheet', 'cash_flow',
    # 'other', 'non_gaap') to a list of markdown-rendered table entries.
    # When present, extract_financial_metrics_node feeds one LLM call per
    # GAAP table instead of char-based chunking — eliminates chunk divergence
    # on numeric values that straddle char boundaries.
    raw_sections: NotRequired[Optional[dict]]
