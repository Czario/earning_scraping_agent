from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


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
