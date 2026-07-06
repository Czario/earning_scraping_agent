from __future__ import annotations

from typing import Optional

from typing_extensions import NotRequired, TypedDict


class EarningsAgentState(TypedDict):
    ticker: str
    company_name: str
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
    # Per-pass skill-effectiveness records appended by analyze_metrics_node on
    # each re-extract loop: {"to_attempt": int, "deltas": [...]}. Pure
    # observability (ADR-0006) — shows which skills' findings were resolved
    # between passes; never influences routing.
    skill_effectiveness: NotRequired[Optional[list]]
    # ── normalize_data targeted extraction ──────────────────────────────────
    # Populated by load_company_concepts_node when EARNINGS_SAVE_TARGET=normalize_data.
    # Empty list (not None) means the node ran but the company was not found,
    # triggering the generic extraction path.
    company_cik: NotRequired[Optional[str]]
    target_concepts: NotRequired[Optional[list]]    # concept dicts from normalized_concepts_quarterly
    # concept_id strings (subset of target_concepts) that had a value in the
    # last N stored periods. Used to prune the extraction prompt to concepts the
    # company actually reports. Empty/None means no pruning (bootstrap / disabled).
    recent_concept_ids: NotRequired[Optional[list[str]]]
    calculated_concepts: NotRequired[Optional[list]]  # system:/calculated concept dicts for derivation
    concept_metrics: NotRequired[Optional[dict]]    # concept_id → float for normalize_data upsert
    derived_concept_ids: NotRequired[Optional[list[str]]]  # concept_ids filled by Tier-3 derivation
    fiscal_year_end_month: NotRequired[Optional[int]]
    fiscal_year_end_code: NotRequired[Optional[str]]  # raw MMDD string, e.g. "0130" or "1231"
    # Keys in metrics{} that were successfully matched to a concept_id during
    # targeted extraction (Tier 1 exact/normalised label match or Tier 2 LLM
    # semantic match).  Populated by extract_financial_metrics_node; consumed
    # by cleanup_metrics_node as a protected set that the LLM cannot remove.
    mapped_metric_keys: NotRequired[Optional[list[str]]]
    # Labels of target_concepts that had no value mapped after all tiers.
    # Stored by extract_financial_metrics_node; consumed by analyze_metrics_node
    # to generate targeted retry hints that tell the LLM exactly what to find.
    missing_concept_labels: NotRequired[Optional[list[str]]]   # all unmapped
    missing_segment_labels: NotRequired[Optional[list[str]]]   # dimensional (|) only
    missing_toplevel_labels: NotRequired[Optional[list[str]]]  # non-dimensional only
    # Labels of target_concepts that had no value mapped after all tiers.
    # Stored by extract_financial_metrics_node; consumed by analyze_metrics_node
    # to generate targeted retry hints that tell the LLM exactly what to find.
    missing_concept_labels: NotRequired[Optional[list[str]]]   # all unmapped
    missing_segment_labels: NotRequired[Optional[list[str]]]   # dimensional only
    missing_toplevel_labels: NotRequired[Optional[list[str]]]  # non-dimensional only
    # ── SEC-derived reporting period ────────────────────────────────────────
    # ``reportDate`` from the EDGAR submissions API — the period-end date for
    # the filing as declared to the SEC ("YYYY-MM-DD" string).  Set by the
    # CLI SEC path; absent (None) on the IR path.  Used as the authoritative
    # end date for normalize_data upserts and for the extraction period hint.
    sec_report_date: NotRequired[Optional[str]]
    # ── Table-aware HTML extraction ─────────────────────────────────────────
    # Populated by extract_html_text_node when GAAP tables are classified.
    # Maps section type ('income_statement', 'balance_sheet', 'cash_flow',
    # 'other', 'non_gaap') to a list of markdown-rendered table entries.
    # When present, extract_financial_metrics_node feeds one LLM call per
    # GAAP table instead of char-based chunking — eliminates chunk divergence
    # on numeric values that straddle char boundaries.
    raw_sections: NotRequired[Optional[dict]]
    # Per-metric chunk provenance from the most recent extraction pass.
    # Maps metric key → list of 0-based chunk indices (into the ordered chunk
    # list) that reported a non-null value for that key.  Used by the scoped
    # retry logic to re-run only the specific chunk(s) that contributed a
    # problematic metric, rather than all chunks in a section.
    chunk_metric_sources: NotRequired[Optional[dict]]  # str → list[int]
    # Per-metric verbatim source snippets ("show me" verification evidence).
    # Maps metric key → the exact text the extraction LLM read the value from
    # (the ``__sources__`` field of the LLM response, merged across chunks).
    # Consumed by ``check_source_grounding`` in analyze_metrics_node to flag
    # values that cannot be grounded in the source document.
    metric_source_snippets: NotRequired[Optional[dict]]  # str → str
    # ── Period type (annual vs quarterly) ───────────────────────────────────
    # Inferred at concept-load time from sec_report_date + fiscal_year_end_month.
    # ``"annual"`` when the report period ends in the fiscal year-end month;
    # ``"quarterly"`` otherwise (default).  Drives which normalized_concepts_*
    # collection is queried for targeted extraction.
    detected_period_type: NotRequired[Optional[str]]  # "annual" | "quarterly"
