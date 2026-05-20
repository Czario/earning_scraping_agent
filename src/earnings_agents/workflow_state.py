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
