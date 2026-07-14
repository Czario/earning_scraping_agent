"""Command-line interface for the earnings scraping pipeline.

Pass one or more company identifiers and the graph runs for each.

Usage examples:
    uv run earnings --cik 0000320193
    uv run earnings --ticker AAPL MSFT GOOGL
    uv run earnings --cik 0000320193 0000789019
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys

import requests
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from earnings_agents.config import (  # noqa: E402
    GEMINI_API_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from earnings_agents.nodes.detect_document_type import detect_document_type_node  # noqa: E402
from earnings_agents.workflow import build_graph  # noqa: E402
from earnings_agents.company_registry import lookup_by_cik, lookup_by_ticker  # noqa: E402
from earnings_agents.tools.edgar_client import get_latest_earnings_url, get_next_8k_status  # noqa: E402
from earnings_agents.hooks import set_detail_callback, set_node_callback  # noqa: E402

SEP = "=" * 64

_logger = logging.getLogger(__name__)


def _fmt_dur(ms: float) -> str:
    """Format a millisecond duration as a human-readable string.

    Examples: 450 ms → "0.5s", 5300 ms → "5.3s", 196200 ms → "3m 16s".
    """
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    return f"{m}m {s - m * 60:.0f}s"


# Human-readable stage names for the rich progress description
_NODE_LABELS: dict[str, str] = {
    "load_company_concepts_node":     "load concepts",
    "detect_document_type_node":      "detect type",
    "extract_html_text_node":         "fetch text",
    "extract_financial_metrics_node": "extract metrics",
    "analyze_metrics_node":           "analyse",
    "cleanup_metrics_node":           "cleanup",
    "mongodb_save_node":              "save",
}

# Short abbreviations used when compacting the completed-stage breadcrumb
# so it fits within the terminal width.
_SHORT_STAGE: dict[str, str] = {
    "load concepts":  "cncpt",
    "detect type":    "type",
    "fetch text":     "fetch",
    "fetch pdf":      "fetch",
    "extract metrics": "extr",
    "analyse":        "anal",
    "cleanup":        "clean",
    "save":           "save",
}


def _format_step_line(node_name: str, state: dict) -> str | None:
    """Return a single-line human-readable summary for a completed node.

    Returns ``None`` when no summary is needed.
    """
    if node_name == "load_company_concepts_node":
        if state.get("status") in ("skipped", "failed"):
            reason = (state.get("error") or "no concepts")[:70]
            return f"  [load concepts]  skipped — {reason}"
        n = len(state.get("target_concepts") or [])
        pt = state.get("detected_period_type") or "quarterly"
        return f"  [load concepts]  {n} concepts  ({pt})"

    if node_name == "detect_document_type_node":
        if state.get("status") == "failed":
            return f"  [detect type]    failed — {(state.get('error') or '')[:60]}"
        return f"  [detect type]    {state.get('file_type') or '?'}"

    if node_name == "extract_html_text_node":
        if state.get("status") == "failed":
            return "  [fetch text]    failed"
        n = len(state.get("raw_text") or "")
        lines = [f"  [fetch text]     {n:,} chars"]
        supp_log = state.get("_supplemental_log") or []
        if supp_log:
            lines.extend(supp_log)
        return "\n".join(lines)

    if node_name == "extract_financial_metrics_node":
        attempt = state.get("extraction_attempts", 1)
        metrics = state.get("metrics") or {}
        n_metrics = len([k for k in metrics if not k.startswith("__")])
        if state.get("status") == "failed":
            return f"  [extract {attempt}/3]    failed — {(state.get('error') or '')[:60]}"
        concept_metrics = state.get("concept_metrics") or {}
        derived_ids = set(state.get("derived_concept_ids") or [])
        target_concepts = state.get("target_concepts") or []
        n_total_concepts = len(target_concepts) + len(state.get("calculated_concepts") or [])
        n_mapped = len(concept_metrics) - len(derived_ids)
        n_derived = len(derived_ids)
        n_absent = len([c for c in target_concepts if c["_id"] not in concept_metrics])
        lines = [f"  [extract {attempt}/3]    → {n_metrics} metrics extracted"]
        if n_total_concepts:
            lines.append(
                f"  [extract {attempt}/3]    concepts: "
                f"{n_mapped} mapped  {n_derived} derived  {n_absent} not in filing"
                f"  (of {n_total_concepts})"
            )
        return "\n".join(lines)

    if node_name == "analyze_metrics_node":
        attempt = state.get("extraction_attempts", 1)
        findings = state.get("findings") or []
        high = [f for f in findings if isinstance(f, dict) and f.get("severity") == "high"]
        medium = [f for f in findings if isinstance(f, dict) and f.get("severity") == "medium"]
        needs = state.get("needs_reextract", False)
        if needs:
            msgs = "; ".join((f.get("message") or "?")[:45] for f in high[:2])
            extra = f"  (+{len(high) - 2} more)" if len(high) > 2 else ""
            lines = [f"  [analyze {attempt}/3]     ✗ {msgs}{extra}  → re-extract"]
            notes = (state.get("extraction_notes") or "").strip()
            if notes:
                lines.append("  ┌─ hints injected into next prompt ───────────────────────")
                for note_line in notes.splitlines()[:15]:
                    lines.append(f"  │  {note_line}")
                remaining = len(notes.splitlines()) - 15
                if remaining > 0:
                    lines.append(f"  │  ... ({remaining} more lines)")
                lines.append("  └─────────────────────────────────────────────────────────")
            return "\n".join(lines)
        if high:
            msgs = "; ".join((f.get("message") or "?")[:45] for f in high[:2])
            return f"  [analyze {attempt}/3]     ⚠ {len(high)} unresolved (max attempts): {msgs}"
        suffix = f"  ({len(medium)} medium)" if medium else ""
        return f"  [analyze {attempt}/3]     ✓ all required found{suffix}"

    if node_name == "cleanup_metrics_node":
        removed = state.get("cleanup_removed") or []
        return f"  [cleanup]        {len(removed)} key(s) removed"

    if node_name == "mongodb_save_node":
        s = state.get("status", "?")
        ticker_val = state.get("ticker", "?")
        period = state.get("sec_report_date") or ""
        year = period[:4] if period else "?"
        n_concepts = len(state.get("concept_metrics") or {})
        if s == "saved":
            return f"  [save]           ✓  {ticker_val}_{year}_latest  ({n_concepts} concepts upserted)"
        if s == "failed":
            return f"  [save]           ✗  {(state.get('error') or 'failed')[:70]}"
        return f"  [save]           {s}"

    return None


def _check_ollama() -> tuple[bool, str]:
    """Return (ok, detail) for Ollama reachability and model availability."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, f"Ollama responded {resp.status_code}"
        available = [m["name"] for m in resp.json().get("models", [])]
        base_name = OLLAMA_MODEL.split(":")[0]
        if not any(m.startswith(base_name) for m in available):
            return False, f"Model '{OLLAMA_MODEL}' not pulled (available: {available or 'none'})"
        return True, f"Ollama OK — model '{OLLAMA_MODEL}' available"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama unreachable: {exc}"


def _check_llm() -> tuple[bool, str]:
    """Return (ok, detail) for the selected LLM provider configuration."""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            return False, "Groq API key missing (set GROQ_API_KEY in .env)"
        return True, f"Groq configured — model '{GROQ_MODEL}'"
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            return False, "Gemini API key missing (set GEMINI_API_KEY in .env)"
        return True, f"Gemini configured — model '{GEMINI_MODEL}'"
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            return False, "DeepSeek API key missing (set DEEPSEEK_API_KEY in .env)"
        return True, f"DeepSeek configured — model '{DEEPSEEK_MODEL}'"
    return _check_ollama()


def _check_mongodb() -> tuple[bool, str]:
    """Return (ok, detail) for MongoDB reachability."""
    try:
        from pymongo import MongoClient  # local import — only needed in dry-run path

        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return True, f"MongoDB OK — {MONGODB_DB}.{MONGODB_COLLECTION}"
    except Exception as exc:  # noqa: BLE001
        return False, f"MongoDB unreachable: {exc}"


def _print_latest_data_status(companies: list[dict], printer=print) -> None:
    """For each company, show the last stored period and which period is needed next.

    Queries ``concept_values_annual`` and ``concept_values_quarterly`` to find
    the most recent period already in normalize_data, then infers the next
    expected period so the operator knows which 8-K to target.
    """
    try:
        from earnings_agents.tools.normalize_data_client import (
            get_company_by_ticker,
            get_latest_period,
        )
    except Exception as exc:  # noqa: BLE001
        printer(f"[WARN] Could not import normalize_data_client: {exc}")
        return

    printer("")
    printer("── normalize_data coverage ──────────────────────────────────────")
    for info in companies:
        ticker = info.get("ticker") or ""
        name = info.get("company_name", ticker)
        label = f"{ticker or name}"

        try:
            company = get_company_by_ticker(ticker) if ticker else None
        except Exception:  # noqa: BLE001
            company = None

        if company is None:
            printer(f"  {label:<12}  not in normalize_data.companies — targeted extraction disabled")
            continue

        cik = company["cik"]
        fy_end_month = company["fiscal_year_end_month"]
        fy_end_code = company.get("fiscal_year_end_code") or f"{fy_end_month:02d}??"

        try:
            latest = get_latest_period(cik)
        except Exception as exc:  # noqa: BLE001
            printer(f"  {label:<12}  DB error: {exc}")
            continue

        if latest is None:
            printer(f"  {label:<12}  no data yet  →  fetch any available 8-K")
            continue

        pt = latest["period_type"]
        fy = latest["fiscal_year"]
        q = latest["quarter"]
        end_dt = latest["end_date"].strftime("%Y-%m-%d") if latest["end_date"] else "?"

        if pt == "annual":
            last_str = f"FY{fy} annual  (end {end_dt})"
            # Annual filing covers Q4 + full year; next needed is Q1 of the next fiscal year.
            next_str = f"FY{fy + 1} Q1 8-K"
        elif (q or 0) >= 4:
            # A Q4 record is equivalent to the annual — the fiscal year is
            # already closed (the annual 10-K *is* the Q4 report). The next
            # needed filing is Q1 of the following fiscal year, NOT "FY annual".
            last_str = f"FY{fy} Q4 (=annual)  (end {end_dt})"
            next_str = f"FY{fy + 1} Q1 8-K"
        else:
            last_str = f"FY{fy} Q{q}  (end {end_dt})"
            if (q or 0) == 3:
                # Q3 is the last standalone quarterly 8-K.
                # The annual 8-K covers Q4 + full year — there is no Q4-only 8-K.
                next_str = f"FY{fy} Annual 8-K  (fiscal year-end {fy_end_code})"
            else:
                next_str = f"FY{fy} Q{(q or 0) + 1} 8-K"

        # Cross-check SEC EDGAR: is the next needed 8-K already filed?
        sec_status = get_next_8k_status(cik, end_dt)
        if sec_status["available"]:
            sec_note = f"[SEC ✓ filed  report {sec_status['sec_report_date']}]"
        elif sec_status.get("latest_edgar_report_date"):
            # EDGAR has a recent 8-K but it didn't clear the next-period threshold —
            # it's the already-stored period's filing. No new 8-K has landed yet.
            sec_note = "[✓ already stored — re-run will skip, no new 8-K on SEC yet]"
        else:
            sec_note = "[SEC ⏳ not yet]"

        printer(f"  {label:<12}  last stored: {last_str:<36}  need: {next_str}")
        printer(f"  {'':<12}  {sec_note}")

    printer("─" * 66)
    printer("")




def _print_insertion_summary(results: list[dict], printer=print) -> None:
    """Print a clean table of all inserted concept values for every saved result.

    Shown once at the very end of the run so operators can quickly verify what
    was upserted.  Values computed by the Tier-3 derivation engine are tagged
    with ``[DERIVED]`` so they are visually distinct from values read directly
    from the filing.
    """
    saved = [r for r in results if r.get("status") == "saved" and r.get("concept_metrics")]
    if not saved:
        return

    for result in saved:
        ticker = result.get("ticker", "?")
        concept_metrics: dict = result.get("concept_metrics") or {}
        target_concepts: list = result.get("target_concepts") or []
        calculated_concepts: list = result.get("calculated_concepts") or []
        derived_ids: set[str] = set(result.get("derived_concept_ids") or [])

        all_concepts = target_concepts + calculated_concepts
        id_to_label: dict[str, str] = {
            c["_id"]: c.get("label", c["_id"]) for c in all_concepts
        }

        period = result.get("sec_report_date", "")
        header = f"Inserted concept values for {ticker}"
        if period:
            header += f"  ({period})"
        printer("")
        printer(f"┌─ {header} {'─' * max(1, 66 - len(header))}")

        rows = sorted(
            concept_metrics.items(),
            key=lambda kv: id_to_label.get(kv[0], kv[0]).lower(),
        )
        from_filing = [(cid, v) for cid, v in rows if cid not in derived_ids]
        derived_rows = [(cid, v) for cid, v in rows if cid in derived_ids]

        for cid, value in from_filing:
            label = id_to_label.get(cid, cid)
            printer(f"│  {label:<45} {value}")

        if derived_rows:
            printer(f"├─ Derived ({len(derived_rows)}) ─{'─' * 54}")
            for cid, value in derived_rows:
                label = id_to_label.get(cid, cid)
                printer(f"│  {label:<45} {value}  [DERIVED]")

        absent = sorted(c["label"] for c in target_concepts if c["_id"] not in concept_metrics)
        if absent:
            printer(f"├─ Not in filing ({len(absent)}) ─{'─' * 51}")
            for lbl in absent:
                printer(f"│  {lbl}")

        printer(f"└{'─' * 68}")


def _resolve_companies(ciks: list[str], tickers: list[str]) -> list[dict]:
    """Return a list of company dicts ready to feed into the graph."""
    companies: list[dict] = []

    for cik in ciks:
        info = lookup_by_cik(cik)
        if not info:
            print(f"[WARN] CIK {cik} not found in tickers.json — skipping")
            continue
        companies.append(info)

    for t in tickers:
        info = lookup_by_ticker(t)
        if not info:
            print(f"[WARN] Ticker {t.upper()} not found in tickers.json — skipping")
            continue
        companies.append(info)

    return companies


def _has_existing_period_data(ticker: str) -> bool:
    """Return True when normalize_data already has at least one stored period for the ticker."""
    if not ticker:
        return False
    try:
        from earnings_agents.tools.normalize_data_client import (
            get_company_by_ticker,
            get_latest_period,
        )

        company = get_company_by_ticker(ticker)
        if company is None:
            return False
        return get_latest_period(company["cik"]) is not None
    except Exception:  # noqa: BLE001 — fail safe: never force a skip on ambiguity
        return False


def _build_initial_state(info: dict, printer=print) -> dict:
    """Build the LangGraph initial state for one company.

    Queries SEC EDGAR for the latest 8-K Exhibit 99.1 URL and injects it
    directly into the state so the pipeline starts at load_company_concepts.
    """
    ticker = info.get("ticker") or ""
    company_name = info["company_name"]
    cik = info["cik"]

    _base = {
        "ticker": ticker or cik,
        "company_name": company_name,
        "file_type": None,
        "raw_text": None,
        "metrics": None,
        "error": None,
        "extraction_attempts": 0,
        "extraction_notes": None,
        "needs_reextract": False,
        "previous_high_finding_keys": None,
    }

    if not _has_existing_period_data(ticker):
        printer(
            f"  [SKIP]   {company_name} ({ticker or cik}) — no existing normalize_data period data; skipping 8-K discovery"
        )
        return {
            **_base,
            "discovered_file_url": None,
            "status": "skipped",
            "error": "no existing normalize_data period data found for this company; skipping 8-K path.",
        }

    # ── Skip guard: check if the 8-K's fiscal period is already stored ──────
    # Uses fiscal_year_end_month + SEC submissions API to determine
    # (fiscal_year, quarter) and checks concept_values_quarterly / _annual.
    skip_state = _resolve_8k_skip_guard(ticker, cik, printer=printer)
    if skip_state is not None:
        return skip_state

    printer(f"  [EDGAR]  {company_name} ({ticker or cik}) querying SEC EDGAR...")
    filing_url, supplemental_urls, sec_report_date = get_latest_earnings_url(cik)
    if not filing_url:
        return {
            **_base,
            "discovered_file_url": None,
            "supplemental_file_urls": [],
            "status": "failed",
            "error": f"No 8-K earnings filing found on SEC EDGAR for CIK {cik}",
        }
    if supplemental_urls:
        printer(f"  [EDGAR]  +{len(supplemental_urls)} supplemental exhibit(s): {[u.rsplit('/', 1)[-1] for u in supplemental_urls]}")
    return {
        **_base,
        "discovered_file_url": filing_url,
        "supplemental_file_urls": supplemental_urls,
        "sec_report_date": sec_report_date,
        "status": "discovered",
    }


def _run_company(graph, info: dict, printer=print) -> dict:
    label = f"{info['company_name']} ({info.get('ticker') or info['cik']})"
    printer(f"\n{SEP}")
    printer(f"  Company : {label}")
    printer(f"  CIK     : {info['cik']}")

    state = _build_initial_state(info, printer=printer)

    if state["status"] == "failed":
        printer(f"  [SKIP]  {state['error']}")
        printer(SEP)
        return state

    if state["status"] == "skipped":
        printer(f"  [SKIP]  {state['error']}")
        printer(SEP)
        return state

    if state["status"] == "already_stored":
        printer(f"  [UP TO DATE]  period {state.get('sec_report_date', '?')} already in normalize_data — skipping")
        printer(SEP)
        return state

    if state["status"] != "failed":
        if state.get("discovered_file_url"):
            printer(f"  Filing  : {state['discovered_file_url']}")
    printer(SEP)

    final = graph.invoke(state)

    printer(f"\n  Status  : {final.get('status')}")
    printer(f"  File URL: {final.get('discovered_file_url')}")
    printer(f"  Type    : {final.get('file_type')}")
    if final.get("metrics"):
        m = final["metrics"]
        printer(f"  Metrics ({len(m)} fields):")
        for lbl, value in m.items():
            printer(f"    {lbl:<40} {value}")
    if final.get("error"):
        printer(f"  Error   : {final.get('error')}")
    printer(SEP)

    return final


def _dry_run_company(
    info: dict,
    printer=print,
) -> dict:
    """Resolve URLs and check service connectivity without running the LLM or saving.

    Returns the initial state augmented with a ``_dry_run_verdict`` key:
    ``"ready"`` | ``"warning"`` | ``"blocked"``.
    """
    label = f"{info['company_name']} ({info.get('ticker') or info['cik']})"
    printer(f"\n{SEP}")
    printer(f"  DRY-RUN : {label}")
    printer(f"  CIK     : {info['cik']}")

    state = _build_initial_state(info, printer=printer)

    llm_ok, llm_detail = _check_llm()
    mongo_ok, mongo_detail = _check_mongodb()

    file_type: str | None = None
    url_blocked = state.get("status") == "failed"
    already_stored = state.get("status") == "already_stored"
    if not url_blocked and not already_stored and state.get("discovered_file_url"):
        dt = detect_document_type_node(state)
        file_type = dt.get("file_type")

    if state.get("status") == "skipped":
        verdict = "skipped"
    elif url_blocked:
        verdict = "blocked"
    elif already_stored:
        verdict = "already_stored"
    elif not llm_ok or not mongo_ok:
        verdict = "warning"
    else:
        verdict = "ready"

    url_display = state.get("discovered_file_url") or "(none)"
    printer(SEP)
    printer(f"  URL     : {url_display}")
    if file_type:
        printer(f"  DocType : {file_type}")
    printer(f"  LLM     : {'OK  ' if llm_ok else 'FAIL'} — {llm_detail}")
    printer(f"  MongoDB : {'OK  ' if mongo_ok else 'FAIL'} — {mongo_detail}")
    printer(f"  Verdict : {verdict.upper()}")
    if verdict == "blocked":
        printer(f"  Reason  : {state.get('error', 'URL resolution failed')}")
    elif verdict == "skipped":
        printer(f"  Reason  : {state.get('error', 'No existing normalize_data period data found')}")
        printer("  Action  : Skipping SEC 8-K discovery because no prior normalize_data period exists")
    elif verdict == "already_stored":
        printer(f"  Reason  : period {state.get('sec_report_date', '?')} already stored in normalize_data")
        printer("  Action  : No action needed — data is up to date")
    elif verdict == "warning":
        if not llm_ok:
            if LLM_PROVIDER == "ollama":
                printer(
                    "  Action  : Start Ollama and pull the required model (see OLLAMA_MODEL in .env)"
                )
            else:
                printer("  Action  : Set GROQ_API_KEY (and optionally GROQ_MODEL) in .env")
        if not mongo_ok:
            printer("  Action  : Start MongoDB or check MONGODB_URI in .env")
    else:
        printer("  Action  : Run without --dry-run to execute the full pipeline")
    printer(SEP)

    return {**state, "_dry_run_verdict": verdict}


# ── Thread-pool workers ──────────────────────────────────────────────────────

def _is_already_saved(ticker: str) -> bool:
    """Return True if an earnings document for *ticker* already exists in MongoDB for the current year."""
    from datetime import datetime, timezone
    from earnings_agents.tools.mongodb_client import get_collection
    year = datetime.now(timezone.utc).year
    doc_id = f"{ticker}_{year}_latest"
    try:
        return get_collection().count_documents({"_id": doc_id}, limit=1) > 0
    except Exception:  # noqa: BLE001
        return False


def _resolve_8k_skip_guard(
    ticker: str,
    cik: str,
    printer=print,
) -> dict | None:
    """Check whether the latest 8-K's fiscal period is already stored in the DB.

    Fetches the SEC submissions API once and:
      1. Gets ``fiscal_year_end_month`` from normalize_data.
      2. Finds the latest 8-K's ``filingDate``.
      3. Uses ``_infer_8k_fiscal_period`` to determine ``(fiscal_year, quarter)``.
      4. Checks ``concept_values_quarterly`` / ``concept_values_annual``.

    Returns a state dict with ``status="already_stored"`` when the period
    already exists, or ``None`` when the pipeline should proceed.

    Also returns ``None`` on any error (fail-safe: never skips on ambiguity).
    """
    if not ticker or not cik:
        return None
    try:
        from earnings_agents.tools.normalize_data_client import (
            fiscal_period_exists,
            get_company_by_ticker,
        )
        from earnings_agents.tools.edgar_client import (
            _EDGAR_SUBMISSIONS,
            _edgar_get,
            _infer_8k_fiscal_period,
            normalize_cik,
        )

        # 1. Get fiscal_year_end_month.
        company = get_company_by_ticker(ticker)
        if company is None:
            return None
        fy_end_month = company.get("fiscal_year_end_month")
        if not fy_end_month:
            return None

        # 2. Fetch SEC submissions (one HTTP call).
        cik_padded = normalize_cik(cik)
        sub_url = _EDGAR_SUBMISSIONS.format(cik=cik_padded)
        from earnings_agents.config import HTTP_TIMEOUT
        try:
            resp = _edgar_get(sub_url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001
            return None

        recent = data.get("filings", {}).get("recent", {})
        forms: list[str] = recent.get("form", [])
        filing_dates: list[str] = recent.get("filingDate", [])
        items_list: list[str] = recent.get("items", [])

        # Find latest 8-K (Item 2.02) filing_date.
        filing_date_str: str | None = None
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            item_str = items_list[i] if i < len(items_list) else ""
            if "2.02" in item_str:
                fd = filing_dates[i] if i < len(filing_dates) else ""
                if fd:
                    filing_date_str = fd
                break

        # 3. Determine (fiscal_year, quarter) the 8-K reports on.
        if filing_date_str:
            fp = _infer_8k_fiscal_period(recent, filing_date_str, fy_end_month)
            if fp is not None:
                fy, q, pt = fp
                q_arg = q if pt == "quarterly" else None
                if fiscal_period_exists(cik, fy, q_arg):
                    printer(
                        f"  [UP TO DATE] period FY{fy}"
                        + (f" Q{q}" if pt == "quarterly" else " (annual)")
                        + f" already stored — skipping"
                    )
                    return {
                        "status": "already_stored",
                        "ticker": ticker,
                        "company_name": company.get("name", ticker),
                        "discovered_file_url": None,
                        "file_type": None,
                        "raw_text": None,
                        "metrics": None,
                        "error": None,
                        "extraction_attempts": 0,
                        "extraction_notes": None,
                        "needs_reextract": False,
                        "previous_high_finding_keys": None,
                    }

        return None
    except Exception:  # noqa: BLE001 — fail safe: never skip on ambiguity
        return None


def _run_company_parallel(args: tuple) -> dict:
    """Thread worker: run one company and update the shared rich Progress."""
    graph, info, skip_existing, progress, overall_task = args
    ticker = info.get("ticker", "")
    name = ticker or info.get("company_name", "?")
    label = f"[cyan]{name}[/]"

    if skip_existing and ticker and _is_already_saved(ticker):
        progress.update(overall_task, advance=1)
        return {"status": "skipped", "ticker": ticker}

    company_task = progress.add_task(f"{label}  resolving\u2026", total=None)

    _completed: list[str] = []
    _current_stage: list[str] = ["resolving\u2026"]
    _current_detail: list[str] = [""]
    _extract_pass: list[int] = [0]  # counts how many extraction passes have started
    _last_chunk_done: list[int] = [0]  # tracks last printed chunk-done count per pass
    _llm_calls: list[int] = [0]  # total LLM API requests fired this run

    def _render() -> str:
        """Build a terminal-width-aware progress description.

        Completed stages are abbreviated (e.g. "\u2713disc \u2713type \u2713fetch"). When
        the full string exceeds the available terminal width, the breadcrumb
        is replaced with a compact count ("(3\u2713)") and, if still too long,
        the active-stage detail is truncated to fit.
        """
        try:
            term_w = progress.console.width
        except Exception:  # noqa: BLE001
            term_w = 80
        # Overhead: spinner(2) + bar(36) + timer(10) + padding(6) \u2248 54 chars
        max_chars = max(30, term_w - 54)

        active_stage = _current_stage[0]
        detail = _current_detail[0]
        active_text = f"{active_stage}: {detail}" if detail else active_stage

        done_parts = [f"\u2713{_SHORT_STAGE.get(s, s[:5])}" for s in _completed]
        done_plain = " ".join(done_parts)

        # Attempt 1: name + all abbreviated completed stages + active stage
        sep = "  \u2192 "
        full_plain = (
            f"{name}  {done_plain}{sep}{active_text}"
            if done_parts
            else f"{name}  \u2192 {active_text}"
        )
        if len(full_plain) <= max_chars:
            done_markup = f"[green]{done_plain}[/]" if done_parts else ""
            return f"{label}  {done_markup}{sep}[bold]{active_text}[/]"

        # Attempt 2: name + count of completed + active stage
        n = len(_completed)
        count_plain = f"{name}  ({n}\u2713){sep}{active_text}"
        if len(count_plain) <= max_chars:
            count_markup = f"[green]({n}\u2713)[/]" if n > 0 else ""
            return f"{label}  {count_markup}{sep}[bold]{active_text}[/]"

        # Attempt 3: name + count + active stage, truncating detail to fit
        avail = max_chars - len(name) - len(active_stage) - 8
        short_detail = detail[:max(0, avail)] if detail else ""
        trunc_ellipsis = "\u2026" if detail and len(detail) > avail else ""
        active_trunc = (
            f"{active_stage}: {short_detail}{trunc_ellipsis}" if short_detail else active_stage
        )
        count_markup = f"[green]({n}\u2713)[/]  " if n > 0 else ""
        return f"{label}  {count_markup}[bold]\u2192 {active_trunc}[/]"

    def _node_cb(node_name: str, event: str, _ticker: str, node_state=None, elapsed_ms: float | None = None) -> None:
        stage = _NODE_LABELS.get(node_name, node_name.replace("_node", ""))
        if event == "start":
            if node_name == "extract_financial_metrics_node":
                _extract_pass[0] += 1
                _last_chunk_done[0] = 0  # reset per-pass chunk counter
                if _extract_pass[0] > 1:
                    stage = f"re-extract #{_extract_pass[0]}"
            _current_stage[0] = stage
            _current_detail[0] = ""
            progress.update(company_task, description=_render())
            from rich.markup import escape as _escape
            progress.console.print(f"[dim]{name}[/]  [bold cyan]▶ {_escape(stage)}[/]")
        elif event == "end":
            finished = _current_stage[0]  # may be "re-extract #2" etc.
            if finished not in _completed:
                _completed.append(finished)
            progress.update(company_task, description=_render())
            # Print a one-line step summary above the spinner so the operator
            # can see the full A-to-Z flow as it unfolds.
            if node_state is not None:
                line = _format_step_line(node_name, node_state)
                if line:
                    from rich.markup import escape as _escape
                    dur = f"  [dim]{_fmt_dur(elapsed_ms)}[/]" if elapsed_ms is not None else ""
                    parts = line.split("\n")
                    for i, part in enumerate(parts):
                        suffix = dur if i == len(parts) - 1 else ""
                        progress.console.print(f"[dim]{name}[/]{_escape(part)}{suffix}")

    def _detail_cb(detail: str) -> None:
        _current_detail[0] = detail
        progress.update(company_task, description=_render())
        # Print one line per chunk completion. The detail string looks like:
        #   "chunks 1✓ 2✓ 3⟳ (2/4)"
        # Parse the done/total counts and only print when done advances.
        if "chunks " in detail and ("✓" in detail or "✗" in detail):
            import re as _re
            m = _re.search(r'\((\d+)/(\d+)\)', detail)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if done > _last_chunk_done[0]:
                    _last_chunk_done[0] = done
                    from rich.markup import escape as _escape
                    pass_label = f"re-extract #{_extract_pass[0]}" if _extract_pass[0] > 1 else "extract"
                    progress.console.print(
                        f"[dim]{name}[/]  {_escape(f'[{pass_label}]')}  chunk {done}/{total}  "
                        + _escape(detail.split("(")[0].strip())
                    )

    def _call_cb(msg: str) -> None:
        from rich.markup import escape as _escape
        esc_msg = _escape(msg)
        # Highlight LLM-related messages (direct calls and LLM-triggered steps)
        # distinctly so they stand out from DB/HTTP operational messages.
        is_llm = " [llm]" in msg or ("llm" in msg.lower() and "\u2192" in msg)
        if is_llm:
            if "\u2192 calling llm" in msg:
                _llm_calls[0] += 1
            progress.console.print(f"[dim]{name}[/][bold yellow]{esc_msg}[/]")
        else:
            progress.console.print(f"[dim]{name}[/]{esc_msg}")

    set_node_callback(_node_cb)
    set_detail_callback(_detail_cb)
    from earnings_agents.hooks import set_call_callback
    set_call_callback(_call_cb)
    result = _run_company(graph, info, printer=lambda _: None)
    set_node_callback(None)
    set_detail_callback(None)
    set_call_callback(None)

    status = result.get("status", "?")
    llm_tag = f"  [dim]({_llm_calls[0]} LLM calls)[/]" if _llm_calls[0] else ""
    if status == "saved":
        desc = f"[green]{name}[/]  saved \u2713{llm_tag}"
    elif status == "failed":
        desc = f"[red]{name}[/]  failed \u2717{llm_tag}"
    elif status == "already_stored":
        desc = f"[cyan]{name}[/]  already up to date \u2713"
    else:
        desc = f"[yellow]{name}[/]  {status}"
    progress.update(company_task, total=1, completed=1, description=desc)
    progress.update(overall_task, advance=1)
    return result


def _dry_run_company_parallel(args: tuple) -> dict:
    """Thread worker: dry-run one company and update the shared rich Progress."""
    info, skip_existing, progress, overall_task = args
    ticker = info.get("ticker", "")
    name = ticker or info.get("company_name", "?")
    label = f"[cyan]{name}[/]"

    if skip_existing and ticker and _is_already_saved(ticker):
        progress.update(overall_task, advance=1)
        return {"_dry_run_verdict": "ready", "status": "skipped", "ticker": ticker}

    company_task = progress.add_task(f"{label}  checking\u2026", total=None)
    result = _dry_run_company(info, printer=lambda _: None)
    verdict = result.get("_dry_run_verdict", "?")
    color = {"ready": "green", "warning": "yellow", "blocked": "red", "already_stored": "cyan"}.get(verdict, "white")
    verdict_label = {"already_stored": "up to date"}.get(verdict, verdict)
    progress.update(company_task, total=1, completed=1,
                    description=f"[{color}]{name}[/]  {verdict_label}")
    progress.update(overall_task, advance=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the earnings scraping pipeline for one or more companies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run earnings --cik 0000320193\n"
            "  uv run earnings --cik 0000320193 0000789019\n"
            "  uv run earnings --ticker AAPL MSFT\n"
        ),
    )
    parser.add_argument(
        "--cik",
        nargs="+",
        metavar="CIK",
        default=[],
        help="One or more CIK numbers (with or without leading zeros)",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="TICKER",
        default=[],
        help="One or more ticker symbols (e.g. AAPL MSFT)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Preview mode: resolve URLs and check service connectivity without running LLM "
            "extraction or saving to MongoDB. Prints a ready/warning/blocked verdict per company."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        metavar="N",
        help="Maximum parallel workers when processing multiple companies (default: 8).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip companies that already have a saved document in MongoDB for the current year.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show full DEBUG-level log output from every pipeline node.",
    )
    parser.add_argument(
        "--allow-inconsistent",
        action="store_true",
        default=False,
        help=(
            "Save the document to MongoDB even when accounting identity checks "
            "fail (Gross margin = Revenue − COGS, Net income = Pre-tax − Tax, …). "
            "Warnings are still recorded in the document's identity_warnings field."
        ),
    )
    parser.add_argument(
        "--no-llm-cleanup",
        action="store_true",
        default=False,
        help=(
            "Skip the LLM cleanup pass that removes duplicate/mis-scaled keys "
            "from the extracted metrics before saving."
        ),
    )
    args = parser.parse_args()

    if not args.cik and not args.ticker:
        parser.error("Provide at least one --cik or --ticker argument.")

    if args.allow_inconsistent:
        # Override the env-default save gate so this run accepts identity warnings.
        import earnings_agents.workflow as _wf
        _wf.STRICT_ACCURACY = False

    if args.no_llm_cleanup:
        import earnings_agents.nodes.cleanup_metrics as _cm
        _cm.CLEANUP_METRICS = False

    companies = _resolve_companies(args.cik, args.ticker)
    if not companies:
        print("No valid companies resolved. Exiting.")
        sys.exit(1)

    _progress = Progress(
        SpinnerColumn(finished_text="[green]\u2713[/]"),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=36),
        TimeElapsedColumn(),
    )

    # Route all logging through rich's console so log lines print above the
    # live progress display instead of writing raw bytes to stderr and
    # breaking the ANSI cursor control.
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=_progress.console,
                show_path=args.verbose,
                rich_tracebacks=args.verbose,
                log_time_format="[%X]",
            )
        ],
        force=True,
    )

    if LLM_PROVIDER == "openai":
        llm_label = f"openai:{OPENAI_MODEL}"
    elif LLM_PROVIDER == "groq":
        llm_label = f"groq:{GROQ_MODEL}"
    elif LLM_PROVIDER == "gemini":
        llm_label = f"gemini:{GEMINI_MODEL}"
    elif LLM_PROVIDER == "deepseek":
        llm_label = f"deepseek:{DEEPSEEK_MODEL}"
    else:
        llm_label = f"ollama:{OLLAMA_MODEL}"
    _progress.console.print(f"[bold cyan]LLM[/]      : {llm_label}")

    if args.dry_run:
        _print_latest_data_status(companies, printer=_progress.console.print)
        total = len(companies)
        with _progress as progress:
            overall_task = progress.add_task("[bold]Dry-run[/]", total=total)
            worker_args = [
                (c, args.skip_existing, progress, overall_task)
                for c in companies
            ]
            results = []
            if total == 1:
                results.append(_dry_run_company_parallel(worker_args[0]))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                    for future in concurrent.futures.as_completed(
                        pool.submit(_dry_run_company_parallel, wa) for wa in worker_args
                    ):
                        results.append(future.result())

        blocked = sum(1 for r in results if r.get("_dry_run_verdict") == "blocked")
        warnings = sum(1 for r in results if r.get("_dry_run_verdict") == "warning")
        skipped = sum(1 for r in results if r.get("_dry_run_verdict") == "skipped")
        up_to_date_count = sum(1 for r in results if r.get("_dry_run_verdict") == "already_stored")
        ready = total - blocked - warnings - skipped - up_to_date_count
        parts = [f"{ready} ready"]
        if up_to_date_count:
            parts.append(f"{up_to_date_count} up to date")
        if skipped:
            parts.append(f"{skipped} skipped")
        if warnings:
            parts.append(f"{warnings} warning")
        if blocked:
            parts.append(f"{blocked} blocked")
        print(f"Dry-run: {total} companies \u2014 {', '.join(parts)}.")
        sys.exit(1 if blocked else 0)

    # Live run
    _print_latest_data_status(companies, printer=_progress.console.print)
    graph = build_graph()
    total = len(companies)
    with _progress as progress:
        overall_task = progress.add_task("[bold]Companies[/]", total=total)
        worker_args = [
            (graph, c, args.skip_existing, progress, overall_task)
            for c in companies
        ]
        results = []
        if total == 1:
            results.append(_run_company_parallel(worker_args[0]))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                for future in concurrent.futures.as_completed(
                    pool.submit(_run_company_parallel, wa) for wa in worker_args
                ):
                    results.append(future.result())

    saved = [r for r in results if r.get("status") == "saved"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    up_to_date = [r for r in results if r.get("status") == "already_stored"]
    failed = [r for r in results if r.get("status") not in ("saved", "skipped", "already_stored")]
    summary = f"Done: {len(saved)}/{total} saved"
    if skipped:
        summary += f", {len(skipped)} skipped"
    if up_to_date:
        summary += f", {len(up_to_date)} already up to date"
    if failed:
        summary += f", {len(failed)} failed"
    print(summary + ".")
    if failed:
        print("Failed: " + ", ".join(r.get("ticker", "?") for r in failed))
        for r in failed:
            if r.get("error"):
                print(f"  {r.get('ticker', '?')}: {r['error']}")
    # Silence MongoDB driver teardown noise (heartbeat/connection-pool debug
    # messages fired by atexit handlers) so the summary is always the last
    # thing printed in the terminal.
    for _log_name in ("pymongo", "mongodb", "bson"):
        logging.getLogger(_log_name).setLevel(logging.WARNING)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

