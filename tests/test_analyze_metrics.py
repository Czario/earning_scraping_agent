"""Tests for the analyze_metrics node and its underlying checkers."""
from __future__ import annotations

from earnings_agents.analysis.critical_metrics import check_presence as presence_summary
from earnings_agents.analysis.findings import (
    check_balance_sheet_identity,
    check_case_duplicates,
    check_composite_keys,
    check_gaap_nongaap_leakage,
    check_presence,
    check_sign_anomalies,
    check_suspect_round,
)
from earnings_agents.nodes.analyze_metrics import analyze_metrics_node
from earnings_agents.nodes.reflect_metrics import MAX_EXTRACTION_ATTEMPTS


# ---------------------------------------------------------------------------
# Presence checker
# ---------------------------------------------------------------------------

def _full_metrics() -> dict:
    """A reasonably complete metrics dict matching all Tier-1 patterns.

    Tier 1 is income-statement only. Balance sheet / cash-flow keys are Tier 2
    and intentionally omitted here to keep the fixture realistic (a typical
    press-release summary may not include them).
    """
    return {
        "Total revenue": 100_000_000_000,
        "Gross profit": 60_000_000_000,
        "Operating income": 30_000_000_000,
        "Net income": 20_000_000_000,
        "Diluted earnings per share": 2.39,
    }


def test_presence_full_dict_has_no_tier1_misses():
    presence = presence_summary(_full_metrics().keys())
    assert presence["tier1_missing"] == []


def test_presence_missing_revenue_and_eps():
    m = _full_metrics()
    del m["Total revenue"]
    del m["Diluted earnings per share"]
    presence = presence_summary(m.keys())
    assert "Total Revenue" in presence["tier1_missing"]
    assert "Diluted EPS" in presence["tier1_missing"]


# ---------------------------------------------------------------------------
# Case-duplicate checker
# ---------------------------------------------------------------------------

def test_case_duplicates_detected_when_values_match():
    metrics = {
        "Non-GAAP Operating expenses": 12_000_000_000,
        "Non-GAAP operating expenses": 12_000_000_000,
        "Net income": 20_000_000_000,
    }
    findings = check_case_duplicates(metrics)
    assert len(findings) == 1
    f = findings[0]
    assert f.type == "case_duplicate"
    assert set(f.keys) == {"Non-GAAP Operating expenses", "Non-GAAP operating expenses"}


def test_case_duplicates_not_flagged_when_values_differ():
    metrics = {
        "Revenue": 100,
        "revenue": 200,
    }
    assert check_case_duplicates(metrics) == []


# ---------------------------------------------------------------------------
# analyze_metrics_node — routing
# ---------------------------------------------------------------------------

def _state(metrics: dict, attempts: int = 1) -> dict:
    return {
        "ticker": "TEST",
        "company_name": "Test Co.",
        "ir_url": "",
        "discovered_file_url": None,
        "file_type": "html",
        "raw_text": "irrelevant for analyze",
        "metrics": metrics,
        "error": None,
        "status": "extracted",
        "extraction_attempts": attempts,
        "extraction_notes": None,
        "identity_warnings": None,
        "cleanup_removed": None,
        "findings": None,
    }


def test_analyze_full_metrics_routes_forward():
    out = analyze_metrics_node(_state(_full_metrics()))
    assert out["status"] == "extracted"           # no loop-back
    assert out["extraction_notes"] is None
    # Tier-2 misses are expected (the fixture only covers Tier-1) and remain
    # informational — none of them should be "high" severity.
    severities = {f["severity"] for f in out["findings"]}
    assert "high" not in severities


def test_analyze_missing_tier1_triggers_reextract():
    m = _full_metrics()
    del m["Total revenue"]
    del m["Net income"]
    out = analyze_metrics_node(_state(m, attempts=1))
    assert out["status"] == "text_extracted"      # loop-back signalled
    assert out["extraction_notes"] is not None
    assert "Total Revenue" in out["extraction_notes"]
    assert any(f["type"] == "missing_critical" for f in out["findings"])


def test_analyze_missing_tier1_at_max_attempts_does_not_loop():
    m = _full_metrics()
    del m["Total revenue"]
    out = analyze_metrics_node(_state(m, attempts=MAX_EXTRACTION_ATTEMPTS))
    assert out["status"] == "extracted"           # cap hit → no further loop
    # findings are still recorded for downstream visibility
    assert any(f["type"] == "missing_critical" for f in out["findings"])


def test_analyze_records_case_duplicate_finding():
    m = _full_metrics()
    m["net income"] = m["Net income"]              # case-only duplicate
    out = analyze_metrics_node(_state(m))
    assert out["status"] == "extracted"           # low severity → no loop
    assert any(f["type"] == "case_duplicate" for f in out["findings"])


# ---------------------------------------------------------------------------
# Sanity: check_presence wrapper builds Finding objects with correct severity
# ---------------------------------------------------------------------------

def test_check_presence_severity_mapping():
    presence = {
        "tier1_missing": ["Total Revenue"],
        "tier2_missing": ["Cost of Revenue"],
        "tier3_present": [],
    }
    findings = check_presence({}, presence)
    sev = {f.type: f.severity for f in findings}
    assert sev["missing_critical"] == "high"
    assert sev["missing_expected"] == "medium"


# ---------------------------------------------------------------------------
# Balance-sheet identity checker
# ---------------------------------------------------------------------------

def test_balance_sheet_identity_passes_when_reconciled():
    m = {
        "Total assets": 100.0,
        "Total liabilities": 60.0,
        "Total stockholders' equity": 40.0,
    }
    assert check_balance_sheet_identity(m) == []


def test_balance_sheet_identity_within_one_percent_tolerance():
    m = {
        "Total assets": 100.0,
        "Total liabilities": 60.5,            # 0.5% drift
        "Total stockholders' equity": 40.0,
    }
    assert check_balance_sheet_identity(m) == []


def test_balance_sheet_identity_flags_nvda_style_mismatch():
    # NVDA defect #4: Total liabilities reported as 64B but components sum to ~56.5B.
    m = {
        "Total assets": 96_000_000_000.0,
        "Total liabilities": 64_000_000_000.0,
        "Total stockholders' equity": 25_000_000_000.0,   # 64 + 25 = 89 ≠ 96
    }
    findings = check_balance_sheet_identity(m)
    assert len(findings) == 1
    assert findings[0].type == "identity_violation"
    assert findings[0].severity == "high"


# ---------------------------------------------------------------------------
# Sign-anomaly checker
# ---------------------------------------------------------------------------

def test_sign_anomaly_flags_negative_inventories():
    # NVDA defect #1: Inventories: -4,420,000,000 (cash-flow row, not balance sheet).
    findings = check_sign_anomalies({"Inventories": -4_420_000_000})
    assert len(findings) == 1
    assert findings[0].type == "sign_anomaly"
    assert findings[0].severity == "medium"
    assert findings[0].keys == ("Inventories",)


def test_sign_anomaly_ignores_positive_balances():
    assert check_sign_anomalies({"Inventories": 5_000_000_000, "Goodwill": 10_000_000_000}) == []


def test_sign_anomaly_ignores_unrelated_keys():
    # Operating cash flow can legitimately be negative for some companies; not on our list.
    assert check_sign_anomalies({"Net cash from operating activities": -1_000_000}) == []


# ---------------------------------------------------------------------------
# Suspect-round-number heuristic
# ---------------------------------------------------------------------------

def test_suspect_round_flags_exact_billion():
    # NVDA defect #5: "Short-term debt: 1,000,000,000" — implausibly round.
    findings = check_suspect_round({"Short-term debt": 1_000_000_000})
    assert len(findings) == 1
    assert findings[0].type == "suspect_round"
    assert findings[0].severity == "low"


def test_suspect_round_flags_one_point_three_billion():
    findings = check_suspect_round({"Sales, general and administrative": 1_300_000_000})
    assert len(findings) == 1


def test_suspect_round_skips_per_share_concepts():
    # 1.00 (an EPS-shaped value) must never trigger.
    assert check_suspect_round({"Diluted EPS": 1.00}) == []
    # Even when the magnitude qualifies, "per share" is on the never-round list.
    assert check_suspect_round({"Revenue per share": 100_000_000}) == []


def test_suspect_round_skips_non_round_values():
    assert check_suspect_round({"Revenue": 1_234_567_890}) == []


def test_suspect_round_skips_megacap_totals():
    # $100B+ exact totals often appear in real megacap statements; the
    # heuristic intentionally exempts them to avoid false positives.
    assert check_suspect_round({"Total revenue": 200_000_000_000}) == []


# ---------------------------------------------------------------------------
# analyze_metrics_node — integration with new checkers
# ---------------------------------------------------------------------------

def test_analyze_balance_sheet_violation_triggers_reextract():
    m = _full_metrics()
    # Supply a full balance sheet where the identity is intentionally broken.
    m["Total assets"] = 500_000_000_000
    m["Total liabilities"] = 200_000_000_000
    m["Total stockholders' equity"] = 10_000_000_000   # 200 + 10 ≠ 500
    out = analyze_metrics_node(_state(m, attempts=1))
    assert out["status"] == "text_extracted"
    assert any(f["type"] == "identity_violation" for f in out["findings"])


# ---------------------------------------------------------------------------
# GAAP / Non-GAAP leakage checker
# ---------------------------------------------------------------------------

def test_gaap_nongaap_leakage_flags_prefixed_keys():
    # Mirrors the NVDA output: primary income statement mixed with reconciliation table.
    metrics = {
        "Net income": 58_321_000_000,
        "GAAP net income": 42_960_000_000,
        "Non-GAAP net income": 38_969_000_000,
        "GAAP operating income": 44_299_000_000,
        "Total impact of non-GAAP adjustments to operating income": 175_000_000,
    }
    findings = check_gaap_nongaap_leakage(metrics)
    assert len(findings) == 1
    f = findings[0]
    assert f.type == "gaap_nongaap_leakage"
    assert f.severity == "low"
    leaked = set(f.keys)
    assert "GAAP net income" in leaked
    assert "Non-GAAP net income" in leaked
    assert "GAAP operating income" in leaked
    assert "Total impact of non-GAAP adjustments to operating income" in leaked
    # Plain "Net income" must NOT be flagged.
    assert "Net income" not in leaked


def test_gaap_nongaap_leakage_clean_metrics_returns_empty():
    metrics = {
        "Revenue": 81_615_000_000,
        "Gross profit": 61_157_000_000,
        "Operating income": 53_536_000_000,
        "Net income": 58_321_000_000,
        "Diluted earnings per share": 2.39,
    }
    assert check_gaap_nongaap_leakage(metrics) == []


def test_gaap_nongaap_leakage_adjusted_key_flagged():
    assert len(check_gaap_nongaap_leakage({"Adjusted operating income": 1_000})) == 1


def test_gaap_nongaap_leakage_recorded_by_analyze_node():
    m = _full_metrics()
    m["GAAP net income"] = 42_960_000_000
    m["Non-GAAP operating income*"] = 44_474_000_000
    out = analyze_metrics_node(_state(m))
    assert any(f["type"] == "gaap_nongaap_leakage" for f in out["findings"])
    # Leakage is low-severity — must NOT trigger a re-extract loop.
    assert out["status"] == "extracted"


# ---------------------------------------------------------------------------
# Composite / synonym-list key checker
# ---------------------------------------------------------------------------

def test_composite_key_flags_comma_joined_synonyms():
    # Mirrors the NVDA defect: LLM copied the SCOPE bullet verbatim as the key.
    metrics = {
        "Net income": 58_321_000_000,
        "Diluted earnings per share, Diluted EPS, Diluted net income per share": 1.59,
    }
    findings = check_composite_keys(metrics)
    assert len(findings) == 1
    f = findings[0]
    assert f.type == "composite_key"
    assert f.severity == "low"
    assert "Diluted earnings per share, Diluted EPS, Diluted net income per share" in f.keys
    assert "Net income" not in f.keys


def test_composite_key_flags_slash_joined_synonyms():
    metrics = {
        "Cost of revenue / Cost of goods sold": 20_458_000_000,
    }
    findings = check_composite_keys(metrics)
    assert len(findings) == 1
    assert findings[0].type == "composite_key"


def test_composite_key_clean_metrics_returns_empty():
    metrics = {
        "Revenue": 81_615_000_000,
        "Diluted earnings per share": 2.39,
        "Net income": 58_321_000_000,
    }
    assert check_composite_keys(metrics) == []


def test_composite_key_recorded_by_analyze_node_no_reextract():
    m = _full_metrics()
    m["Diluted earnings per share, Diluted EPS, Diluted net income per share"] = 1.59
    out = analyze_metrics_node(_state(m))
    assert any(f["type"] == "composite_key" for f in out["findings"])
    assert out["status"] == "extracted"   # low severity — no re-extract

