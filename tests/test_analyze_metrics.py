"""Tests for the analyze_metrics node and its underlying checkers."""
from __future__ import annotations

import pytest

from earnings_agents.analysis.critical_metrics import check_presence as presence_summary
from earnings_agents.analysis.findings import (
    check_case_duplicates,
    check_composite_keys,
    check_gaap_nongaap_leakage,
    check_gross_profit_identity,
    check_presence,
    check_source_grounding,
    check_suspect_round,
    derive_corrected_total_opex,
)
from earnings_agents.nodes.analyze_metrics import analyze_metrics_node
from earnings_agents.config import MAX_EXTRACTION_ATTEMPTS


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
    assert out["needs_reextract"] is True         # loop-back signalled
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
# Gross-profit income-statement identity checker
# ---------------------------------------------------------------------------

def test_gross_profit_identity_passes_when_reconciled():
    m = {
        "Revenues": 81_615_000_000.0,
        "Cost of revenue": 20_458_000_000.0,
        "Gross profit": 61_157_000_000.0,   # 81,615 − 20,458 = 61,157
    }
    assert check_gross_profit_identity(m) == []


def test_gross_profit_identity_flags_nvda_stale_cost_of_revenue():
    # The reported defect: Cost of Revenue = 48B (stale), Gross Profit = 59.62B.
    # 81,615 − 48,000 = 33,615 ≠ 59,620 → identity broken.
    m = {
        "Revenues": 81_615_000_000.0,
        "Cost of Revenue": 48_000_000_000.0,
        "Gross Profit": 59_620_000_000.0,
    }
    findings = check_gross_profit_identity(m)
    assert len(findings) == 1
    assert findings[0].type == "identity_violation"
    assert findings[0].severity == "high"


def test_gross_profit_identity_flags_cost_exceeding_revenue():
    # Cost of revenue ≥ revenue is impossible; reported even without gross profit.
    m = {
        "Revenue": 26_000_000_000.0,
        "Cost of revenue": 48_000_000_000.0,
    }
    findings = check_gross_profit_identity(m)
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert "negative" in findings[0].message.lower()


def test_gross_profit_identity_silent_when_components_missing():
    assert check_gross_profit_identity({"Revenues": 81_615_000_000.0}) == []


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

def test_analyze_gross_profit_violation_triggers_reextract():
    m = _full_metrics()
    # Break the income-statement identity: Revenue − Cost of revenue ≠ Gross profit.
    m["Cost of revenue"] = 80_000_000_000   # 100B − 80B = 20B ≠ 60B gross profit
    out = analyze_metrics_node(_state(m, attempts=1))
    assert out["needs_reextract"] is True
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


@pytest.mark.parametrize("key", [
    "Earnings Per Share, Basic",
    "Earnings Per Share, Diluted",
    "Weighted Average Number of Shares Outstanding, Basic",
    "Weighted Average Number of Shares Outstanding, Diluted",
    "Income (Loss) from Continuing Operations, Net of Tax, Attributable to Parent",
    "Other Comprehensive Income (Loss), Net of Tax, Portion Attributable to Parent",
    "Other Comprehensive Income (Loss), Cash Flow Hedge, Gain (Loss), after Reclassification and Tax",
])
def test_composite_key_does_not_flag_real_gaap_labels(key):
    """Legitimate GAAP labels with commas must not be treated as synonym lists."""
    assert check_composite_keys({key: 1.23}) == []


def test_composite_key_recorded_by_analyze_node_no_reextract():
    m = _full_metrics()
    m["Diluted earnings per share, Diluted EPS, Diluted net income per share"] = 1.59
    out = analyze_metrics_node(_state(m))
    assert any(f["type"] == "composite_key" for f in out["findings"])
    assert out["status"] == "extracted"   # low severity — no re-extract


# ---------------------------------------------------------------------------
# Opex-label collision checker
# ---------------------------------------------------------------------------

from earnings_agents.analysis.findings import check_opex_label_collision


def test_opex_collision_flags_when_opex_equals_operating_income():
    """Mirrors the real MSFT Q3 FY2026 defect: Total operating expenses = 38,398M = Operating income."""
    metrics = {
        "Revenue": 82_886_000_000.0,
        "Operating income": 38_398_000_000.0,
        "Total operating expenses": 38_398_000_000.0,   # wrong — should be ~17,660M or 44,488M
    }
    findings = check_opex_label_collision(metrics)
    assert len(findings) == 1
    f = findings[0]
    assert f.type == "suspect_value"
    assert f.severity == "medium"
    assert "Total operating expenses" in f.keys
    assert "Operating income" in f.keys


def test_opex_collision_flags_when_opex_equals_revenue():
    metrics = {
        "Revenue": 50_000_000_000.0,
        "Total operating expenses": 50_000_000_000.0,
    }
    findings = check_opex_label_collision(metrics)
    assert len(findings) == 1
    assert findings[0].type == "suspect_value"


def test_opex_collision_clean_when_opex_differs():
    """Opex = COGS + opex lines ≠ operating income → no finding."""
    metrics = {
        "Revenue": 82_886_000_000.0,
        "Operating income": 38_398_000_000.0,
        "Total operating expenses": 44_488_000_000.0,   # 26,828 + 17,660
    }
    assert check_opex_label_collision(metrics) == []


# ---------------------------------------------------------------------------
# Income-statement ordering invariants
# ---------------------------------------------------------------------------

from earnings_agents.analysis.findings import (  # noqa: E402
    check_eps_dilution_ordering,
    check_operating_vs_gross,
)


def test_operating_vs_gross_flags_when_operating_exceeds_gross():
    metrics = {
        "Gross profit": 60_000_000_000.0,
        "Operating income": 70_000_000_000.0,   # impossible: OI > GP
    }
    findings = check_operating_vs_gross(metrics)
    assert len(findings) == 1
    f = findings[0]
    assert f.type == "suspect_value"
    assert f.severity == "medium"
    assert set(f.keys) == {"Operating income", "Gross profit"}


def test_operating_vs_gross_ok_when_operating_below_gross():
    metrics = {
        "Gross profit": 60_000_000_000.0,
        "Operating income": 30_000_000_000.0,
    }
    assert check_operating_vs_gross(metrics) == []


def test_operating_vs_gross_ignored_on_loss():
    """A negative operating income makes the ordering meaningless."""
    metrics = {
        "Gross profit": 10_000_000_000.0,
        "Operating income": -2_000_000_000.0,
    }
    assert check_operating_vs_gross(metrics) == []


def test_eps_dilution_flags_when_diluted_exceeds_basic():
    metrics = {
        "Basic earnings per share": 2.39,
        "Diluted earnings per share": 2.50,   # impossible: diluted > basic
    }
    findings = check_eps_dilution_ordering(metrics)
    assert len(findings) == 1
    f = findings[0]
    assert f.type == "suspect_value"
    assert f.severity == "medium"
    assert set(f.keys) == {"Diluted earnings per share", "Basic earnings per share"}


def test_eps_dilution_ok_when_diluted_below_basic():
    metrics = {
        "Basic earnings per share": 2.45,
        "Diluted earnings per share": 2.39,
    }
    assert check_eps_dilution_ordering(metrics) == []


def test_eps_dilution_ignored_on_loss():
    """For a net loss EPS is anti-dilutive (basic == diluted); not flagged."""
    metrics = {
        "Basic earnings per share": -1.20,
        "Diluted earnings per share": -1.20,
    }
    assert check_eps_dilution_ordering(metrics) == []


def test_eps_dilution_does_not_match_share_counts():
    """Weighted-average share-count lines must not be read as EPS values."""
    metrics = {
        "Weighted average shares outstanding, basic": 7_400_000_000.0,
        "Weighted average shares outstanding, diluted": 7_500_000_000.0,
    }
    # diluted share count > basic is normal and must NOT be flagged as an EPS anomaly
    assert check_eps_dilution_ordering(metrics) == []


def test_ordering_checkers_recorded_by_analyze_node_no_reextract():
    m = _full_metrics()
    m["Basic earnings per share"] = 2.30
    m["Diluted earnings per share"] = 2.50   # diluted > basic
    out = analyze_metrics_node(_state(m))
    assert any(f["type"] == "suspect_value" for f in out["findings"])
    # medium severity alone must not trigger a re-extract loop
    assert out["status"] == "extracted"
    assert out["needs_reextract"] is False


# ---------------------------------------------------------------------------
# Expanded tier registries
# ---------------------------------------------------------------------------

def test_presence_detects_new_tier2_metrics():
    metrics = {
        "Weighted average shares outstanding, basic": 7_400_000_000,
        "Interest expense": 500_000_000,
    }
    presence = presence_summary(metrics.keys())
    assert "Weighted Avg Shares Basic" not in presence["tier2_missing"]
    assert "Interest Expense" not in presence["tier2_missing"]


def test_presence_detects_new_tier3_metrics():
    metrics = {
        "Depreciation and amortization": 1_000_000_000,
        "Stock-based compensation": 800_000_000,
        "EBITDA": 12_000_000_000,
        "Dividends declared per common share": 0.25,
    }
    presence = presence_summary(metrics.keys())
    assert {
        "Depreciation & Amortization",
        "Stock-Based Compensation",
        "EBITDA",
        "Dividends per Share",
    } <= set(presence["tier3_present"])



def test_opex_collision_no_opex_key_returns_empty():
    """If the document doesn't have 'Total operating expenses', no finding is emitted."""
    metrics = {
        "Revenue": 82_886_000_000.0,
        "Operating income": 38_398_000_000.0,
        "Operating expenses": 17_660_000_000.0,
    }
    assert check_opex_label_collision(metrics) == []


def test_opex_collision_recorded_by_analyze_node_no_reextract():
    """medium severity — appended to notes but does NOT trigger re-extract on its own."""
    m = _full_metrics()
    m["Total operating expenses"] = m.get("Operating income", 38_398_000_000.0)
    m["Operating income"] = m.get("Operating income", 38_398_000_000.0)
    out = analyze_metrics_node(_state(m, attempts=1))
    assert any(f["type"] == "suspect_value" for f in out["findings"])
    # No high-severity findings expected → no re-extract
    assert out["needs_reextract"] is False


# ---------------------------------------------------------------------------
# Auto-correction: derive Total operating expenses from components
# ---------------------------------------------------------------------------

_COGS = 26_828_000_000.0
_OPEX_SUB = 17_660_000_000.0   # R&D + S&M + G&A
_OPINC = 38_398_000_000.0       # wrong value LLM assigned to Total opex
_CORRECTED = _COGS + _OPEX_SUB  # 44_488_000_000.0


def _collision_metrics_with_components() -> dict:
    """Metrics where Total operating expenses collides with Operating income
    AND the correct components (Cost of revenue, Operating expenses) are present."""
    return {
        "Total revenue": 82_886_000_000.0,
        "Revenue": 82_886_000_000.0,
        "Cost of revenue": _COGS,
        "Gross profit": 56_058_000_000.0,
        "Operating expenses": _OPEX_SUB,
        "Operating income": _OPINC,
        "Total operating expenses": _OPINC,   # ← wrong (collision)
        "Net income": 31_778_000_000.0,
        "Diluted earnings per share": 4.27,
    }


def test_derive_corrected_total_opex_returns_sum():
    m = _collision_metrics_with_components()
    key, value = derive_corrected_total_opex(m)
    assert key == "Total operating expenses"
    assert value == pytest.approx(_CORRECTED)


def test_derive_corrected_total_opex_returns_none_when_cogs_missing():
    m = _collision_metrics_with_components()
    del m["Cost of revenue"]
    key, value = derive_corrected_total_opex(m)
    assert key is None
    assert value is None


def test_derive_corrected_total_opex_returns_none_when_opex_sub_missing():
    m = _collision_metrics_with_components()
    del m["Operating expenses"]
    key, value = derive_corrected_total_opex(m)
    assert key is None
    assert value is None


def test_opex_auto_correction_applied_when_components_present():
    """analyze_metrics_node should correct Total operating expenses in metrics."""
    m = _collision_metrics_with_components()
    out = analyze_metrics_node(_state(m, attempts=1))

    # Corrected value in out["metrics"]
    assert out["metrics"]["Total operating expenses"] == pytest.approx(_CORRECTED)

    # auto_corrected finding emitted
    ac = [f for f in out["findings"] if f["type"] == "auto_corrected"]
    assert len(ac) == 1
    assert ac[0]["severity"] == "low"
    assert ac[0]["evidence"]["corrected_value"] == pytest.approx(_CORRECTED)
    assert ac[0]["evidence"]["old_value"] == pytest.approx(_OPINC)

    # Still no re-extract (medium collision + low correction, no high)
    assert out["needs_reextract"] is False


# ---------------------------------------------------------------------------
# Source-grounding ("show me") verification — check_source_grounding
# ---------------------------------------------------------------------------

_GROUNDING_SOURCE = (
    "Condensed Consolidated Statements of Operations (in thousands)\n"
    "Total revenue 1,385,629\n"
    "Cost of revenue 357,108\n"
    "Net income 156,837\n"
)


def test_grounding_no_snippets_is_noop():
    """Absent snippets → no findings (feature degrades gracefully)."""
    metrics = {"Total revenue": 1_385_629_000}
    assert check_source_grounding(metrics, None, _GROUNDING_SOURCE) == []
    assert check_source_grounding(metrics, {}, _GROUNDING_SOURCE) == []


def test_grounding_verbatim_snippet_passes():
    metrics = {"Total revenue": 1_385_629_000}
    snippets = {"Total revenue": "Total revenue 1,385,629"}
    assert check_source_grounding(metrics, snippets, _GROUNDING_SOURCE) == []


def test_grounding_paraphrased_snippet_with_present_number_passes():
    """Snippet reworded but its number is in the source → grounded."""
    metrics = {"Net income": 156_837_000}
    snippets = {"Net income": "net income of 156,837 for the quarter"}
    assert check_source_grounding(metrics, snippets, _GROUNDING_SOURCE) == []


def test_grounding_fabricated_value_flags_high():
    """Snippet and its number are both absent → high source_unverified finding."""
    metrics = {"Operating income": 999_999_000}
    snippets = {"Operating income": "Operating income 999,999"}
    out = check_source_grounding(metrics, snippets, _GROUNDING_SOURCE)
    assert len(out) == 1
    assert out[0].type == "source_unverified"
    assert out[0].severity == "high"
    assert out[0].keys == ("Operating income",)


def test_grounding_skips_null_metric_and_reserved_keys():
    metrics = {"Total revenue": None}
    snippets = {
        "Total revenue": "Total revenue 42",          # value is null → skip
        "__period__": "made up period 4242",          # reserved key → skip
    }
    assert check_source_grounding(metrics, snippets, _GROUNDING_SOURCE) == []


def test_analyze_node_flags_unverified_source_and_loops():
    """analyze_metrics_node wires the grounding checker and routes on it."""
    m = _full_metrics()
    state = _state(m, attempts=1)
    state["raw_text"] = _GROUNDING_SOURCE
    state["metric_source_snippets"] = {
        "Net income": "Net income 1,234,567,890",   # absent from source
    }
    out = analyze_metrics_node(state)
    assert any(f["type"] == "source_unverified" for f in out["findings"])
    assert out["needs_reextract"] is True


def test_opex_no_correction_when_components_missing():
    """When Cost of revenue is absent, no auto_corrected finding is emitted."""
    m = _collision_metrics_with_components()
    del m["Cost of revenue"]
    out = analyze_metrics_node(_state(m, attempts=1))

    # suspect_value finding still present (collision detected)
    assert any(f["type"] == "suspect_value" for f in out["findings"])
    # no auto_corrected finding
    assert not any(f["type"] == "auto_corrected" for f in out["findings"])
    # metrics value unchanged (still the colliding wrong value)
    assert out["metrics"]["Total operating expenses"] == pytest.approx(_OPINC)

