"""Tests for analysis/calculators.py — income-statement derivation engine."""
from __future__ import annotations

import pytest

from earnings_agents.analysis.calculators import (
    _identify_role,
    derive_missing_concept_metrics,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _c(cid: str, label: str) -> dict:
    return {
        "_id": cid,
        "label": label,
        "concept": f"us-gaap:{label.replace(' ', '')}",
        "path": "0",
        "statement_type": "income_statement",
    }


# ── _identify_role ────────────────────────────────────────────────────────────

class TestIdentifyRole:
    def test_revenue_variants(self):
        for label in ("Total revenue", "Net revenue", "Net sales", "Revenue"):
            assert _identify_role(label) == "revenue", label

    def test_cost_of_revenue_not_revenue(self):
        assert _identify_role("Cost of revenue") == "cost_of_revenue"
        assert _identify_role("Cost of goods sold") == "cost_of_revenue"

    def test_gross_profit(self):
        assert _identify_role("Gross profit") == "gross_profit"

    def test_gross_margin_pct_not_gross_profit(self):
        assert _identify_role("Gross margin %") == "gross_margin_pct"
        assert _identify_role("Gross profit margin") == "gross_margin_pct"

    def test_operating_income(self):
        for label in ("Operating income", "Operating profit", "Operating loss",
                      "Income from operations"):
            assert _identify_role(label) == "operating_income", label

    def test_rd_expense(self):
        assert _identify_role("Research and development") == "rd_expense"
        assert _identify_role("R&D expense") == "rd_expense"

    def test_net_income(self):
        for label in ("Net income", "Net loss", "Net earnings"):
            assert _identify_role(label) == "net_income", label

    def test_pretax_income(self):
        assert _identify_role("Income before income taxes") == "pretax_income"
        assert _identify_role("Pre-tax income") == "pretax_income"

    def test_tax_expense(self):
        assert _identify_role("Income tax expense") == "tax_expense"
        assert _identify_role("Provision for income taxes") == "tax_expense"

    def test_eps_diluted(self):
        assert _identify_role("Diluted net income per share") == "eps_diluted"
        assert _identify_role("Net income per diluted share") == "eps_diluted"

    def test_eps_basic(self):
        assert _identify_role("Basic net income per share") == "eps_basic"

    def test_unknown_label_returns_none(self):
        assert _identify_role("Total assets") is None
        assert _identify_role("Cash and cash equivalents") is None


# ── derive_missing_concept_metrics ────────────────────────────────────────────

class TestGrossProfitDerivation:
    def test_derives_gross_profit_from_revenue_and_cogs(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("cogs", "Cost of revenue"),
            _c("gp", "Gross profit"),
        ]
        result = derive_missing_concept_metrics({"rev": 1000.0, "cogs": 400.0}, concepts)
        assert result["gp"] == pytest.approx(600.0)

    def test_does_not_overwrite_existing_gross_profit(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("cogs", "Cost of revenue"),
            _c("gp", "Gross profit"),
        ]
        result = derive_missing_concept_metrics({"rev": 1000.0, "cogs": 400.0, "gp": 601.0}, concepts)
        assert result["gp"] == pytest.approx(601.0)

    def test_skips_when_cogs_missing(self):
        concepts = [_c("rev", "Total revenue"), _c("gp", "Gross profit")]
        result = derive_missing_concept_metrics({"rev": 1000.0}, concepts)
        assert "gp" not in result

    def test_negative_gross_profit_allowed(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("cogs", "Cost of revenue"),
            _c("gp", "Gross profit"),
        ]
        result = derive_missing_concept_metrics({"rev": 100.0, "cogs": 150.0}, concepts)
        assert result["gp"] == pytest.approx(-50.0)


class TestOperatingIncomeDerivation:
    def test_from_gross_profit_and_total_opex(self):
        concepts = [
            _c("gp", "Gross profit"),
            _c("opex", "Total operating expenses"),
            _c("oi", "Operating income"),
        ]
        result = derive_missing_concept_metrics({"gp": 600.0, "opex": 200.0}, concepts)
        assert result["oi"] == pytest.approx(400.0)

    def test_fallback_from_individual_opex_items(self):
        concepts = [
            _c("gp", "Gross profit"),
            _c("rd", "Research and development"),
            _c("sm", "Sales and marketing"),
            _c("ga", "General and administrative"),
            _c("oi", "Operating income"),
        ]
        result = derive_missing_concept_metrics(
            {"gp": 600.0, "rd": 50.0, "sm": 80.0, "ga": 30.0}, concepts
        )
        assert result["oi"] == pytest.approx(440.0)

    def test_total_opex_takes_priority_over_items(self):
        concepts = [
            _c("gp", "Gross profit"),
            _c("opex", "Total operating expenses"),
            _c("rd", "Research and development"),
            _c("oi", "Operating income"),
        ]
        # total_opex=200 → OI=400; individual items would give 600−50=550
        result = derive_missing_concept_metrics(
            {"gp": 600.0, "opex": 200.0, "rd": 50.0}, concepts
        )
        assert result["oi"] == pytest.approx(400.0)

    def test_skips_when_gross_profit_absent(self):
        concepts = [
            _c("opex", "Total operating expenses"),
            _c("oi", "Operating income"),
        ]
        result = derive_missing_concept_metrics({"opex": 200.0}, concepts)
        assert "oi" not in result


class TestNetIncomeDerivation:
    def test_derives_net_income_from_pretax_and_tax(self):
        concepts = [
            _c("pt", "Income before income taxes"),
            _c("tax", "Income tax expense"),
            _c("ni", "Net income"),
        ]
        result = derive_missing_concept_metrics({"pt": 400.0, "tax": 80.0}, concepts)
        assert result["ni"] == pytest.approx(320.0)

    def test_pretax_skipped_when_interest_items_present(self):
        """Pre-tax derivation is disabled when below-the-line items are present."""
        concepts = [
            _c("oi", "Operating income"),
            _c("ii", "Interest income"),
            _c("pt", "Income before income taxes"),
        ]
        # interest_income is present → pretax derivation disabled
        result = derive_missing_concept_metrics({"oi": 400.0, "ii": 20.0}, concepts)
        assert "pt" not in result


class TestMarginDerivation:
    def test_gross_margin_pct(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("gp", "Gross profit"),
            _c("gm", "Gross profit margin %"),
        ]
        result = derive_missing_concept_metrics({"rev": 1000.0, "gp": 600.0}, concepts)
        assert result["gm"] == pytest.approx(60.0)

    def test_operating_margin_pct(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("oi", "Operating income"),
            _c("om", "Operating margin %"),
        ]
        result = derive_missing_concept_metrics({"rev": 1000.0, "oi": 150.0}, concepts)
        assert result["om"] == pytest.approx(15.0)

    def test_net_margin_pct(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("ni", "Net income"),
            _c("nm", "Net margin %"),
        ]
        result = derive_missing_concept_metrics({"rev": 1000.0, "ni": 100.0}, concepts)
        assert result["nm"] == pytest.approx(10.0)

    def test_implausible_margin_skipped(self):
        """Margins outside [−200, 200] are rejected."""
        concepts = [
            _c("rev", "Total revenue"),
            _c("gp", "Gross profit"),
            _c("gm", "Gross profit margin %"),
        ]
        result = derive_missing_concept_metrics({"rev": 10.0, "gp": 9_999.0}, concepts)
        assert "gm" not in result


class TestEpsDerivation:
    def test_eps_diluted(self):
        concepts = [
            _c("ni", "Net income"),
            _c("sd", "Weighted average diluted shares"),
            _c("epsd", "Diluted net income per share"),
        ]
        # Net income 156_837_000 (USD), shares 214_000_000 (full count) → ~0.733
        result = derive_missing_concept_metrics(
            {"ni": 156_837_000.0, "sd": 214_000_000.0}, concepts
        )
        assert result["epsd"] == pytest.approx(0.733, abs=0.001)

    def test_eps_implausible_value_skipped(self):
        concepts = [
            _c("ni", "Net income"),
            _c("sd", "Weighted average diluted shares"),
            _c("epsd", "Diluted net income per share"),
        ]
        # Tiny shares count → absurdly large EPS; should be rejected
        result = derive_missing_concept_metrics({"ni": 1_000_000.0, "sd": 1.0}, concepts)
        assert "epsd" not in result


class TestChainedDerivation:
    def test_revenue_cogs_to_gp_to_oi(self):
        """GP derived first, then used to derive OI."""
        concepts = [
            _c("rev", "Total revenue"),
            _c("cogs", "Cost of revenue"),
            _c("gp", "Gross profit"),
            _c("opex", "Total operating expenses"),
            _c("oi", "Operating income"),
        ]
        result = derive_missing_concept_metrics(
            {"rev": 1000.0, "cogs": 400.0, "opex": 200.0}, concepts
        )
        assert result["gp"] == pytest.approx(600.0)
        assert result["oi"] == pytest.approx(400.0)

    def test_full_chain_revenue_to_net_income(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("cogs", "Cost of revenue"),
            _c("gp", "Gross profit"),
            _c("opex", "Total operating expenses"),
            _c("oi", "Operating income"),
            _c("pt", "Income before income taxes"),
            _c("tax", "Income tax expense"),
            _c("ni", "Net income"),
        ]
        result = derive_missing_concept_metrics(
            {"rev": 1000.0, "cogs": 400.0, "opex": 200.0, "tax": 50.0}, concepts
        )
        assert result["gp"] == pytest.approx(600.0)
        assert result["oi"] == pytest.approx(400.0)
        assert result["pt"] == pytest.approx(400.0)   # pretax ≈ OI (no interest items)
        assert result["ni"] == pytest.approx(350.0)


class TestEdgeCases:
    def test_empty_concept_metrics(self):
        concepts = [_c("gp", "Gross profit")]
        result = derive_missing_concept_metrics({}, concepts)
        assert result == {}

    def test_empty_all_concepts(self):
        result = derive_missing_concept_metrics({"rev": 1000.0}, [])
        assert result == {"rev": 1000.0}

    def test_calculated_concept_with_unknown_label_skipped(self):
        concepts = [
            _c("rev", "Total revenue"),
            _c("xx", "Custom KPI XYZ"),   # no matching role
        ]
        result = derive_missing_concept_metrics({"rev": 1000.0}, concepts)
        assert "xx" not in result

    def test_existing_values_not_mutated(self):
        """Input dict is not modified."""
        concepts = [
            _c("rev", "Total revenue"),
            _c("cogs", "Cost of revenue"),
            _c("gp", "Gross profit"),
        ]
        original = {"rev": 1000.0, "cogs": 400.0}
        derive_missing_concept_metrics(original, concepts)
        assert original == {"rev": 1000.0, "cogs": 400.0}

    def test_copies_role_value_for_multiple_calculated_concepts(self):
        """When two calculated concepts carry the same role, both get filled."""
        concepts = [
            _c("rev", "Total revenue"),
            _c("cogs", "Cost of revenue"),
            _c("gp1", "Gross profit"),
            _c("gp2", "Gross profit"),  # duplicate role in different calculated collections
        ]
        result = derive_missing_concept_metrics({"rev": 1000.0, "cogs": 400.0}, concepts)
        assert result["gp1"] == pytest.approx(600.0)
        assert result["gp2"] == pytest.approx(600.0)
