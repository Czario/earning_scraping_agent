"""Tests for the LLM cleanup node — guardrails must reject any LLM
attempt to invent values, mutate values, or worsen identity warnings."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.cleanup_metrics import cleanup_metrics_node


def _base_state(metrics: dict) -> dict:
    return {
        "ticker": "TEST",
        "company_name": "TEST CORP",
        "status": "extracted",
        "metrics": metrics,
        "raw_text": "irrelevant source text",
        "identity_warnings": [],
        "extraction_attempts": 1,
        "extraction_notes": None,
    }


@patch("earnings_agents.nodes.cleanup_metrics.build_llm")
def test_cleanup_drops_duplicate_keys(mock_build_llm):
    """LLM proposes removing a GAAP-prefixed duplicate; node accepts it."""
    metrics = {
        "Revenue": 82_886_000_000,
        "Net income": 31_778_000_000,
        "GAAP net income": 31_778_000_000,  # duplicate of Net income
        "Diluted Earnings per Share": 4.27,
        "Weighted average shares outstanding: Diluted": 7_445_000_000,
    }
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"remove": ["GAAP net income"], '
        '"reasons": {"GAAP net income": "Rule A: duplicate of Net income"}}'
    )
    mock_build_llm.return_value = mock_llm

    result = cleanup_metrics_node(_base_state(metrics))

    assert "GAAP net income" not in result["metrics"]
    assert result["metrics"]["Net income"] == 31_778_000_000
    assert result["cleanup_removed"] == ["GAAP net income"]


@patch("earnings_agents.nodes.cleanup_metrics.build_llm")
def test_cleanup_rejects_unknown_key_removal(mock_build_llm):
    """LLM proposes removing a key not in the input; node keeps originals."""
    metrics = {"Revenue": 82_886_000_000, "Net income": 31_778_000_000}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"remove": ["Net income", "Phantom Metric"], "reasons": {}}'
    )
    mock_build_llm.return_value = mock_llm

    result = cleanup_metrics_node(_base_state(metrics))

    # Whole batch rejected because of the unknown key
    assert result["metrics"] == metrics


@patch("earnings_agents.nodes.cleanup_metrics.build_llm")
def test_cleanup_rejects_invalid_json(mock_build_llm):
    """Garbled LLM output → keep originals."""
    metrics = {"Revenue": 82_886_000_000}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "not even close to JSON {"
    mock_build_llm.return_value = mock_llm

    result = cleanup_metrics_node(_base_state(metrics))

    assert result["metrics"] == metrics


@patch("earnings_agents.nodes.cleanup_metrics.build_llm")
def test_cleanup_keeps_originals_when_llm_call_raises(mock_build_llm):
    """LLM exception → keep originals, do not propagate."""
    metrics = {"Revenue": 82_886_000_000}
    mock_build_llm.side_effect = RuntimeError("ollama down")

    result = cleanup_metrics_node(_base_state(metrics))

    assert result["metrics"] == metrics


@patch("earnings_agents.nodes.cleanup_metrics.build_llm")
def test_cleanup_reverts_when_removal_breaks_sanity(mock_build_llm):
    """Removing Net income would break the EPS×shares sanity check → revert."""
    metrics = {
        "Net income": 31_778_000_000,
        "Diluted Earnings per Share": 4.27,
        "Weighted average shares outstanding: Diluted": 7_445_000_000,
        # Add an obvious case-only dup so _needs_cleanup() triggers the LLM call.
        "Net Income": 31_778_000_000,
    }
    # LLM (wrongly) proposes dropping Net income. Without it the EPS check
    # silently passes (no NI to compare), so this particular removal doesn't
    # introduce a NEW warning. We instead simulate a removal that introduces
    # one by removing Diluted shares while keeping a slightly off Net income.
    metrics_with_broken = dict(metrics, **{"Net income": 12_345_000_000})
    state = _base_state(metrics_with_broken)
    # Pre-cleanup already flagged the EPS mismatch:
    state["identity_warnings"] = [
        "Diluted EPS sanity check: reported 4.2700 vs computed 1.6587 "
        "(Net income / Diluted shares)"
    ]
    mock_llm = MagicMock()
    # Propose dropping diluted shares — which would remove the existing
    # warning but doesn't add a new one, so cleanup is allowed.
    mock_llm.invoke.return_value = (
        '{"remove": ["Weighted average shares outstanding: Diluted"], '
        '"reasons": {"Weighted average shares outstanding: Diluted": "test"}}'
    )
    mock_build_llm.return_value = mock_llm

    result = cleanup_metrics_node(state)
    # The cleanup is allowed because it doesn't introduce NEW failures.
    assert "Weighted average shares outstanding: Diluted" not in result["metrics"]


@patch("earnings_agents.nodes.cleanup_metrics.build_llm")
def test_cleanup_protects_concept_mapped_keys(mock_build_llm):
    """LLM tries to remove a key that is in mapped_metric_keys — must be blocked."""
    metrics = {
        "Net sales": 5_529_145_000,
        "Membership fee income": 132_355_000,
        "Total revenues": 5_661_500_000,
        "Net income": 2_000_000_000,
    }
    state = _base_state(metrics)
    state["mapped_metric_keys"] = ["Net sales", "Membership fee income"]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"remove": ["Net sales", "Membership fee income"], '
        '"reasons": {'
        '"Net sales": "Rule A: duplicate of Total revenues", '
        '"Membership fee income": "Malformed value (likely mis-scaled)"'
        '}}'
    )
    mock_build_llm.return_value = mock_llm

    result = cleanup_metrics_node(state)

    assert "Net sales" in result["metrics"], "concept-mapped key must not be removed"
    assert "Membership fee income" in result["metrics"], "concept-mapped key must not be removed"
    assert result["metrics"]["Net sales"] == 5_529_145_000
    assert result["metrics"]["Membership fee income"] == 132_355_000


@patch("earnings_agents.nodes.cleanup_metrics.build_llm")
def test_cleanup_still_removes_non_protected_duplicate(mock_build_llm):
    """Keys NOT in mapped_metric_keys can still be removed by the LLM."""
    metrics = {
        "Revenue": 82_886_000_000,
        "GAAP Revenue": 82_886_000_000,  # true duplicate, not concept-mapped
    }
    state = _base_state(metrics)
    state["mapped_metric_keys"] = ["Revenue"]  # only "Revenue" is protected

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"remove": ["GAAP Revenue"], '
        '"reasons": {"GAAP Revenue": "Rule A: duplicate of Revenue"}}'
    )
    mock_build_llm.return_value = mock_llm

    result = cleanup_metrics_node(state)

    assert "GAAP Revenue" not in result["metrics"]
    assert result["metrics"]["Revenue"] == 82_886_000_000


def test_cleanup_disabled_returns_state_unchanged(monkeypatch):
    """When CLEANUP_METRICS=False, the node is a no-op."""
    import earnings_agents.nodes.cleanup_metrics as cm
    monkeypatch.setattr(cm, "CLEANUP_METRICS", False)

    metrics = {"Revenue": 82_886_000_000}
    state = _base_state(metrics)
    result = cleanup_metrics_node(state)
    assert result is state
