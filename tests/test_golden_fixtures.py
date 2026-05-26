"""Golden-fixture regression tests for extract_financial_metrics_node.

Each fixture JSON file in tests/fixtures/golden/ defines:
  - raw_text:           text fed to the node as state["raw_text"]
  - llm_responses:      ordered list of strings returned by the mocked LLM
                        (one entry per chunk invocation)
  - expected:           dict of metric_key → expected numeric value
  - must_be_absent:     list of keys that must NOT appear in the final metrics
  - chunk_size_override:  (optional) int — forces chunking by patching _chunk_text
                          to split raw_text on the XXXCHUNKBREAKXXX sentinel

These tests exercise the deterministic pipeline stages — prescan, scaling,
merging — without running a real LLM.  If the scaling logic, merge strategy,
or prescan patterns change, at least one fixture will fail.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.extract_financial_metrics import (
    extract_financial_metrics_node,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "golden"

_FIXTURE_FILES = sorted(_FIXTURES_DIR.glob("*.json"))


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _base_state(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": fixture["ticker"],
        "company_name": fixture["company_name"],
        "ir_url": "https://example.com/ir",
        "discovered_file_url": "https://example.com/earnings.html",
        "file_type": "html",
        "raw_text": fixture["raw_text"],
        "raw_sections": None,
        "target_concepts": None,
        "metrics": None,
        "error": None,
        "status": "text_extracted",
        "extraction_attempts": 0,
        "extraction_notes": None,
        "needs_reextract": False,
        "previous_high_finding_keys": None,
        "identity_warnings": None,
        "cleanup_removed": None,
        "findings": None,
        "company_cik": None,
        "concept_metrics": None,
        "fiscal_year_end_month": None,
        "fiscal_year_end_code": None,
    }


def _make_mock_build_llm(responses: list[str]):
    """Return a side_effect factory that yields successive LLM mocks.

    build_llm() is called once per chunk inside _invoke_chunk_with_retry.
    Each call must return a distinct mock object whose .invoke() returns
    the next response in the sequence.
    """
    response_iter = iter(responses)

    def _factory(**kwargs):  # noqa: ANN001
        m = MagicMock()
        m.invoke.return_value = next(response_iter)
        return m

    return _factory


def _split_raw_text_into_chunks(raw_text: str) -> list[str]:
    """Split on the XXXCHUNKBREAKXXX sentinel used in multi-chunk fixtures."""
    return [part.strip() for part in raw_text.split("XXXCHUNKBREAKXXX") if part.strip()]


@pytest.mark.parametrize("fixture_path", _FIXTURE_FILES, ids=lambda p: p.stem)
def test_golden_fixture(fixture_path: Path) -> None:
    """Run the full extract node against a golden fixture and verify output."""
    fixture = _load_fixture(fixture_path)
    state = _base_state(fixture)
    responses: list[str] = fixture["llm_responses"]
    expected: dict[str, Any] = fixture["expected"]
    must_be_absent: list[str] = fixture.get("must_be_absent", [])
    chunk_size_override: int | None = fixture.get("chunk_size_override")

    factory = _make_mock_build_llm(responses)

    patches: list[Any] = [
        patch(
            "earnings_agents.nodes.extract_financial_metrics.build_llm",
            side_effect=factory,
        ),
    ]

    # For multi-chunk fixtures we replace _chunk_text so the short synthetic
    # raw_text is split into the intended number of chunks without requiring
    # a chunk_size smaller than the text length.
    if chunk_size_override is not None:
        chunks = _split_raw_text_into_chunks(fixture["raw_text"])
        assert len(chunks) == len(responses), (
            f"Fixture {fixture_path.stem}: {len(responses)} llm_responses "
            f"but raw_text splits into {len(chunks)} chunks at XXXCHUNKBREAKXXX"
        )
        patches.append(
            patch(
                "earnings_agents.nodes.extract_financial_metrics._chunk_text",
                return_value=chunks,
            )
        )

    with patches[0]:
        if len(patches) > 1:
            with patches[1]:
                result = extract_financial_metrics_node(state)
        else:
            result = extract_financial_metrics_node(state)

    # Node must complete extraction successfully.
    assert result["status"] == "extracted", (
        f"Node failed with status={result['status']!r}, error={result.get('error')!r}"
    )
    assert result["error"] is None

    metrics: dict[str, Any] = result["metrics"]

    # Every expected key must be present and numerically close.
    for key, expected_val in expected.items():
        assert key in metrics, f"Expected metric {key!r} missing from output"
        assert metrics[key] == pytest.approx(expected_val, rel=1e-6), (
            f"Metric {key!r}: expected {expected_val}, got {metrics[key]}"
        )

    # Sentinel keys must have been stripped during post-processing.
    for key in must_be_absent:
        assert key not in metrics, (
            f"Key {key!r} must be absent from final metrics but was present"
        )
