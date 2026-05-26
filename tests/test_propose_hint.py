"""Tests for the earnings-propose-hint CLI."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.cli.propose_hint import (
    _build_findings_block,
    _build_identity_block,
    _load_existing_hints,
    _write_proposed,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "status": "degraded",
        "scraped_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "findings": [
            {
                "type": "missing_tier1",
                "severity": "high",
                "message": "Revenue not found",
            },
            {
                "type": "sign_anomaly",
                "severity": "medium",
                "message": "Net income is negative",
            },
        ],
        "identity_warnings": ["Assets ≠ Liabilities + Equity (delta: 500M)"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _build_findings_block
# ---------------------------------------------------------------------------

class TestBuildFindingsBlock:
    def test_includes_finding_type_and_message(self):
        doc = _make_doc()
        block = _build_findings_block([doc])
        assert "missing_tier1" in block
        assert "Revenue not found" in block

    def test_deduplicates_identical_findings_across_docs(self):
        doc1 = _make_doc()
        doc2 = _make_doc()
        block = _build_findings_block([doc1, doc2])
        assert block.count("Revenue not found") == 1

    def test_empty_findings_returns_fallback(self):
        doc = _make_doc(findings=[])
        block = _build_findings_block([doc])
        assert "no structured findings" in block

    def test_non_dict_findings_are_skipped(self):
        doc = _make_doc(findings=["bad", None, 42])
        block = _build_findings_block([doc])
        assert "no structured findings" in block

    def test_includes_severity_label(self):
        doc = _make_doc()
        block = _build_findings_block([doc])
        assert "HIGH" in block or "MEDIUM" in block


# ---------------------------------------------------------------------------
# _build_identity_block
# ---------------------------------------------------------------------------

class TestBuildIdentityBlock:
    def test_includes_warning_text(self):
        doc = _make_doc()
        block = _build_identity_block([doc])
        assert "Assets" in block

    def test_deduplicates_identical_warnings(self):
        doc1 = _make_doc()
        doc2 = _make_doc()
        block = _build_identity_block([doc1, doc2])
        assert block.count("Assets") == 1

    def test_no_warnings_returns_none_label(self):
        doc = _make_doc(identity_warnings=[])
        block = _build_identity_block([doc])
        assert "(none)" in block


# ---------------------------------------------------------------------------
# _load_existing_hints
# ---------------------------------------------------------------------------

class TestLoadExistingHints:
    def test_returns_no_hint_file_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("earnings_agents.cli.propose_hint._HINTS_DIR", tmp_path)
        result = _load_existing_hints("AAPL")
        assert "no existing hint file" in result

    def test_returns_content_when_file_present(self, tmp_path, monkeypatch):
        (tmp_path / "MSFT.md").write_text("- Always look for segment revenue.", encoding="utf-8")
        monkeypatch.setattr("earnings_agents.cli.propose_hint._HINTS_DIR", tmp_path)
        result = _load_existing_hints("MSFT")
        assert "segment revenue" in result

    def test_empty_file_returns_empty_label(self, tmp_path, monkeypatch):
        (tmp_path / "GOOGL.md").write_text("   ", encoding="utf-8")
        monkeypatch.setattr("earnings_agents.cli.propose_hint._HINTS_DIR", tmp_path)
        result = _load_existing_hints("GOOGL")
        assert result == "(empty)"


# ---------------------------------------------------------------------------
# _write_proposed
# ---------------------------------------------------------------------------

class TestWriteProposed:
    def test_creates_file_in_proposed_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", tmp_path / "_proposed")
        out = _write_proposed("AAPL", "- Some hint.", force=False)
        assert out.exists()
        assert "Some hint." in out.read_text()

    def test_exits_when_file_exists_and_no_force(self, tmp_path, monkeypatch):
        proposed_dir = tmp_path / "_proposed"
        proposed_dir.mkdir()
        (proposed_dir / "AAPL.md").write_text("existing", encoding="utf-8")
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", proposed_dir)
        with pytest.raises(SystemExit):
            _write_proposed("AAPL", "new content", force=False)

    def test_force_overwrites_existing_file(self, tmp_path, monkeypatch):
        proposed_dir = tmp_path / "_proposed"
        proposed_dir.mkdir()
        (proposed_dir / "AAPL.md").write_text("old", encoding="utf-8")
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", proposed_dir)
        out = _write_proposed("AAPL", "new content", force=True)
        assert "new content" in out.read_text()

    def test_written_file_contains_review_header(self, tmp_path, monkeypatch):
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", tmp_path / "_proposed")
        out = _write_proposed("MSFT", "- hint", force=False)
        text = out.read_text()
        assert "Review and move" in text
        assert "MSFT" in text


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

class TestMain:
    def _mock_llm(self, response: str):
        llm = MagicMock()
        llm.invoke.return_value = response
        return llm

    @patch("earnings_agents.cli.propose_hint.get_collection")
    @patch("earnings_agents.cli.propose_hint.build_llm")
    def test_happy_path_creates_proposed_file(
        self, mock_build_llm, mock_get_col, tmp_path, monkeypatch
    ):
        mock_col = MagicMock()
        mock_col.find.return_value = [_make_doc()]
        mock_get_col.return_value = mock_col
        mock_build_llm.return_value = self._mock_llm("- Use 'Net sales' for Revenue.")

        monkeypatch.setattr("earnings_agents.cli.propose_hint._HINTS_DIR", tmp_path)
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", tmp_path / "_proposed")

        main(["--ticker", "AAPL"])

        out = tmp_path / "_proposed" / "AAPL.md"
        assert out.exists()
        assert "Net sales" in out.read_text()

    @patch("earnings_agents.cli.propose_hint.get_collection")
    def test_exits_gracefully_when_no_docs(self, mock_get_col, tmp_path, monkeypatch):
        mock_col = MagicMock()
        mock_col.find.return_value = []
        mock_get_col.return_value = mock_col
        monkeypatch.setattr("earnings_agents.cli.propose_hint._HINTS_DIR", tmp_path)
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", tmp_path / "_proposed")
        # Should print a message and return (not raise SystemExit)
        main(["--ticker", "AAPL"])
        # No proposed file created
        assert not (tmp_path / "_proposed" / "AAPL.md").exists()

    @patch("earnings_agents.cli.propose_hint.get_collection")
    def test_mongodb_error_exits_with_code_1(self, mock_get_col):
        mock_get_col.side_effect = ConnectionError("cannot connect")
        with pytest.raises(SystemExit) as exc_info:
            main(["--ticker", "AAPL"])
        assert exc_info.value.code == 1

    @patch("earnings_agents.cli.propose_hint.get_collection")
    @patch("earnings_agents.cli.propose_hint.build_llm")
    def test_llm_error_exits_with_code_1(self, mock_build_llm, mock_get_col, tmp_path, monkeypatch):
        mock_col = MagicMock()
        mock_col.find.return_value = [_make_doc()]
        mock_get_col.return_value = mock_col
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")
        mock_build_llm.return_value = mock_llm
        monkeypatch.setattr("earnings_agents.cli.propose_hint._HINTS_DIR", tmp_path)
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", tmp_path / "_proposed")
        with pytest.raises(SystemExit) as exc_info:
            main(["--ticker", "AAPL"])
        assert exc_info.value.code == 1

    @patch("earnings_agents.cli.propose_hint.get_collection")
    @patch("earnings_agents.cli.propose_hint.build_llm")
    def test_ticker_uppercased(self, mock_build_llm, mock_get_col, tmp_path, monkeypatch):
        mock_col = MagicMock()
        mock_col.find.return_value = [_make_doc()]
        mock_get_col.return_value = mock_col
        mock_build_llm.return_value = self._mock_llm("- A hint.")
        monkeypatch.setattr("earnings_agents.cli.propose_hint._HINTS_DIR", tmp_path)
        monkeypatch.setattr("earnings_agents.cli.propose_hint._PROPOSED_DIR", tmp_path / "_proposed")

        main(["--ticker", "aapl"])

        # MongoDB query should use uppercased ticker
        call_kwargs = mock_col.find.call_args[0][0]
        assert call_kwargs["ticker"] == "AAPL"
