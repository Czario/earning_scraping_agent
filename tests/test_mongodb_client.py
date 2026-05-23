"""Tests for mongodb_client.py — upsert retry behaviour."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from earnings_agents.tools.mongodb_client import _MONGO_MAX_RETRIES, upsert_earnings


def _make_collection(side_effects):
    col = MagicMock()
    col.update_one.side_effect = side_effects
    return col


@patch("earnings_agents.tools.mongodb_client.time.sleep", return_value=None)
@patch("earnings_agents.tools.mongodb_client.get_collection")
def test_upsert_succeeds_on_first_attempt(mock_get_col, mock_sleep):
    """Happy path: single attempt, no sleep, upsert called once."""
    col = _make_collection([None])  # update_one returns None on success
    mock_get_col.return_value = col

    upsert_earnings({"_id": "AAPL_2026_latest", "ticker": "AAPL"})

    assert col.update_one.call_count == 1
    mock_sleep.assert_not_called()


@patch("earnings_agents.tools.mongodb_client.time.sleep", return_value=None)
@patch("earnings_agents.tools.mongodb_client.get_collection")
def test_upsert_retries_on_transient_error_then_succeeds(mock_get_col, mock_sleep):
    """Two transient failures followed by success: retried twice with back-off."""
    col = _make_collection([
        RuntimeError("network blip"),
        RuntimeError("primary step-down"),
        None,  # third attempt succeeds
    ])
    mock_get_col.return_value = col

    upsert_earnings({"_id": "MSFT_2026_latest", "ticker": "MSFT"})

    assert col.update_one.call_count == 3
    assert mock_sleep.call_count == 2


@patch("earnings_agents.tools.mongodb_client.time.sleep", return_value=None)
@patch("earnings_agents.tools.mongodb_client.get_collection")
def test_upsert_raises_after_all_retries_exhausted(mock_get_col, mock_sleep):
    """When every attempt fails, the final exception propagates to the caller."""
    col = _make_collection(
        [RuntimeError("disk full")] * (_MONGO_MAX_RETRIES + 1)
    )
    mock_get_col.return_value = col

    with pytest.raises(RuntimeError, match="disk full"):
        upsert_earnings({"_id": "GOOGL_2026_latest", "ticker": "GOOGL"})

    assert col.update_one.call_count == _MONGO_MAX_RETRIES + 1
    assert mock_sleep.call_count == _MONGO_MAX_RETRIES


@patch("earnings_agents.tools.mongodb_client.time.sleep", return_value=None)
@patch("earnings_agents.tools.mongodb_client.get_collection")
def test_upsert_raises_immediately_on_missing_id(mock_get_col, mock_sleep):
    """ValueError is raised synchronously when _id is absent — no retries."""
    mock_get_col.return_value = MagicMock()

    with pytest.raises(ValueError, match="_id"):
        upsert_earnings({"ticker": "AAPL"})

    mock_get_col.return_value.update_one.assert_not_called()
    mock_sleep.assert_not_called()
