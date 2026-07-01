"""Unit tests for normalize_data_client helpers (no DB required)."""
import pytest
from unittest.mock import MagicMock, patch
from earnings_agents.tools.normalize_data_client import (
    _clean_label,
    _extract_member_tag,
    compute_fiscal_period,
    detect_period_type,
    parse_period_end_date,
    parse_period_start_date,
)
from datetime import date


# ── _clean_label ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_head, expected_member", [
    # Plain label — no member
    ("Net sales", "Net sales", ""),
    # Dimensional label with member concept appended after blank lines
    ("Net sales\n\n\nus-gaap:ProductMember", "Net sales", "Product"),
    # Label that IS only a concept reference — should produce empty head
    ("us-gaap:Revenues", "", ""),
    # Whitespace-only head
    ("  \n\nus-gaap:Revenues", "", ""),
])
def test_clean_label(raw, expected_head, expected_member):
    head, member = _clean_label(raw)
    assert head == expected_head
    assert member == expected_member


# ── _extract_member_tag ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected_tag", [
    # Dimensional label
    ("Net sales\n\n\nus-gaap:ProductMember", "us-gaap:ProductMember"),
    # No member
    ("Net sales", ""),
    # Plain concept with no Member suffix — not a member tag
    ("us-gaap:Revenues", ""),
    # Multiple references — first Member wins
    ("Fee income\n\nus-gaap:MembershipMember\nus-gaap:ProductMember", "us-gaap:MembershipMember"),
])
def test_extract_member_tag(raw, expected_tag):
    assert _extract_member_tag(raw) == expected_tag


# ── detect_period_type ────────────────────────────────────────────────────────

@pytest.mark.parametrize("period_str, expected", [
    # Quarterly
    ("Three Months Ended March 31, 2026",       "quarterly"),
    ("Thirteen Weeks Ended May 3, 2025",         "quarterly"),
    ("Quarter Ended June 30, 2025",              "quarterly"),
    ("Three months ended January 31, 2026",      "quarterly"),
    # Annual
    ("Year Ended December 31, 2025",             "annual"),
    ("Twelve Months Ended December 31, 2025",    "annual"),
    ("52 Weeks Ended February 1, 2025",          "annual"),
    ("53 Weeks Ended February 3, 2024",          "annual"),
    ("Full Year Ended January 28, 2023",         "annual"),
    ("Annual Period Ended June 30, 2025",        "annual"),
])
def test_detect_period_type(period_str, expected):
    assert detect_period_type(period_str) == expected


# ── parse_period_end_date ─────────────────────────────────────────────────────

@pytest.mark.parametrize("period_str, expected", [
    ("Three Months Ended March 31, 2026",  date(2026, 3, 31)),
    ("Year Ended December 31, 2025",       date(2025, 12, 31)),
    ("Thirteen Weeks Ended May 3, 2025",   date(2025, 5, 3)),
    ("no date here",                       None),
])
def test_parse_period_end_date(period_str, expected):
    assert parse_period_end_date(period_str) == expected


# ── compute_fiscal_period ─────────────────────────────────────────────────────

@pytest.mark.parametrize("end_date, fy_end_month, expected_fy, expected_q", [
    # MSFT (June FY end): March 31 2026 → FY2026 Q3
    (date(2026, 3, 31), 6, 2026, 3),
    # MSFT: June 30 2026 → FY2026 Q4
    (date(2026, 6, 30), 6, 2026, 4),
    # MSFT: September 30 2025 → FY2026 Q1
    (date(2025, 9, 30), 6, 2026, 1),
    # Calendar FY (December): December 31 2025 → FY2025 Q4
    (date(2025, 12, 31), 12, 2025, 4),
    # Calendar FY: March 31 2026 → FY2026 Q1
    (date(2026, 3, 31), 12, 2026, 1),
    # BJ (Jan FY end): January 31 2026 → FY2026 Q4
    (date(2026, 1, 31), 1, 2026, 4),
    # BJ: November 1 2025 → FY2026 Q4
    (date(2025, 11, 1), 1, 2026, 4),
])
def test_compute_fiscal_period(end_date, fy_end_month, expected_fy, expected_q):
    fy, q = compute_fiscal_period(end_date, fy_end_month)
    assert fy == expected_fy
    assert q == expected_q


# ── compute_fiscal_period WITH period_str (52-week + explicit durations) ─────

@pytest.mark.parametrize("period_str, end_date, fy_end_month, expected_fy, expected_q", [
    # BJ (Jan FY end, 52-week calendar) — week-count overrides calendar-month calc
    ("Thirteen Weeks Ended May 2, 2026",        date(2026,  5,  2), 1, 2027, 1),
    ("Twenty-Six Weeks Ended August 1, 2026",   date(2026,  8,  1), 1, 2027, 2),
    ("Thirty-Nine Weeks Ended November 1, 2025",date(2025, 11,  1), 1, 2026, 3),
    # Digit forms
    ("13 Weeks Ended May 2, 2026",              date(2026,  5,  2), 1, 2027, 1),
    ("26 Weeks Ended August 1, 2026",           date(2026,  8,  1), 1, 2027, 2),
    ("39 Weeks Ended November 1, 2025",         date(2025, 11,  1), 1, 2026, 3),
    # Month-based (Six/Nine unambiguous; Three falls back to date math)
    ("Six Months Ended June 30, 2026",          date(2026,  6, 30), 12, 2026, 2),
    ("Nine Months Ended September 30, 2026",    date(2026,  9, 30), 12, 2026, 3),
    # Ordinal labels
    ("First Quarter Ended March 31, 2026",      date(2026,  3, 31), 12, 2026, 1),
    ("Second Quarter Ended June 30, 2026",      date(2026,  6, 30), 12, 2026, 2),
    ("Q3 Results Ended September 30, 2026",     date(2026,  9, 30), 12, 2026, 3),
    # MSFT (June FY end) with explicit period
    ("Nine Months Ended March 31, 2026",        date(2026,  3, 31),  6, 2026, 3),
])
def test_compute_fiscal_period_with_period_str(
    period_str, end_date, fy_end_month, expected_fy, expected_q
):
    fy, q = compute_fiscal_period(end_date, fy_end_month, period_str)
    assert fy == expected_fy, f"FY mismatch for {period_str!r}: got {fy}, want {expected_fy}"
    assert q == expected_q, f"Q mismatch for {period_str!r}: got {q}, want {expected_q}"


# ── parse_period_start_date ───────────────────────────────────────────────────

@pytest.mark.parametrize("period_str, end_date, expected_start", [
    # Week-based (13w × 7 = 91 days back + 1)
    ("Thirteen Weeks Ended May 2, 2026",         date(2026, 5, 2),  date(2026, 2, 1)),
    ("Twenty-Six Weeks Ended August 1, 2026",    date(2026, 8, 1),  date(2026, 2, 1)),
    ("Thirty-Nine Weeks Ended November 1, 2026", date(2026, 11, 1), date(2026, 2, 2)),
    # Month-based
    ("Three Months Ended March 31, 2026",        date(2026, 3, 31), date(2026, 1, 1)),
    ("Six Months Ended June 30, 2026",           date(2026, 6, 30), date(2026, 1, 1)),
    ("Nine Months Ended September 30, 2026",     date(2026, 9, 30), date(2026, 1, 1)),
    # Cross-year boundary (Q1 for Nov FY-end company: Dec–Feb)
    ("Three Months Ended February 28, 2026",     date(2026, 2, 28), date(2025, 12, 1)),
    # Unrecognised string → None
    ("Year Ended December 31, 2025",             date(2025, 12, 31), None),
    ("No date here",                             date(2026, 1, 1),  None),
])
def test_parse_period_start_date(period_str, end_date, expected_start):
    result = parse_period_start_date(period_str, end_date)
    assert result == expected_start, (
        f"start_date mismatch for {period_str!r}: got {result}, want {expected_start}"
    )


# ── _infer_period_type (load_company_concepts_node helper) ───────────────────

from earnings_agents.nodes.load_company_concepts_node import _infer_period_type  # noqa: E402


@pytest.mark.parametrize("sec_report_date, fy_end_month, expected", [
    # Report date month matches FY end month → annual
    ("2025-12-31", 12, "annual"),
    ("2025-06-30", 6,  "annual"),
    ("2026-01-31", 1,  "annual"),
    # Mismatch → quarterly
    ("2025-09-30", 12, "quarterly"),
    ("2025-03-31", 6,  "quarterly"),
    # Missing inputs → quarterly
    (None, 12,         "quarterly"),
    ("2025-12-31", None, "quarterly"),
    (None, None,       "quarterly"),
    # Malformed date → quarterly
    ("not-a-date", 12, "quarterly"),
])
def test_infer_period_type(sec_report_date, fy_end_month, expected):
    assert _infer_period_type(sec_report_date, fy_end_month) == expected


@pytest.mark.parametrize("sec_report_date, fy_end_month, fy_end_code, expected", [
    # 52/53-week filer (FYE "nearest Jan 31"): year-end drifts into early Feb.
    ("2026-02-03", 1, "0131", "annual"),
    # …and into late Jan of the prior calendar month boundary.
    ("2026-01-28", 1, "0131", "annual"),
    # 52/53-week filer (FYE "nearest Apr 30"): drift into early May.
    ("2026-05-02", 4, "0430", "annual"),
    # Drift just past the 7-day tolerance is NOT annual.
    ("2026-02-08", 1, "0131", "quarterly"),
    # A genuine quarter-end (~3 months away) is never caught by proximity.
    ("2025-10-31", 1, "0131", "quarterly"),
    # Without a day code, only month-equality applies → drift stays quarterly.
    ("2026-02-03", 1, None, "quarterly"),
])
def test_infer_period_type_52_53_week_drift(
    sec_report_date, fy_end_month, fy_end_code, expected
):
    assert _infer_period_type(sec_report_date, fy_end_month, fy_end_code) == expected


# ── load_company_concepts_node period-type signal combination ────────────────

def test_load_concepts_cadence_overrides_wrong_date_signal():
    """Cadence (last stored = Q3 → annual) fixes a mis-dated annual release.

    Mirrors the Oracle failure: the 8-K announcement date (June 10, one month
    past the May-31 fiscal year-end) makes the date signal say "quarterly", but
    the last stored period is Q3, so the next release must be annual.
    """
    from earnings_agents.nodes import load_company_concepts_node as node

    company = {"cik": "0001341439", "fiscal_year_end_month": 5,
               "fiscal_year_end_code": "0531", "name": "ORACLE CORP"}
    state = {"ticker": "ORCL", "sec_report_date": "2026-06-10"}

    with (
        patch.object(node, "get_company_by_ticker", return_value=company),
        patch.object(node, "get_next_period_type", return_value="annual"),
        patch.object(node, "get_statement_concepts", return_value=[]),
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert result["detected_period_type"] == "annual"


def test_load_concepts_date_signal_catches_db_gap():
    """When a quarter was skipped (cadence says quarterly) but the date lands on
    the fiscal year-end, the date signal still classifies the release as annual."""
    from earnings_agents.nodes import load_company_concepts_node as node

    company = {"cik": "000123", "fiscal_year_end_month": 12,
               "fiscal_year_end_code": "1231", "name": "Example Co"}
    state = {"ticker": "EXMP", "sec_report_date": "2025-12-31"}

    with (
        patch.object(node, "get_company_by_ticker", return_value=company),
        # Cadence thinks the next release is a quarter (e.g. last stored was Q1).
        patch.object(node, "get_next_period_type", return_value="quarterly"),
        patch.object(node, "get_statement_concepts", return_value=[]),
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert result["detected_period_type"] == "annual"


def test_load_concepts_quarterly_when_both_signals_agree():
    """A genuine mid-year quarter stays quarterly when neither signal says annual."""
    from earnings_agents.nodes import load_company_concepts_node as node

    company = {"cik": "000123", "fiscal_year_end_month": 12,
               "fiscal_year_end_code": "1231", "name": "Example Co"}
    state = {"ticker": "EXMP", "sec_report_date": "2025-09-30"}

    with (
        patch.object(node, "get_company_by_ticker", return_value=company),
        patch.object(node, "get_next_period_type", return_value="quarterly"),
        patch.object(node, "get_statement_concepts", return_value=[]),
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert result["detected_period_type"] == "quarterly"


def test_load_concepts_after_q4_record_is_next_fy_q1_quarterly():
    """CRWD-style: last stored Q4 (=annual, year closed) + new FY's Q1 filing.

    The new filing (Q1 of the next fiscal year) must be detected as quarterly,
    NOT annual — a Q4 record is the closed year, so the next release is Q1.
    Exercises the REAL cadence (only get_latest_period is mocked).
    """
    from datetime import datetime, timezone
    from earnings_agents.nodes import load_company_concepts_node as node
    from earnings_agents.tools import normalize_data_client as ndc

    company = {"cik": "0001535527", "fiscal_year_end_month": 1,
               "fiscal_year_end_code": "0131", "name": "CROWDSTRIKE HOLDINGS, INC."}
    latest = {
        "period_type": "quarterly", "fiscal_year": 2026, "quarter": 4,
        "end_date": datetime(2026, 1, 31, tzinfo=timezone.utc),
    }
    state = {"ticker": "CRWD", "sec_report_date": "2026-06-03"}

    with (
        patch.object(node, "get_company_by_ticker", return_value=company),
        patch.object(ndc, "get_latest_period", return_value=latest),
        patch.object(node, "get_statement_concepts", return_value=[]),
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert result["detected_period_type"] == "quarterly"


# ── get_statement_concepts uses correct collection ───────────────────────────

def test_get_statement_concepts_uses_quarterly_collection_by_default():
    """get_statement_concepts(...) without period_type queries normalized_concepts_quarterly."""
    from earnings_agents.tools import normalize_data_client as ndc

    # find() must return something with a .sort() method that itself is iterable
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter([])
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        ndc.get_statement_concepts("000123", statement_types=["income_statement"])

    called_col = mock_db.__getitem__.call_args[0][0]
    assert called_col == "normalized_concepts_quarterly"


def test_get_statement_concepts_uses_annual_collection_when_specified():
    """get_statement_concepts(..., period_type='annual') queries normalized_concepts_annual."""
    from earnings_agents.tools import normalize_data_client as ndc

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = iter([])
    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        ndc.get_statement_concepts(
            "000123", statement_types=["income_statement"], period_type="annual"
        )

    called_col = mock_db.__getitem__.call_args[0][0]
    assert called_col == "normalized_concepts_annual"


# ── get_latest_period returns correct structure ───────────────────────────────

def test_upsert_concept_values_separates_quarterly_and_annual_filters():
    """Upserts include period-type information so annual vs quarterly rows stay distinct."""
    from datetime import date
    from earnings_agents.tools import normalize_data_client as ndc

    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)

        result = ndc.upsert_concept_values(
            cik="000123",
            company_name="Example Co",
            concept_metrics={"507f1f77bcf86cd799439011": 1.23},
            period_str="Three Months Ended March 31, 2026",
            fiscal_year_end_month=12,
            report_date=date(2026, 3, 31),
        )

    assert result == 1
    ops = mock_collection.bulk_write.call_args[0][0]
    assert len(ops) == 1
    state = ops[0].__getstate__()[1]
    assert state["_filter"] == {
        "concept_id": ndc.ObjectId("507f1f77bcf86cd799439011"),
        "reporting_period.end_date": ndc.datetime(2026, 3, 31, 0, 0, 0, tzinfo=ndc.timezone.utc),
        "reporting_period.form_type": "10-Q",
        "reporting_period.quarter": 1,
    }
    # earning_data flag must be present in the upserted document
    update_doc = state["_doc"]["$set"]
    assert update_doc["earning_data"] is True


def test_upsert_concept_values_uses_annual_collection_for_annual_periods():
    """Annual inserts target the annual collection and keep annual form_type in the filter."""
    from datetime import date
    from earnings_agents.tools import normalize_data_client as ndc

    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)

        ndc.upsert_concept_values(
            cik="000123",
            company_name="Example Co",
            concept_metrics={"507f1f77bcf86cd799439011": 1.23},
            period_str="Year Ended December 31, 2025",
            fiscal_year_end_month=12,
            report_date=date(2025, 12, 31),
        )

    called_collection = mock_db.__getitem__.call_args_list[0][0][0]
    assert called_collection == "concept_values_annual"
    ops = mock_collection.bulk_write.call_args[0][0]
    state = ops[0].__getstate__()[1]
    assert state["_filter"]["reporting_period.form_type"] == "10-K"
    assert "reporting_period.quarter" not in state["_filter"]
    # earning_data flag must be present; annual reporting_period must have no start_date
    update_doc = state["_doc"]["$set"]
    assert update_doc["earning_data"] is True
    assert "start_date" not in update_doc["reporting_period"]


def test_upsert_concept_values_period_type_override_is_authoritative():
    """An explicit period_type_override wins over month-equality and __period__.

    Guards the single-source-of-truth contract: ``detected_period_type`` from
    the upstream node routes the save collection so the prompt's column
    selection and the persisted period type can never diverge.
    """
    from datetime import date
    from earnings_agents.tools import normalize_data_client as ndc

    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)

        # period_str/report_date both look quarterly, but the override says annual.
        ndc.upsert_concept_values(
            cik="000123",
            company_name="Example Co",
            concept_metrics={"507f1f77bcf86cd799439011": 1.23},
            period_str="Three Months Ended March 31, 2026",
            fiscal_year_end_month=12,
            report_date=date(2026, 3, 31),
            period_type_override="annual",
        )

    called_collection = mock_db.__getitem__.call_args_list[0][0][0]
    assert called_collection == "concept_values_annual"
    ops = mock_collection.bulk_write.call_args[0][0]
    state = ops[0].__getstate__()[1]
    assert state["_filter"]["reporting_period.form_type"] == "10-K"
    assert "reporting_period.quarter" not in state["_filter"]



def test_get_latest_period_returns_most_recent_across_both_collections():
    """get_latest_period picks the most recent end_date from both collections."""
    from datetime import datetime, timezone
    from earnings_agents.tools import normalize_data_client as ndc

    quarterly_doc = {
        "reporting_period": {
            "end_date": datetime(2025, 9, 30, tzinfo=timezone.utc),
            "fiscal_year": 2026,
            "quarter": 1,
        }
    }
    annual_doc = {
        "reporting_period": {
            "end_date": datetime(2025, 6, 30, tzinfo=timezone.utc),
            "fiscal_year": 2025,
        }
    }

    def make_collection(doc):
        col = MagicMock()
        col.find_one.return_value = doc
        return col

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(side_effect=lambda name: {
        "concept_values_quarterly": make_collection(quarterly_doc),
        "concept_values_annual": make_collection(annual_doc),
    }[name])

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_latest_period("000123")

    assert result is not None
    assert result["period_type"] == "quarterly"
    assert result["fiscal_year"] == 2026
    assert result["quarter"] == 1


def test_get_latest_period_returns_none_when_no_data():
    """get_latest_period returns None when both collections have no data for the CIK."""
    from earnings_agents.tools import normalize_data_client as ndc

    mock_col = MagicMock()
    mock_col.find_one.return_value = None
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_col)

    with patch.object(ndc, "_get_client") as mock_client:
        mock_client.return_value.__getitem__ = MagicMock(return_value=mock_db)
        result = ndc.get_latest_period("000123")

    assert result is None


# ── get_next_period_type (filing-cadence signal) ─────────────────────────────

@pytest.mark.parametrize("latest, expected", [
    # After Q3 the next release is always the annual report (no standalone Q4).
    ({"period_type": "quarterly", "fiscal_year": 2026, "quarter": 3}, "annual"),
    # Mid-year quarters → the next standalone quarter.
    ({"period_type": "quarterly", "fiscal_year": 2026, "quarter": 1}, "quarterly"),
    ({"period_type": "quarterly", "fiscal_year": 2026, "quarter": 2}, "quarterly"),
    # A legacy Q4 record == the annual (year closed) → next is next-FY Q1.
    # This must NOT resolve to "annual" (that would mislabel the new FY's Q1).
    ({"period_type": "quarterly", "fiscal_year": 2026, "quarter": 4}, "quarterly"),
    # After the annual report → Q1 of the next fiscal year.
    ({"period_type": "annual", "fiscal_year": 2025, "quarter": None}, "quarterly"),
    # Defensive: a missing/zero quarter is treated as not-yet-Q3.
    ({"period_type": "quarterly", "fiscal_year": 2026, "quarter": None}, "quarterly"),
])
def test_get_next_period_type(latest, expected):
    from earnings_agents.tools import normalize_data_client as ndc

    with patch.object(ndc, "get_latest_period", return_value=latest):
        assert ndc.get_next_period_type("000123") == expected


def test_get_next_period_type_returns_none_without_prior_data():
    """No stored period → cadence cannot be inferred → None."""
    from earnings_agents.tools import normalize_data_client as ndc

    with patch.object(ndc, "get_latest_period", return_value=None):
        assert ndc.get_next_period_type("000123") is None


def test_get_next_period_type_gated_skips_non_newer_release():
    """A non-newer release does NOT advance the cycle — returns the stored type.

    Re-processing the already-stored Q3 release (same or older period-end) must
    report the stored period's own type ("quarterly") rather than advancing to
    "annual" — otherwise the Q3 release would be reclassified as annual,
    duplicating it across the quarterly/annual collections.
    """
    from datetime import date, datetime, timezone
    from earnings_agents.tools import normalize_data_client as ndc

    latest = {
        "period_type": "quarterly", "fiscal_year": 2026, "quarter": 3,
        "end_date": datetime(2026, 3, 31, tzinfo=timezone.utc),
    }
    with patch.object(ndc, "get_latest_period", return_value=latest):
        # Exact re-run (same period-end) → no advance → stored type.
        assert ndc.get_next_period_type("000123", date(2026, 3, 31)) == "quarterly"
        # Older release → no advance → stored type.
        assert ndc.get_next_period_type("000123", date(2025, 12, 31)) == "quarterly"


def test_get_next_period_type_gated_allows_genuine_next_annual():
    """A genuinely newer release after Q3 still resolves to annual."""
    from datetime import date, datetime, timezone
    from earnings_agents.tools import normalize_data_client as ndc

    latest = {
        "period_type": "quarterly", "fiscal_year": 2026, "quarter": 3,
        "end_date": datetime(2026, 3, 31, tzinfo=timezone.utc),
    }
    with patch.object(ndc, "get_latest_period", return_value=latest):
        # New annual period-end (~3 months later) → annual.
        assert ndc.get_next_period_type("000123", date(2026, 6, 30)) == "annual"


def test_get_next_period_type_rerun_of_annual_stays_annual():
    """Re-processing the stored annual release stays annual (no spurious advance)."""
    from datetime import date, datetime, timezone
    from earnings_agents.tools import normalize_data_client as ndc

    latest = {
        "period_type": "annual", "fiscal_year": 2026, "quarter": None,
        "end_date": datetime(2026, 5, 31, tzinfo=timezone.utc),
    }
    with patch.object(ndc, "get_latest_period", return_value=latest):
        # Same period-end → no advance → stored type "annual".
        assert ndc.get_next_period_type("000123", date(2026, 5, 31)) == "annual"
        # Strictly newer → advance: after annual the next release is Q1.
        assert ndc.get_next_period_type("000123", date(2026, 8, 31)) == "quarterly"


def test_load_concepts_rerun_q3_not_reclassified_as_annual():
    """Re-running the same Q3 release must stay quarterly, not flip to annual.

    Exercises the REAL cadence gate (only get_latest_period is mocked) to prove
    the cross-collection duplicate path is closed end-to-end at the node.
    """
    from datetime import datetime, timezone
    from earnings_agents.nodes import load_company_concepts_node as node
    from earnings_agents.tools import normalize_data_client as ndc

    company = {"cik": "0000789019", "fiscal_year_end_month": 6,
               "fiscal_year_end_code": "0630", "name": "MICROSOFT CORP"}
    latest = {
        "period_type": "quarterly", "fiscal_year": 2026, "quarter": 3,
        "end_date": datetime(2026, 3, 31, tzinfo=timezone.utc),
    }
    state = {"ticker": "MSFT", "sec_report_date": "2026-03-31"}

    with (
        patch.object(node, "get_company_by_ticker", return_value=company),
        patch.object(ndc, "get_latest_period", return_value=latest),
        patch.object(node, "get_statement_concepts", return_value=[]),
    ):
        result = node.load_company_concepts_node(state)  # type: ignore[arg-type]

    assert result["detected_period_type"] == "quarterly"



# ── _is_period_already_stored (CLI helper) ────────────────────────────────────

from earnings_agents.cli.earnings import _is_period_already_stored  # noqa: E402


def _make_latest_period(end_date_str: str) -> dict:
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(end_date_str).replace(tzinfo=timezone.utc)
    m = MagicMock()
    m.date.return_value = dt.date()
    return {
        "period_type": "quarterly",
        "fiscal_year": 2026,
        "quarter": 1,
        "end_date": m,
    }


def test_is_period_already_stored_match():
    """Returns True when EDGAR sec_report_date equals the latest stored end_date."""
    company = {"cik": "000123", "fiscal_year_end_month": 12, "fiscal_year_end_code": "1231"}
    latest = _make_latest_period("2025-09-30")

    with patch("earnings_agents.cli.earnings._is_period_already_stored.__module__"):
        pass  # ensure import path works

    with (
        patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker", return_value=company),
        patch("earnings_agents.tools.normalize_data_client.get_latest_period", return_value=latest),
    ):
        # Import inside patch context so the patched functions are used
        from importlib import import_module
        mod = import_module("earnings_agents.cli.earnings")
        result = mod._is_period_already_stored("AAPL", "2025-09-30")

    assert result is True


def test_is_period_already_stored_mismatch():
    """Returns False when EDGAR sec_report_date differs from stored end_date (new 8-K available)."""
    company = {"cik": "000123", "fiscal_year_end_month": 12, "fiscal_year_end_code": "1231"}
    latest = _make_latest_period("2025-06-30")  # Q2 stored

    with (
        patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker", return_value=company),
        patch("earnings_agents.tools.normalize_data_client.get_latest_period", return_value=latest),
    ):
        from importlib import import_module
        mod = import_module("earnings_agents.cli.earnings")
        result = mod._is_period_already_stored("AAPL", "2025-09-30")  # Q3 EDGAR

    assert result is False


def test_is_period_already_stored_no_data_yet():
    """Returns False when normalize_data has no data at all for the company."""
    company = {"cik": "000123", "fiscal_year_end_month": 12}

    with (
        patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker", return_value=company),
        patch("earnings_agents.tools.normalize_data_client.get_latest_period", return_value=None),
    ):
        from importlib import import_module
        mod = import_module("earnings_agents.cli.earnings")
        result = mod._is_period_already_stored("AAPL", "2025-09-30")

    assert result is False


def test_is_period_already_stored_company_not_in_db():
    """Returns False (don't skip) when ticker not found in normalize_data."""
    with patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker", return_value=None):
        from importlib import import_module
        mod = import_module("earnings_agents.cli.earnings")
        result = mod._is_period_already_stored("UNKN", "2025-09-30")

    assert result is False


def test_is_period_already_stored_missing_args():
    """Returns False when ticker or sec_report_date is falsy — never skip on missing data."""
    from importlib import import_module
    mod = import_module("earnings_agents.cli.earnings")
    assert mod._is_period_already_stored("", "2025-09-30") is False
    assert mod._is_period_already_stored("AAPL", None) is False
    assert mod._is_period_already_stored("", None) is False


def test_print_latest_data_status_reports_next_quarter_needed():
    """The coverage summary distinguishes quarterly periods and shows the next 8-K needed."""
    from earnings_agents.cli.earnings import _print_latest_data_status

    lines = []

    latest = {
        "period_type": "quarterly",
        "fiscal_year": 2026,
        "quarter": 1,
        "end_date": MagicMock(strftime=MagicMock(return_value="2025-09-30")),
    }

    with (
        patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker", return_value={"cik": "000123", "fiscal_year_end_month": 12, "fiscal_year_end_code": "1231"}),
        patch("earnings_agents.tools.normalize_data_client.get_latest_period", return_value=latest),
    ):
        _print_latest_data_status([{"ticker": "AAPL", "company_name": "Apple Inc"}], printer=lines.append)

    joined = "\n".join(lines)
    assert "last stored: FY2026 Q1" in joined
    assert "need: FY2026 Q2 8-K" in joined


def test_print_latest_data_status_q3_requires_annual_not_q4():
    """After Q3, the next 8-K is the Annual (which covers Q4+full year). There is no Q4 quarterly 8-K."""
    from earnings_agents.cli.earnings import _print_latest_data_status

    lines = []

    latest = {
        "period_type": "quarterly",
        "fiscal_year": 2026,
        "quarter": 3,
        "end_date": MagicMock(strftime=MagicMock(return_value="2026-09-30")),
    }

    with (
        patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker", return_value={"cik": "000123", "fiscal_year_end_month": 12, "fiscal_year_end_code": "1231"}),
        patch("earnings_agents.tools.normalize_data_client.get_latest_period", return_value=latest),
    ):
        _print_latest_data_status([{"ticker": "AAPL", "company_name": "Apple Inc"}], printer=lines.append)

    joined = "\n".join(lines)
    assert "last stored: FY2026 Q3" in joined
    # Must show Annual, NOT Q4
    assert "Annual 8-K" in joined
    assert "1231" in joined
    assert "Q4 8-K" not in joined


def test_print_latest_data_status_reports_next_annual_needed():
    """The coverage summary distinguishes annual periods and points to the next fiscal-year 8-K."""
    from earnings_agents.cli.earnings import _print_latest_data_status

    lines = []

    latest = {
        "period_type": "annual",
        "fiscal_year": 2025,
        "quarter": None,
        "end_date": MagicMock(strftime=MagicMock(return_value="2025-12-31")),
    }

    with (
        patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker", return_value={"cik": "000123", "fiscal_year_end_month": 12, "fiscal_year_end_code": "1231"}),
        patch("earnings_agents.tools.normalize_data_client.get_latest_period", return_value=latest),
    ):
        _print_latest_data_status([{"ticker": "AAPL", "company_name": "Apple Inc"}], printer=lines.append)

    joined = "\n".join(lines)
    assert "last stored: FY2025 annual" in joined
    assert "need: FY2026 Q1 8-K" in joined


def test_is_period_already_stored_db_error_returns_false():
    """Returns False on DB error — fail safe, never skip on ambiguity."""
    with patch("earnings_agents.tools.normalize_data_client.get_company_by_ticker",
               side_effect=Exception("DB down")):
        from importlib import import_module
        mod = import_module("earnings_agents.cli.earnings")
        result = mod._is_period_already_stored("AAPL", "2025-09-30")

    assert result is False


def test_build_initial_state_skips_sec_path_without_history():
    """Returns a skipped state without querying EDGAR when no normalize_data history exists."""
    info = {"ticker": "AAPL", "company_name": "Apple Inc", "cik": "0000320193"}

    with (
        patch("earnings_agents.cli.earnings._has_existing_period_data", return_value=False),
        patch("earnings_agents.cli.earnings.get_latest_earnings_url") as mock_url,
    ):
        from importlib import import_module
        mod = import_module("earnings_agents.cli.earnings")
        state = mod._build_initial_state(info, printer=lambda *_: None)

    assert state["status"] == "skipped"
    assert "no existing normalize_data period data" in state["error"]
    mock_url.assert_not_called()


