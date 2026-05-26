"""Unit tests for normalize_data_client helpers (no DB required)."""
import pytest
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
