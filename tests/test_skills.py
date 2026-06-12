"""Tests for ``analysis/skills.py`` — the failure-mode skill catalog."""
from __future__ import annotations

import pytest

from earnings_agents.analysis.skills import (
    SKILL_REGISTRY,
    Skill,
    compute_skill_effectiveness,
    finding_signature,
    iter_detectors,
    remediation_block,
    skill_by_id,
    skills_for_finding_types,
)

# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------

EXPECTED_DETECTOR_SKILLS = [
    "case_duplicate",
    "composite_key",
    "gaap_nongaap_leakage",
    "gross_profit_identity",
    "operating_vs_gross",
    "eps_dilution_ordering",
    "suspect_round",
    "opex_label_collision",
]
EXPECTED_SPECIAL_SKILLS = ["presence", "source_grounding"]


def test_registry_has_expected_ids():
    all_ids = [s.id for s in SKILL_REGISTRY]
    for expected in EXPECTED_DETECTOR_SKILLS + EXPECTED_SPECIAL_SKILLS:
        assert expected in all_ids, f"Missing skill id: {expected}"


def test_all_ids_unique():
    ids = [s.id for s in SKILL_REGISTRY]
    assert len(ids) == len(set(ids)), "Duplicate skill ids found"


def test_all_skills_have_non_empty_fields():
    for skill in SKILL_REGISTRY:
        assert skill.id, f"Skill missing id: {skill}"
        assert skill.title, f"Skill {skill.id!r} missing title"
        assert skill.finding_types, f"Skill {skill.id!r} has empty finding_types"
        assert skill.remediation.strip(), f"Skill {skill.id!r} has empty remediation"


def test_all_skills_are_frozen():
    skill = SKILL_REGISTRY[0]
    with pytest.raises((AttributeError, TypeError)):
        skill.id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# iter_detectors
# ---------------------------------------------------------------------------

def test_iter_detectors_count():
    detectors = iter_detectors()
    assert len(detectors) == len(EXPECTED_DETECTOR_SKILLS)


def test_iter_detectors_order_matches_registry():
    """Detectors must come out in SKILL_REGISTRY order (same as old CHECKER_REGISTRY)."""
    detector_skills = [s for s in SKILL_REGISTRY if s.detector is not None]
    detectors = iter_detectors()
    assert [d.__name__ for d in detectors] == [
        s.detector.__name__ for s in detector_skills
    ]


def test_iter_detectors_are_callable():
    for d in iter_detectors():
        assert callable(d)


def test_special_skills_have_no_detector():
    for sid in EXPECTED_SPECIAL_SKILLS:
        skill = skill_by_id(sid)
        assert skill is not None
        assert skill.detector is None, f"Special skill {sid!r} should have detector=None"


def test_detector_skills_have_detector():
    for sid in EXPECTED_DETECTOR_SKILLS:
        skill = skill_by_id(sid)
        assert skill is not None
        assert skill.detector is not None, f"Skill {sid!r} should have a detector"


# ---------------------------------------------------------------------------
# skill_by_id
# ---------------------------------------------------------------------------

def test_skill_by_id_found():
    skill = skill_by_id("gross_profit_identity")
    assert skill is not None
    assert skill.id == "gross_profit_identity"


def test_skill_by_id_missing():
    assert skill_by_id("nonexistent_skill_xyz") is None


# ---------------------------------------------------------------------------
# skills_for_finding_types
# ---------------------------------------------------------------------------

def test_skills_for_finding_types_identity_violation():
    matched = skills_for_finding_types(["identity_violation"])
    ids = [s.id for s in matched]
    assert "gross_profit_identity" in ids


def test_skills_for_finding_types_no_match():
    matched = skills_for_finding_types(["unknown_finding_type_xyz"])
    assert matched == []


def test_skills_for_finding_types_no_duplicates():
    # suspect_value is shared by multiple skills; each should appear once
    matched = skills_for_finding_types(["suspect_value"])
    ids = [s.id for s in matched]
    assert len(ids) == len(set(ids))


def test_skills_for_finding_types_order_follows_registry():
    all_types = [ft for s in SKILL_REGISTRY for ft in s.finding_types]
    matched = skills_for_finding_types(all_types)
    registry_order = [s.id for s in SKILL_REGISTRY]
    matched_order = [s.id for s in matched]
    prev_idx = -1
    for sid in matched_order:
        idx = registry_order.index(sid)
        assert idx > prev_idx, "skills_for_finding_types broke registry order"
        prev_idx = idx


# ---------------------------------------------------------------------------
# remediation_block
# ---------------------------------------------------------------------------

def test_remediation_block_returns_string():
    block = remediation_block(["identity_violation"])
    assert isinstance(block, str)
    assert len(block) > 0


def test_remediation_block_contains_skill_id():
    block = remediation_block(["case_duplicate"])
    assert "(case_duplicate)" in block


def test_remediation_block_empty_on_no_match():
    block = remediation_block(["unknown_type_xyz"])
    assert block == ""


def test_remediation_block_multiple_types():
    """Multiple finding types should expand to multiple bullet entries."""
    block = remediation_block(["case_duplicate", "composite_key"])
    assert "(case_duplicate)" in block
    assert "(composite_key)" in block


def test_remediation_block_bullet_format():
    """Each line must start with '- ('."""
    block = remediation_block(["suspect_round"])
    for line in block.splitlines():
        assert line.startswith("- ("), f"Unexpected line format: {line!r}"


# ---------------------------------------------------------------------------
# Skill.detect helper
# ---------------------------------------------------------------------------

def test_detect_returns_empty_for_none_detector():
    presence_skill = skill_by_id("presence")
    assert presence_skill is not None
    result = presence_skill.detect({})
    assert result == []


def test_detect_runs_checker():
    """case_duplicate skill should fire on a dict with only-case-differing keys."""
    skill = skill_by_id("case_duplicate")
    assert skill is not None
    metrics = {
        "Net Income": 100,
        "net income": 100,
    }
    findings = skill.detect(metrics)
    assert len(findings) >= 1
    assert all(f.type == "case_duplicate" for f in findings)


def test_detect_no_findings_on_clean_metrics():
    skill = skill_by_id("case_duplicate")
    assert skill is not None
    findings = skill.detect({"Net Income": 100, "Revenue": 500})
    assert findings == []


# ---------------------------------------------------------------------------
# Every skill's declared finding_types must be valid FindingType literals
# ---------------------------------------------------------------------------

def test_skills_emit_valid_finding_types():
    from earnings_agents.analysis.findings import FindingType
    import typing

    valid_types = set(typing.get_args(FindingType))

    for skill in SKILL_REGISTRY:
        for ft in skill.finding_types:
            assert ft in valid_types, (
                f"Skill {skill.id!r} declares finding type {ft!r} "
                f"which is not in FindingType Literal"
            )


# ---------------------------------------------------------------------------
# finding_signature
# ---------------------------------------------------------------------------

def test_finding_signature_uses_sorted_keys():
    f = {"type": "case_duplicate", "keys": ["Net income", "net income"]}
    g = {"type": "case_duplicate", "keys": ["net income", "Net income"]}
    assert finding_signature(f) == finding_signature(g)
    assert finding_signature(f) == "case_duplicate::Net income|net income"


def test_finding_signature_falls_back_to_evidence_metric():
    f = {
        "type": "missing_critical",
        "keys": [],
        "evidence": {"tier": 1, "metric": "Gross Profit"},
        "message": "Tier-1 metric not found: Gross Profit",
    }
    assert finding_signature(f) == "missing_critical::Gross Profit"


def test_finding_signature_falls_back_to_message():
    f = {"type": "suspect_round", "message": "Revenue is suspiciously round"}
    assert finding_signature(f) == "suspect_round::Revenue is suspiciously round"


def test_finding_signature_distinguishes_types():
    a = {"type": "missing_critical", "evidence": {"metric": "Revenue"}}
    b = {"type": "missing_expected", "evidence": {"metric": "Revenue"}}
    assert finding_signature(a) != finding_signature(b)


# ---------------------------------------------------------------------------
# compute_skill_effectiveness
# ---------------------------------------------------------------------------

def _presence(metric: str, ftype: str = "missing_critical") -> dict:
    return {"type": ftype, "evidence": {"metric": metric}, "message": f"missing {metric}"}


def test_effectiveness_resolved_finding():
    prev = [_presence("Gross Profit"), _presence("Revenue")]
    curr = [_presence("Revenue")]
    records = compute_skill_effectiveness(prev, curr)
    assert len(records) == 1
    rec = records[0]
    assert rec["finding_type"] == "missing_critical"
    assert rec["resolved"] == 1
    assert rec["persisted"] == 1
    assert rec["new"] == 0
    assert "presence" in rec["skills"]


def test_effectiveness_new_finding():
    prev: list[dict] = []
    curr = [_presence("Revenue")]
    records = compute_skill_effectiveness(prev, curr)
    assert records[0]["new"] == 1
    assert records[0]["resolved"] == 0
    assert records[0]["persisted"] == 0


def test_effectiveness_all_resolved_returns_record():
    prev = [_presence("Revenue")]
    curr: list[dict] = []
    records = compute_skill_effectiveness(prev, curr)
    assert len(records) == 1
    assert records[0]["resolved"] == 1


def test_effectiveness_no_movement_omits_type():
    # identical findings → persisted only, still reported (movement includes persisted)
    prev = [_presence("Revenue")]
    curr = [_presence("Revenue")]
    records = compute_skill_effectiveness(prev, curr)
    assert len(records) == 1
    assert records[0]["persisted"] == 1
    assert records[0]["resolved"] == 0
    assert records[0]["new"] == 0


def test_effectiveness_empty_inputs():
    assert compute_skill_effectiveness([], []) == []


def test_effectiveness_sorted_by_finding_type():
    prev = [
        _presence("Revenue", "suspect_round"),
        _presence("Gross Profit", "missing_critical"),
    ]
    curr: list[dict] = []
    records = compute_skill_effectiveness(prev, curr)
    types = [r["finding_type"] for r in records]
    assert types == sorted(types)


def test_effectiveness_groups_per_type():
    prev = [
        _presence("Revenue", "missing_critical"),
        _presence("EPS", "suspect_round"),
    ]
    curr = [_presence("Revenue", "missing_critical")]
    records = compute_skill_effectiveness(prev, curr)
    by_type = {r["finding_type"]: r for r in records}
    assert by_type["suspect_round"]["resolved"] == 1
    assert by_type["missing_critical"]["persisted"] == 1

