"""LLM-assisted concept mapper: matches extracted metric keys to XBRL concepts."""
from __future__ import annotations

import logging

from earnings_agents.tools.llm_concept_mapper import llm_map_concepts as _llm_map_concepts_impl

logger = logging.getLogger(__name__)

_TAXONOMY_PREFIXES = ("us-gaap:", "ifrs-full:", "dei:", "srt:")


def _llm_map_concepts(
    extracted_keys: list[str],
    unmapped_concepts: list[dict],
    llm: object,
    already_used_keys: set[str] | None = None,
) -> dict[str, str]:
    """Backward-compatible wrapper — delegates to tools.llm_concept_mapper."""
    return _llm_map_concepts_impl(
        extracted_keys=extracted_keys,
        unmapped_concepts=unmapped_concepts,
        llm=llm,
        already_used_keys=already_used_keys,
    )


def _build_concept_prompt_list(target_concepts: list[dict]) -> str:
    """Format concept list for the targeted prompt.

    Each line: ``  \u2022 "Label"  [bracket_hint]``
    Listed in path order (income statement order).

    The label is the search hint the LLM uses to locate the right row in the
    document.  The bracketed tag is the JSON key contract AND a semantic
    grounding hint.  Strategy for building the hint:

    1. For concepts with a dotted ``path`` (dimensional children), prefix the
       label with the nearest parent concept's label so the LLM knows WHICH
       metric the dimension applies to, e.g.:
       ``"Americas Segment [Member]"  -- Revenue from Contract with Customer``
       This is the most informative hint; it resolves the ambiguity of bare
       member labels like "iPhone [Member]" (product revenue? other metric?).
    2. Always show ``[taxonomy_key]`` in brackets so the LLM has the XBRL
       concept name as additional grounding.  Company-specific prefixes such
       as ``aapl:`` are shown too -- the LLM knows these from training.
    """
    # Build a lookup: parent_path -> list of parent labels (for dimensional hints)
    # A concept's parent path is its path with the last ".xxx" suffix removed.
    # We prefer parents whose taxonomy_key starts with a known standard prefix
    # (us-gaap:, ifrs-full:, etc.) because those labels are most meaningful.
    path_to_labels: dict[str, list[str]] = {}
    for c in target_concepts:
        p = c.get("path") or ""
        lbl = c.get("label") or ""
        if p and lbl:
            path_to_labels.setdefault(p, []).append(lbl)

    def _parent_hint(path: str, taxonomy_key: str) -> str:
        """Return a short parent-context string for dimensional concepts."""
        if "." not in path:
            return ""
        parent_path = path.rsplit(".", 1)[0]
        parent_labels = path_to_labels.get(parent_path, [])
        # Prefer labels whose concept has a known standard-taxonomy key
        # (those are the "official" GAAP line items, not members/aliases).
        std_parents = [
            c.get("label", "")
            for c in target_concepts
            if c.get("path") == parent_path
            and any(
                (c.get("taxonomy_key") or "").lower().startswith(p)
                for p in _TAXONOMY_PREFIXES
            )
        ]
        chosen = std_parents[0] if std_parents else (parent_labels[0] if parent_labels else "")
        return f" -- {chosen}" if chosen else ""

    lines: list[str] = []
    for c in target_concepts:
        label = c.get("label", "")
        if not label:
            continue
        taxonomy_key = (c.get("taxonomy_key") or "").strip()
        concept = (c.get("concept") or "").strip()
        path = c.get("path") or ""

        parent_ctx = _parent_hint(path, taxonomy_key)

        # Bracket hint: prefer taxonomy_key (may equal concept for XBRL concepts).
        # Show any non-empty taxonomy_key -- company-specific prefixes (aapl:,
        # msft:, etc.) are meaningful to the LLM via training-time knowledge.
        hint = taxonomy_key or concept

        if hint:
            lines.append(f'  \u2022 "{label}"{parent_ctx}  [{hint}]')
        else:
            lines.append(f'  \u2022 "{label}"{parent_ctx}')
    return "\n".join(lines)

