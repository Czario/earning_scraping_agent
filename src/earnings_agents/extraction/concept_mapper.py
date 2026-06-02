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

    Each line: ``  • "Label"  [concept_tag]``
    Listed in path order (income statement order).

    The label is the JSON key contract -- the LLM reliably echoes the quoted
    string verbatim.  The bracketed concept tag is a semantic grounding hint
    (the LLM knows US-GAAP taxonomy from training) and assists the post-
    extraction LLM mapping step.
    """
    lines: list[str] = []
    for c in target_concepts:
        label = c.get("label", "")
        if not label:
            continue
        concept = c.get("concept", "") or ""
        concept_lc = concept.lower()
        if any(concept_lc.startswith(p) for p in _TAXONOMY_PREFIXES):
            lines.append(f'  \u2022 "{label}"  [{concept}]')
        else:
            lines.append(f'  \u2022 "{label}"')
    return "\n".join(lines)
