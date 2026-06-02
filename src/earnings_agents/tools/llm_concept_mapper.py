"""LLM-assisted semantic concept mapper for XBRL extraction.

Provides :func:`llm_map_concepts`, which asks the LLM to match extracted
metric keys to XBRL concepts from a normalize_data registry.

Extracted from ``extraction/concept_mapper.py`` so the LLM call, guardrails,
and prompt construction are independently testable and reusable.
"""
from __future__ import annotations

import json as _json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_LLM_MAP_PROMPT = """\
You are a financial concept mapper.

Below are metric keys extracted from an earnings press release and a list of
target concepts (XBRL tag + display label + concept_id) that we want to map to.

For each target concept, decide which extracted key best matches it (if any).

Rules:
  1. Each extracted key may be assigned to AT MOST ONE concept.
  2. Only assign a key when you are confident -- do NOT guess.
  3. If no extracted key fits a concept, return null for that concept.
  4. Do not invent new keys; only use keys from the extracted list.

Extracted metric keys:
{extracted_keys}

Target concepts:
{concept_rows}

Return ONLY a flat JSON object mapping concept_id -> matched extracted key (or null):
{{"<concept_id>": "<extracted_key_or_null>", ...}}
"""


def llm_map_concepts(
    extracted_keys: list[str],
    unmapped_concepts: list[dict],
    llm: Any,
    already_used_keys: set[str] | None = None,
) -> dict[str, str]:
    """Ask the LLM to semantically match *extracted_keys* to *unmapped_concepts*.

    Returns a dict of ``concept_id -> extracted_key`` for confident matches only.
    Null/missing entries from the LLM response are silently dropped.

    Guardrails
    ----------
    - LLM can only pick from the supplied *extracted_keys* list (no hallucination).
    - Each extracted key is used at most once (first assignment wins).
    - Keys in *already_used_keys* (Tier-1 deterministic matches) are excluded
      from the pool so the LLM cannot reassign them to a different concept.
    - Non-string or null LLM return values are discarded.
    """
    if not extracted_keys or not unmapped_concepts:
        return {}

    keys_block = "\n".join(f'  - "{k}"' for k in extracted_keys)
    rows_block = "\n".join(
        f'  - concept_id: "{c["_id"]}"  '
        f'GAAP: {c.get("concept", "")}  '
        f'label: "{c.get("label", "")}"'
        for c in unmapped_concepts
    )
    prompt = _LLM_MAP_PROMPT.format(
        extracted_keys=keys_block,
        concept_rows=rows_block,
    )
    try:
        raw = llm.invoke(prompt)
        if hasattr(raw, "content"):
            raw = raw.content
        raw = str(raw).strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        mapping: dict = _json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM concept mapping call failed: %s", exc)
        return {}

    valid_keys = set(extracted_keys)
    used_keys: set[str] = set(already_used_keys) if already_used_keys else set()
    result: dict[str, str] = {}
    for concept_id, matched_key in mapping.items():
        if not isinstance(matched_key, str):
            continue  # null or wrong type
        if matched_key not in valid_keys:
            logger.debug(
                "llm_map_concepts: ignoring hallucinated key %r for concept %s",
                matched_key, concept_id,
            )
            continue
        if matched_key in used_keys:
            logger.debug(
                "llm_map_concepts: key %r already used, skipping concept %s",
                matched_key, concept_id,
            )
            continue
        result[concept_id] = matched_key
        used_keys.add(matched_key)
    return result
