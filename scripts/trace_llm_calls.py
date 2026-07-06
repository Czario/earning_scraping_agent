"""Trace all LLM calls for one ticker and print exact prompt + response.

Usage:
    uv run python scripts/trace_llm_calls.py --ticker SOFI

Monkeypatches every LLM .invoke() call so you see the full prompt and
response for: [roles], [extraction], [tier2].
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from unittest.mock import patch, MagicMock

# ── Patch LLM before importing anything that builds it ───────────────────────

_call_log: list[dict] = []
_call_counter = [0]

_DIVIDER = "=" * 80


def _wrap(original_llm_instance, label: str):
    """Wrap a real LLM instance so every .invoke() is logged."""
    original_invoke = original_llm_instance.invoke

    def _traced_invoke(prompt, *args, **kwargs):
        _call_counter[0] += 1
        n = _call_counter[0]

        print(f"\n{_DIVIDER}")
        print(f"  LLM CALL #{n}  —  {label}")
        print(_DIVIDER)
        print("── PROMPT ──")
        # Print full prompt, wrapping long lines for readability
        for line in str(prompt).splitlines():
            if len(line) > 120:
                print(textwrap.fill(line, width=120, subsequent_indent="    "))
            else:
                print(line)

        response = original_invoke(prompt, *args, **kwargs)

        raw = response.content if hasattr(response, "content") else str(response)
        print(f"\n── RESPONSE ──")
        for line in raw.splitlines():
            print(line)
        print(_DIVIDER)

        _call_log.append({"n": n, "label": label, "prompt": str(prompt), "response": raw})
        return response

    original_llm_instance.invoke = _traced_invoke
    return original_llm_instance


# ── Hook into build_llm ───────────────────────────────────────────────────────

import earnings_agents.llm_factory as _lf

_orig_build_llm = _lf.build_llm


def _patched_build_llm(*args, **kwargs):
    llm = _orig_build_llm(*args, **kwargs)
    # Determine the call label from the call stack context
    import traceback
    stack = "".join(traceback.format_stack())
    if "llm_identify_roles" in stack:
        label = "[roles]  identify_roles LLM"
    elif "llm_map_concepts" in stack:
        label = "[tier2]  semantic_mapping LLM"
    elif "build_llm" in stack and "extract_financial_metrics" in stack:
        label = "[llm]    extraction LLM"
    else:
        label = "[other]  LLM"
    return _wrap(llm, label)


_lf.build_llm = _patched_build_llm

# Also patch inside the node module (it imports build_llm at import time)
import earnings_agents.nodes.extract_financial_metrics as _efm
_efm.build_llm = _patched_build_llm

# ── Run the pipeline ──────────────────────────────────────────────────────────

from earnings_agents.workflow import build_graph
from earnings_agents.workflow_state import EarningsAgentState


def _run(ticker: str) -> None:
    from earnings_agents.company_registry import lookup_by_ticker
    from earnings_agents.cli.earnings import _build_initial_state

    print(f"\n{'#'*80}")
    print(f"#  LLM CALL TRACE — {ticker}")
    print(f"{'#'*80}\n")

    graph = build_graph()

    info = lookup_by_ticker(ticker)
    if not info:
        print(f"ERROR: ticker {ticker} not found in company registry")
        sys.exit(1)
    state = _build_initial_state(info)
    final = graph.invoke(state)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{_DIVIDER}")
    print(f"  SUMMARY — {len(_call_log)} LLM call(s) for {ticker}")
    print(_DIVIDER)
    for entry in _call_log:
        prompt_lines = entry["prompt"].count("\n")
        resp_lines = entry["response"].count("\n")
        print(f"  Call #{entry['n']}  {entry['label']}")
        print(f"           prompt: {len(entry['prompt'])} chars / {prompt_lines} lines")
        print(f"           response: {len(entry['response'])} chars / {resp_lines} lines")
    print()

    print(f"  Pipeline status: {final.get('status')}")
    cm = final.get("concept_metrics") or {}
    print(f"  Concepts mapped: {len(cm)}")
    print()

    # Show tier mapping breakdown
    all_tc = final.get("target_concepts") or []
    mapped_ids = set(cm.keys())
    unmapped = [c for c in all_tc if c["_id"] not in mapped_ids]
    print(f"  target_concepts: {len(all_tc)}")
    print(f"  mapped:          {len(mapped_ids)}")
    print(f"  unmapped:        {len(unmapped)}")
    if unmapped:
        print("  Unmapped concepts:")
        for c in unmapped:
            print(f"    - {c.get('label')} ({c.get('taxonomy_key')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SOFI")
    args = parser.parse_args()
    _run(args.ticker.upper())
