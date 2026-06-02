---
name: diagnose
description: >
  Disciplined diagnosis loop for hard bugs in the earnings pipeline.
  Reproduce → minimise → hypothesise → instrument → fix → regression-test.
  Use when a node fails unexpectedly, extraction produces wrong/missing metrics,
  the extract↔analyze loop gets stuck, MongoDB upsert fails, or a ticker run
  produces incorrect output.
---

# Diagnose

A discipline for hard bugs in the earnings pipeline. Skip phases only when explicitly justified.

Before exploring code, read `CONTEXT.md` for domain terminology and any ADRs in `docs/adr/` for the area you're touching.

---

## Phase 1 — Build a feedback loop

This is the most important phase. Without a fast, deterministic, agent-runnable pass/fail signal you will not find the cause. Spend disproportionate effort here.

### Ways to construct one — try in roughly this order

1. **Failing pytest** — target the matching test file first:
   ```bash
   uv run pytest -q tests/test_<module>.py
   uv run pytest -q -k <node_or_function_name>
   ```
2. **Single-ticker verbose run** — exercise the real pipeline end-to-end:
   ```bash
   uv run earnings --ticker AAPL -v
   uv run earnings --source ir --ticker AAPL --ir-url "<url>" -v
   ```
3. **State inspection** — add a temporary `print(state)` or `logger.debug(state)` after the suspect node to dump `EarningsAgentState` fields.
4. **MongoDB document check** — inspect what was actually upserted:
   - `_id` format: `{TICKER}_{YEAR}_latest` (e.g. `AAPL_2024_latest`)
   - Check `status`, `error`, `findings`, `identity_warnings` fields.
5. **Isolated node call** — construct a minimal `EarningsAgentState` dict and call the node function directly in a throwaway script or pytest fixture.
6. **Chunk replay** — if the issue is in metric extraction, save `state["raw_text"]` or `state["raw_sections"]` to a fixture file and replay through `extract_financial_metrics_node`.

### Common pipeline failure modes

| Symptom | First suspect |
|---|---|
| `status == "failed"`, `error` set | Node that set the error; trace backwards from `workflow.py` routing |
| Tier-1 metric missing repeatedly | `extract_financial_metrics_node` prompt, chunk boundaries, or `__scale__` mis-set |
| Loop hits `MAX_EXTRACTION_ATTEMPTS` | `analyze_metrics_node` generating same high-severity findings each pass; check `previous_high_finding_keys` |
| `needs_reextract` stuck `True` | `analyze_metrics_node` routing logic or a `missing_critical` finding that can't be resolved |
| `identity_warnings` non-empty | Balance-sheet identity checker in `analysis/findings.py` |
| MongoDB upsert silently missing | `EARNINGS_SAVE_TARGET` env var; `mongodb_client.py` |
| Wrong metric value / scale | `__scale__` key handling post-merge in `extract_financial_metrics_node` |

### Iterate on the loop

- Can I make it faster? (Use a saved fixture instead of a live HTTP fetch.)
- Can I make the signal sharper? (Assert on the specific missing/wrong key, not just "status != failed".)
- Can I make it deterministic? (Freeze LLM responses with a mock; use golden fixtures in `tests/fixtures/golden/`.)

---

## Phase 2 — Reproduce

Run the loop. Watch the bug appear. Confirm:

- The failure mode matches what was reported, not a nearby symptom.
- It reproduces across multiple runs (or, for LLM non-determinism, at a high enough rate to debug against).
- You have captured the exact `error` message, wrong metric key/value, or wrong routing decision.

Do not proceed until you reproduce the bug.

---

## Phase 3 — Hypothesise

Generate 3–5 ranked hypotheses before testing any. Each must be falsifiable:

> "If `<cause>` is the root, then `<probe>` will make the bug disappear / appear more clearly."

Show the ranked list before testing. Proceed with your ranking if the user is AFK.

### Common hypothesis categories for this codebase

- **LLM output** — the model returned a malformed JSON, wrong scale, or hallucinated key name.
- **Chunk boundary** — a metric value was split across chunks; merging produced a duplicate or dropped value.
- **Routing logic** — `_route_after_analysis` or `_route_after_extraction` in `workflow.py` took an unexpected branch.
- **State mutation** — a node returned a partial state dict missing a required key.
- **Regex miss** — a Tier-1/Tier-2 pattern in `analysis/critical_metrics.py` doesn't match the company's phrasing.
- **MongoDB** — `_id` collision, wrong collection, or `STRICT_ACCURACY` blocking save.

---

## Phase 4 — Instrument

Each probe must map to a specific hypothesis. Change one variable at a time.

- **Prefer** targeted `logger.debug` at node entry/exit boundaries over logging everything.
- Tag every debug log with a unique prefix — e.g. `[DBG-a4f2]` — so cleanup is a single grep.
- **Never** add broad `print`/log dumps and grep through them.
- For LLM non-determinism: mock the LLM call and return a fixed response to isolate the downstream logic.

---

## Phase 5 — Fix + regression test

Write the regression test **before** the fix when a correct seam exists.

A correct seam for this codebase:
- `tests/test_<node>.py` — unit/integration test for the specific node or analysis function.
- Golden fixture test in `tests/test_golden_fixtures.py` — for end-to-end metric extraction correctness.
- Direct checker call — e.g. `check_presence(metrics.keys())` from `analysis/critical_metrics.py`.

Workflow:
1. Turn the minimised repro into a failing test.
2. Watch it fail (`uv run pytest -q tests/test_<module>.py`).
3. Apply the fix.
4. Watch it pass.
5. Re-run the full suite: `uv run pytest -q`.

If no correct seam exists, document it — the architecture is preventing the bug from being locked down.

---

## Phase 6 — Cleanup + post-mortem

Required before declaring done:

- [ ] Original repro no longer reproduces (re-run the Phase 1 loop).
- [ ] Regression test passes.
- [ ] All `[DBG-...]` instrumentation removed (`grep -r "DBG-" src/`).
- [ ] Throwaway scripts/fixtures deleted (or moved to `tests/fixtures/`).
- [ ] The winning hypothesis is stated in the commit message.

Then ask: what would have prevented this bug?
If the answer involves architectural change (no good test seam, tangled routing, hidden coupling), hand off to `/improve-codebase-architecture` with the specifics.
