---
name: improve-codebase-architecture
description: >
  Surface architectural friction in the earnings pipeline and propose deepening
  opportunities — refactors that turn shallow pass-through modules into deep,
  testable ones. Use when nodes feel tangled, analysis logic is hard to test in
  isolation, or routing helpers are accreting special cases.
---

# Improve Codebase Architecture

Surface architectural friction and propose deepening opportunities — refactors that turn shallow modules into deep ones. The aim is testability and AI-navigability of the earnings pipeline.

Before exploring, read `CONTEXT.md` for domain vocabulary and any ADRs in `docs/adr/` to avoid re-litigating settled decisions.

---

## Glossary

Use these terms exactly in every suggestion:

- **Module** — anything with an interface and an implementation: a node function, an analysis checker, a tool client, a config constant group.
- **Interface** — everything a caller must know: function signature, expected state keys in/out, error modes, side effects.
- **Implementation** — the code inside.
- **Depth** — leverage at the interface: a lot of behaviour behind a small interface. Deep = high leverage. Shallow = interface nearly as complex as implementation.
- **Seam** — where an interface lives; a place behaviour can be altered without editing in place.
- **Adapter** — a concrete thing satisfying an interface at a seam (e.g. `OllamaLLM` vs `GroqLLM` both satisfy the LLM seam in `llm_factory.py`).
- **Deletion test** — imagine deleting the module. If complexity vanishes, it was a pass-through. If it reappears across N callers, it was earning its keep.

---

## Phase 1 — Explore

Walk the codebase organically. Note where you experience friction:

- Where does understanding one concept require bouncing between many small files? (e.g. tracing `needs_reextract` through `workflow_state.py` → `analyze_metrics.py` → `workflow.py`)
- Where are modules shallow — the caller must know as much as the implementation?
- Where have pure functions been extracted just for testability, but the real bugs hide in how they're called?
- Where do nodes leak internal implementation details into `EarningsAgentState` fields?
- Which nodes are untested, or can only be tested through a full pipeline run?
- Which routing helpers in `workflow.py` are accreting special cases?

Apply the **deletion test** to anything suspect: would deleting it concentrate complexity, or just move it?

### Known friction areas to examine

| Area | Known friction |
|---|---|
| `nodes/extract_financial_metrics.py` | Chunking, merging, scale application, and LLM retry are all interleaved |
| `workflow.py` routing helpers | Each helper is a separate function; routing logic is distributed |
| `analysis/findings.py` checkers | Checkers are independent but all called from one monolithic `analyze_metrics_node` |
| `tools/edgar_client.py` | `normalize_cik` and `get_latest_earnings_url` are canonical; any duplication is a smell |
| `nodes/cleanup_metrics.py` | Two-pass (deterministic + LLM) with different rules; easy to confuse responsibilities |

---

## Phase 2 — Present candidates

Write a self-contained HTML report to `$TMPDIR/earnings-architecture-review-<timestamp>.html`.
Open it: `open <path>` (macOS).

For each candidate card include:

- **Files** — which files/modules are involved
- **Problem** — why the current architecture causes friction (use the glossary terms: shallow, pass-through, seam, etc.)
- **Solution** — plain English description of what would change
- **Benefits** — testability, locality, leverage; how tests would improve
- **Before / After diagram** — Mermaid where graph-shaped; hand-built HTML/CSS where editorial
- **Recommendation strength** — `Strong` | `Worth exploring` | `Speculative`

Use `CONTEXT.md` vocabulary for domain terms. Use the glossary above for architecture terms.

If a candidate contradicts an ADR in `docs/adr/`, mark it clearly and only surface it when the friction is real enough to warrant reopening the ADR.

End with a **Top Recommendation** section: which candidate to tackle first and why.

**Do NOT propose interfaces yet.** After the file is written, ask: "Which of these would you like to explore?"

---

## Phase 3 — Grilling loop

Once the user picks a candidate, drop into a grilling conversation. Walk the design tree with them: constraints, the shape of the deepened module, what sits behind the seam, what tests survive.

Side effects happen inline as decisions crystallise:

- New term not in `CONTEXT.md`? Add it — same discipline as `/grill-with-docs`.
- User rejects the candidate with a load-bearing reason? Offer an ADR (path: `docs/adr/NNNN-slug.md`), framed as: "Want me to record this so future architecture reviews don't re-suggest it?"

### Guardrails that must not be crossed during any refactor

- `EarningsAgentState` shape must be preserved; new fields use `NotRequired`.
- Status progression: `pending → discovered → fetched → text_extracted → extracted → saved | failed`.
- `needs_reextract: bool` is the sole routing signal from `analyze_metrics`; do not revert to overloading `status`.
- Metric keys are never renamed or normalised.
- `MAX_EXTRACTION_ATTEMPTS` remains the hard cap; no infinite retry paths.
- MongoDB `_id` format: `{TICKER}_{YEAR}_latest`.
