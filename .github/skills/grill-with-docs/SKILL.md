---
name: grill-with-docs
description: >
  Grilling session that stress-tests a plan against this project's domain model,
  sharpens terminology, and updates CONTEXT.md and docs/adr/ inline as decisions
  crystallise. Use before adding a new node, changing routing logic, modifying
  the analysis loop, or introducing a new metric registry entry.
---

# Grill With Docs

Interview me relentlessly about every aspect of this plan until we reach a shared understanding.
Walk down each branch of the design tree, resolving dependencies between decisions one by one.
For each question, provide your recommended answer.

Ask questions **one at a time**, waiting for feedback before continuing.

If a question can be answered by exploring the codebase, explore it instead of asking.

---

## Domain awareness

Before starting the interview, read and internalise:

1. `CONTEXT.md` — the canonical domain glossary for this project.
2. `AGENTS.md` — architecture overview, node map, state machine, and key conventions.
3. `.github/copilot-instructions.md` — coding conventions and guardrails.
4. Any ADRs in `docs/adr/` that touch the area being discussed.

The state machine, node boundaries, and routing helpers in `workflow.py` are the backbone of this pipeline. Any plan that touches them deserves extra scrutiny.

---

## During the session

### Challenge against the glossary

When the user uses a term that conflicts with `CONTEXT.md`, call it out immediately.

> "Your glossary defines 'degraded' as a MongoDB document status meaning unresolved Tier-1 findings after all attempts. But you seem to be using it to mean something different — which is it?"

### Sharpen fuzzy language

When the user uses vague or overloaded terms, propose a precise canonical term.

> "You're saying 'retry' — do you mean incrementing `extraction_attempts` and looping back through `extract_financial_metrics`, or adding a per-chunk HTTP retry? Those are different mechanisms."

### Discuss concrete scenarios

Stress-test domain relationships with specific examples. For extraction changes:
- "What does `EarningsAgentState` look like at the start of this node?"
- "What does it look like when it exits successfully? When it fails?"
- "Which `Finding` types could this new logic emit, and what severity?"

For routing changes:
- "Walk me through the state transitions for the HD ticker with a missing Tier-1 metric. Which routing helper fires, and what does it return?"

### Cross-reference with code

When the user states how something works, check whether the code agrees. If you find a contradiction, surface it.

> "You said `cleanup_metrics` can rename keys. But `AGENTS.md` says 'never renames, normalizes, or edits the text of retained metric keys'. Which is authoritative?"

### Watch for guardrail violations

Flag immediately if a proposed change would violate a known guardrail:

- Status progression: `pending → discovered → fetched → text_extracted → extracted → saved | failed`
- `needs_reextract` is the only routing signal from `analyze_metrics` — do not overload `status`.
- Metric keys must be preserved exactly as found in source documents.
- MongoDB `_id` format must remain `{TICKER}_{YEAR}_latest`.
- `MAX_EXTRACTION_ATTEMPTS` is the hard cap — no infinite retry paths.
- `nodes/reflect_metrics.py` has been deleted and must not be recreated.

### Update CONTEXT.md inline

When a term is resolved during the session, update `CONTEXT.md` right there. Do not batch these up.

`CONTEXT.md` is a **glossary only** — no implementation details, no specs, no scratch-pad content.

### Offer ADRs sparingly

Only offer to create an ADR when **all three** are true:
1. Hard to reverse — changing your mind later has meaningful cost.
2. Surprising without context — a future reader would wonder "why did they do it this way?"
3. The result of a real trade-off — genuine alternatives existed.

ADRs live in `docs/adr/` with sequential numbering: `0001-slug.md`, `0002-slug.md`. Create the directory lazily when the first ADR is needed.

---

## File structure

```
/
├── CONTEXT.md          ← domain glossary (single context repo)
├── docs/
│   └── adr/
│       ├── 0001-*.md
│       └── 0002-*.md
├── AGENTS.md           ← architecture reference (read-only during session)
└── src/earnings_agents/
```
