---
applyTo: "**"
---

# Agent Harness Engineering Principles

Use these principles for reliable multi-step agent behavior in this repository.

## 1) ReAct execution discipline
- Run tasks as a bounded cycle: reason -> act -> observe -> decide next step.
- Define explicit stop conditions before entering long loops.
- Fail fast on repeated no-progress iterations.

## 2) Context engineering first
- Start with short orientation: scope, constraints, expected output.
- Keep context minimal and relevant; prefer retrieval over context dumping.
- Re-state success criteria before finalizing high-impact work.

## 3) Memory and progress tracking
- Persist long-horizon progress to durable artifacts (notes, checklists, files).
- Update task state after meaningful actions so recovery is deterministic.
- Prefer filesystem-backed state over transient chat memory for continuity.

## 4) Tool orchestration and interoperability
- Use least-privilege tool access: expose only tools needed for the current step.
- Prefer specialized tools for file/search/edit over ad-hoc shell equivalents when available.
- Normalize tool results and error paths so retries and fallbacks remain predictable.

## 5) Constraints and safety management
- Enforce loop/time budgets for iterative workflows.
- Detect repeated tool calls with identical arguments and switch strategy.
- Require explicit human approval before destructive or irreversible operations.
- Gate completion on verification (tests/checks) proportionate to change impact.

## 6) Observability and continuous improvement
- Record decision intent per meaningful change: what was changed and expected effect.
- Validate outcomes against evidence (tests, diagnostics, run output) before declaring done.
- Treat harness/process edits as hypotheses: keep only changes that measurably improve outcomes.

## Practical default checklist
- Clarify task objective and boundaries.
- Plan bounded steps with a stop condition.
- Execute with ReAct and capture observations.
- Verify with focused checks first, then broader checks as needed.
- Summarize results with evidence and remaining risks.

## Source alignment
- Decoding AI: Agentic Harness Engineering (2026-03-31).
- LinkedIn article provided by user on harness-over-model emphasis.
- arXiv:2604.25850 (observability-driven automatic harness evolution).
