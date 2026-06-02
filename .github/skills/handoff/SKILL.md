---
name: handoff
description: >
  Compact the current conversation into a handoff document so a fresh agent
  can continue the work. Use at the end of long sessions, when switching
  context, or before handing a task to another agent.
argument-hint: "What will the next session focus on? (e.g. 'debugging HD extraction', 'adding a new node', 'fixing Tier-1 miss for AAPL')"
---

# Handoff

Write a handoff document summarising the current conversation so a fresh agent can continue the work without re-reading the entire session.

Save to the OS temp directory (`$TMPDIR`, fallback `/tmp`) — **not** the workspace. File: `<tmpdir>/earnings-handoff-<YYYYMMDD-HHMM>.md`.

If the user passed an argument, treat it as a description of what the next session will focus on and tailor the document accordingly.

---

## Document structure

### 1. Objective
One paragraph: what was the goal of this session?

### 2. Current state
- **Ticker(s) worked on**: which company/companies
- **Pipeline status**: which nodes were touched, what `status` field the last run left
- **MongoDB document ID**: the `_id` of any upserted document (format: `{TICKER}_{YEAR}_latest`)
- **Open issues**: what is still broken or unresolved

### 3. Key decisions made
Bullet list of meaningful choices made during the session. Do not duplicate content already in ADRs in `docs/adr/` — reference them by path instead.

### 4. Files changed
List the files touched, with a one-line description of what changed. Do not duplicate content already in diffs or commits.

### 5. Commands to resume
Exact commands the next agent should run to reproduce the current state:
```bash
uv sync
uv run pytest -q tests/test_<module>.py   # verify current baseline
uv run earnings --ticker <TICKER> -v      # reproduce the last run
```

### 6. Findings / extraction state
If the session involved metric extraction debugging, include:
- Which `findings` (Finding types + severity) were present
- Whether `needs_reextract` was cycling
- Current `extraction_attempts` count vs `MAX_EXTRACTION_ATTEMPTS`
- Any `extraction_notes` content that was being injected

### 7. Suggested skills for next session
List the skills from `.github/skills/` the next agent should invoke:
- `/diagnose` — if a bug is still open
- `/grill-with-docs` — if domain terminology needs sharpening before coding
- `/zoom-out` — if the next agent is unfamiliar with the affected nodes
- `/improve-codebase-architecture` — if the session revealed structural friction
- `/tdd` — if new behavior needs test-first development

---

## Rules

- **Do not** reproduce full file contents — reference by path.
- **Do not** include API keys, passwords, MongoDB URIs, or `.env` values.
- **Do not** paste raw LLM outputs — summarise what they revealed.
- Reference `CONTEXT.md` for domain term definitions rather than re-explaining them inline.
- Keep the document under 300 lines. If it's longer, you're duplicating rather than summarising.
