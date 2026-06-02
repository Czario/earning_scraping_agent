---
name: zoom-out
description: >
  Tell the agent to zoom out and give a higher-level perspective on an unfamiliar
  part of the earnings pipeline. Use when you're dropped into a node you haven't
  touched before, need to understand how a module fits into the LangGraph graph,
  or want to see all callers/consumers of a state field or analysis checker.
disable-model-invocation: true
---

# Zoom Out

I don't know this area of the codebase well. Go up a layer of abstraction.

Give me a map of all relevant modules and callers using the vocabulary from `CONTEXT.md` and the architecture described in `AGENTS.md`.

Specifically, tell me:

1. **Where does this fit in the graph?** Which nodes feed into it and which nodes consume its output? Reference the routing helpers in `workflow.py`.
2. **Which `EarningsAgentState` fields does it read and write?** List them with their types and what they mean in this context.
3. **What can go wrong here?** List the failure modes — what does `status = "failed"` look like from this node, and what `Finding` types could this area produce?
4. **Where are the tests?** Point me to the relevant test file(s) in `tests/` and any golden fixtures in `tests/fixtures/golden/`.
