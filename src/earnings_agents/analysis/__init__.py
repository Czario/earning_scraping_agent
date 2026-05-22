"""Pure-Python analysis helpers (no LLM, no I/O).

These modules consume an extracted metrics dict and produce structured
``Finding`` objects that downstream nodes (``analyze_metrics``, ``cleanup_metrics``)
act upon.
"""
