"""Extraction subpackage: helpers for chunking, merging, and concept-mapping
financial text extracted from earnings releases.

This package contains the decomposed logic from the monolithic
``nodes/extract_financial_metrics.py``:

  chunker       — text chunking, section splitting, document pre-scanning
  merger        — per-chunk result merging and LLM response parsing
  concept_mapper — LLM-assisted semantic matching of extracted keys to XBRL concepts
"""
