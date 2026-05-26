# Company-Specific Extraction Hints for AAPL (Apple Inc.)
#
# These hints are injected into every LLM extraction prompt for this ticker.
# Edit this file to fix recurring extraction errors. Human-curated only —
# the pipeline never writes to this file automatically.

- Apple reports dollar amounts in millions and share counts also in millions.
  Both are stated in a single "(In millions, except number of shares which are reflected in thousands)" header when applicable.
- "Net sales" is Apple's label for Revenue. Do NOT rename it — extract as "Net sales".
- EPS labels are "Basic earnings per share" and "Diluted earnings per share".
- Share count labels are "Weighted-average basic shares outstanding" and "Weighted-average diluted shares outstanding".
- The income statement period is typically "Three months ended" or "Six months ended" — extract from the most recent column only.
- Apple may present both quarterly and year-to-date columns; extract only the CURRENT QUARTER column unless the document is an annual filing.
