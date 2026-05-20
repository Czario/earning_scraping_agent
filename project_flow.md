

SEC flow is:

1. CLI input and company resolution  
- You run the command through earnings.py.  
- It resolves ticker/CIK using company_registry.py.  
- For source=sec, it calls EDGAR URL discovery in earnings.py.

2. EDGAR filing URL discovery (Exhibit 99.1)  
- SEC lookup happens in edgar_client.py.  
- Steps inside that function:  
1. Fetch SEC submissions JSON edgar_client.py  
2. Find latest 8-K with item 2.02 edgar_client.py  
3. Open filing index and locate EX-99.1 edgar_client.py  
4. Fallback to primary document if needed edgar_client.py

3. Graph starts with pre-resolved SEC URL  
- Initial state is created as status=discovered in earnings.py.  
- Graph entry point is discover_earnings_release in workflow.py.  
- Because discovered_file_url is already set, discovery node short-circuits immediately in discover_earnings_release.py.

4. Detect document type  
- Next node is detect_document_type in workflow.py.  
- Extension-first detection (.htm/.html/.pdf), then HEAD fallback detect_document_type.py.

5. Extract raw text  
- Router sends to HTML or PDF extractor workflow.py.  
- SEC filings are usually HTML, so this path runs: extract_html_text.py.  
- Important SEC-specific behavior:  
1. SEC-compliant User-Agent extract_html_text.py  
2. SGML wrapper stripping for EDGAR archives extract_html_text.py  
3. JS fallback disabled for sec.gov extract_html_text.py

6. Metric extraction (chunked LLM)  
- Workflow then runs extract_financial_metrics workflow.py.  
- Node logic in extract_financial_metrics.py:  
1. Chunk raw text extract_financial_metrics.py  
2. Ask Ollama per chunk extract_financial_metrics.py  
3. Parse JSON and apply __scale__ in Python extract_financial_metrics.py  
4. Merge arbitrary company-label metrics extract_financial_metrics.py

7. Save to MongoDB  
- Final node upserts document as TICKER_YEAR_latest in workflow.py.  
- Success status becomes saved workflow.py.

One important note:
- edgar_client.py contains duplicated definitions of normalize_cik and get_latest_earnings_url after the first set. Runtime still works because the later definitions override earlier ones, but it should be cleaned to avoid confusion.