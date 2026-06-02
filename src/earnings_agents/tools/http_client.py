"""Thin HTTP client used by nodes that fetch raw documents.

Centralises headers, timeout handling, and error normalisation so nodes do not
import ``requests`` directly.  Two header presets are provided:

- ``SEC_HEADERS``  — programmatic User-Agent required by SEC EDGAR.
- ``BROWSER_HEADERS`` — generic browser User-Agent for non-SEC pages.

Public functions
----------------
head(url) -> requests.Response
    Issue a HEAD request and return the response.
get(url, *, sec=False) -> requests.Response
    Issue a GET request, automatically choosing the correct header preset.
"""
from __future__ import annotations

import requests

from earnings_agents.config import HTTP_TIMEOUT

# SEC EDGAR requires a descriptive, non-browser User-Agent with contact info.
SEC_HEADERS: dict[str, str] = {
    "User-Agent": "earning-agents data-pipeline@truegrids.com",
    "Accept-Encoding": "gzip, deflate",
}

# Generic browser User-Agent for non-SEC pages.
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def head(url: str) -> requests.Response:
    """Send a HEAD request with browser headers and return the response.

    Raises ``requests.RequestException`` on network errors.
    """
    return requests.head(
        url,
        headers=BROWSER_HEADERS,
        timeout=HTTP_TIMEOUT,
        allow_redirects=True,
    )


def get(url: str, *, sec: bool = False) -> requests.Response:
    """Send a GET request and return the response.

    Parameters
    ----------
    url:
        Target URL.
    sec:
        When ``True``, send SEC-compliant headers instead of browser headers.

    Raises ``requests.RequestException`` on network errors.
    """
    headers = SEC_HEADERS if sec else BROWSER_HEADERS
    return requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
