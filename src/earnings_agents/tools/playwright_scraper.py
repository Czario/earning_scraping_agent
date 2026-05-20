from __future__ import annotations

import logging

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from earnings_agents.config import HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def fetch_page_js(url: str) -> str:
    """Fetch a JavaScript-rendered page using Playwright headless Chromium.

    Returns the fully rendered HTML, or an empty string on failure.
    """
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": _USER_AGENT})
            page.goto(url, timeout=HTTP_TIMEOUT * 1_000)
            page.wait_for_load_state("networkidle", timeout=HTTP_TIMEOUT * 1_000)
            html = page.content()
            browser.close()
            return html
    except PlaywrightTimeout:
        logger.warning("Playwright timeout for %s", url)
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Playwright error for %s: %s", url, exc)
        return ""
