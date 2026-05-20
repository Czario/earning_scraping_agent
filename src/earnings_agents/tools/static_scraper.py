from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from earnings_agents.config import HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_page(url: str) -> tuple[str, bool]:
    """Fetch a URL with plain HTTP.

    Returns:
        (html_text, success): html_text is empty string on failure.
    """
    try:
        response = requests.get(url, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.text, True
    except requests.RequestException as exc:
        logger.warning("Static fetch failed for %s: %s", url, exc)
        return "", False


def extract_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Extract all anchor links with their visible text from an HTML page.

    Resolves relative URLs against *base_url*.
    """
    parsed_base = urlparse(base_url)
    soup = BeautifulSoup(html, "lxml")
    links: list[dict[str, str]] = []

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"].strip()
        text: str = tag.get_text(strip=True)

        if not href or href.startswith(("#", "javascript:")):
            continue

        if href.startswith("//"):
            href = f"{parsed_base.scheme}:{href}"
        elif href.startswith("/"):
            href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
        elif not href.startswith("http"):
            href = base_url.rstrip("/") + "/" + href

        links.append({"url": href, "text": text})

    return links
