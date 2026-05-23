from __future__ import annotations

import atexit
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from earnings_agents.config import MONGODB_COLLECTION, MONGODB_DB, MONGODB_URI

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None


def _close_client() -> None:
    """Close the module-level Mongo client at interpreter shutdown."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001 — best-effort during shutdown
            pass
        _client = None


def get_collection() -> Collection:
    """Return the earnings MongoDB collection, reusing the module-level client."""
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
        atexit.register(_close_client)
    return _client[MONGODB_DB][MONGODB_COLLECTION]


_MONGO_MAX_RETRIES: int = 3
_MONGO_RETRY_BASE_DELAY: float = 0.5  # seconds; doubles on each retry


def upsert_earnings(doc: dict) -> None:
    """Insert or update an earnings document identified by its ``_id`` field.

    Retries up to ``_MONGO_MAX_RETRIES`` times on transient errors (network
    blip, primary step-down, etc.) with exponential back-off.
    """
    if "_id" not in doc:
        raise ValueError("Earnings document must have an '_id' field")

    doc = {**doc, "scraped_at": datetime.now(timezone.utc)}
    collection = get_collection()
    for attempt in range(_MONGO_MAX_RETRIES + 1):
        try:
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True,
            )
            logger.info("Upserted earnings document: %s", doc["_id"])
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == _MONGO_MAX_RETRIES:
                raise
            delay = _MONGO_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "MongoDB upsert failed for %s (attempt %d/%d): %s; retrying in %.1f s",
                doc["_id"], attempt + 1, _MONGO_MAX_RETRIES + 1, exc, delay,
            )
            time.sleep(delay)


def find_existing(ticker: str, fiscal_year: int, quarter: str) -> Optional[dict]:
    """Return an existing earnings document if present, else None."""
    doc_id = f"{ticker}_{fiscal_year}_{quarter}"
    return get_collection().find_one({"_id": doc_id})
