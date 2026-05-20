from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from earnings_agents.config import MONGODB_COLLECTION, MONGODB_DB, MONGODB_URI

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None


def get_collection() -> Collection:
    """Return the earnings MongoDB collection, reusing the module-level client."""
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client[MONGODB_DB][MONGODB_COLLECTION]


def upsert_earnings(doc: dict) -> None:
    """Insert or update an earnings document identified by its ``_id`` field."""
    if "_id" not in doc:
        raise ValueError("Earnings document must have an '_id' field")

    doc = {**doc, "scraped_at": datetime.now(timezone.utc)}
    collection = get_collection()
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": doc},
        upsert=True,
    )
    logger.info("Upserted earnings document: %s", doc["_id"])


def find_existing(ticker: str, fiscal_year: int, quarter: str) -> Optional[dict]:
    """Return an existing earnings document if present, else None."""
    doc_id = f"{ticker}_{fiscal_year}_{quarter}"
    return get_collection().find_one({"_id": doc_id})
