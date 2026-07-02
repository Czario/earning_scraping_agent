"""Shared Redis queue helpers for the 8-K worker."""
from __future__ import annotations

import json
from typing import Any

from redis import Redis

from earnings_agents.config import REDIS_URL


def get_redis_client() -> Redis:
    """Return a Redis client connected to the configured REDIS_URL."""
    return Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=10,
    )


def serialize_message(payload: dict[str, Any]) -> str:
    """Serialize a payload dict to a JSON string for the Redis queue."""
    return json.dumps(payload)
