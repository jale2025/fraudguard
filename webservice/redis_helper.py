"""
Shared redis client for the prediction service.

``api_to_database`` bumps the queue counters and ``drift_report`` reads them. Both
used to build their own ``redis.Redis`` per call -- ``api_to_database`` even once per
batch. The client is thread-safe and holds a connection pool, so one per process is
both cheaper and enough.
"""

import os
from functools import lru_cache

import redis


@lru_cache(maxsize=1)
def redis_client() -> redis.Redis:
    """Return the process-wide redis client."""
    return redis.Redis(
        host=os.environ["REDIS_HOST"], port=6379, db=0, decode_responses=True
    )
