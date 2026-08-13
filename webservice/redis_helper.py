import os
from functools import lru_cache

import redis


@lru_cache(maxsize=1)
def redis_client() -> redis.Redis:
    """Return the process-wide redis client."""
    return redis.Redis(
        host=os.environ["REDIS_HOST"], port=6379, db=0, decode_responses=True
    )
