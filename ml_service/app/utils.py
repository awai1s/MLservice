import time
from functools import wraps
import redis
import json
from .config import settings

r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

def timeboxed(ms_budget):
    def deco(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = (time.time() - start) * 1000
                # (Optional) log elapsed
        return inner
    return deco

def cacheable(prefix: str, ttl: int = settings.CACHE_TTL_S):
    def deco(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key = f"{prefix}:{json.dumps([args, kwargs], sort_keys=True)}"
            cached = r.get(key)
            if cached:
                return json.loads(cached)
            out = fn(*args, **kwargs)
            r.setex(key, ttl, json.dumps(out))
            return out
        return inner
    return deco
