from __future__ import annotations

import asyncio
import inspect
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

import redis.asyncio as redis

from app.config.env import getenv

T = TypeVar("T")

_redis_client: Optional[redis.Redis] = None
_memory_cache: dict[str, tuple[float, bytes]] = {}


def _serialize(value: Any) -> bytes:
    return pickle.dumps(value)


def _deserialize(payload: bytes) -> Any:
    return pickle.loads(payload)


def _mark_stale(value: Any) -> Any:
    if hasattr(value, "attrs"):
        try:
            value.attrs["stale"] = True
            return value
        except Exception:
            pass
    if isinstance(value, dict):
        value["stale"] = True
        return value
    if is_dataclass(value):
        try:
            setattr(value, "stale", True)
        except Exception:
            return {"data": asdict(value), "stale": True}
        return value
    return {"data": value, "stale": True}


async def _maybe_await(value_or_fn: Any) -> Any:
    if inspect.isawaitable(value_or_fn):
        return await value_or_fn
    if callable(value_or_fn):
        result = value_or_fn()
        if inspect.isawaitable(result):
            return await result
        return result
    return value_or_fn


async def get_redis_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    redis_url = getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_client = redis.from_url(redis_url, decode_responses=False)
        await _redis_client.ping()
        return _redis_client
    except Exception:
        return None


async def get_or_set(key: str, fetch_fn: Callable[[], Awaitable[T] | T], ttl: int) -> T:
    client = await get_redis_client()
    if client is not None:
        try:
            cached = await client.get(key)
            if cached is not None:
                return _deserialize(cached)
        except Exception:
            pass

    memory_hit = _memory_cache.get(key)
    if memory_hit is not None:
        expires_at, payload = memory_hit
        if expires_at > asyncio.get_event_loop().time():
            return _deserialize(payload)

    try:
        value = await _maybe_await(fetch_fn)
        payload = _serialize(value)
        if client is not None:
            try:
                await client.set(key, payload, ex=ttl)
            except Exception:
                pass
        _memory_cache[key] = (asyncio.get_event_loop().time() + ttl, payload)
        return value
    except Exception:
        if client is not None:
            try:
                cached = await client.get(key)
                if cached is not None:
                    return _mark_stale(_deserialize(cached))
            except Exception:
                pass
        if memory_hit is not None:
            _, payload = memory_hit
            return _mark_stale(_deserialize(payload))
        raise
