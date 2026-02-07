"""
Redis caching layer for API responses.

Provides caching for expensive operations like recommendations and synergy scores.
Falls back gracefully if Redis is unavailable.

Configuration:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    CACHE_TTL: Default cache TTL in seconds (default: 3600)

Usage:
    from src.api.cache import cache, invalidate_commander_cache

    # Cache a function result
    @cache.cached(prefix="recommendations", ttl=3600)
    def get_recommendations(commander: str) -> dict:
        ...

    # Invalidate cache after data sync
    invalidate_commander_cache("Atraxa, Praetors' Voice")
"""

import json
import os
import hashlib
import functools
import logging
from typing import Optional, Any, Callable
from datetime import timedelta

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("redis package not installed, caching disabled")


class RedisCache:
    """
    Redis-based cache with automatic serialization.

    Falls back gracefully if Redis is unavailable.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        default_ttl: int = 3600,
        prefix: str = "manasink",
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL
            default_ttl: Default TTL in seconds (1 hour)
            prefix: Key prefix for all cache entries
        """
        self.url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._client: Optional["redis.Redis"] = None
        self._connected = False

    def _get_client(self) -> Optional["redis.Redis"]:
        """Get or create Redis client."""
        if not HAS_REDIS:
            return None

        if self._client is None:
            try:
                self._client = redis.from_url(
                    self.url,
                    decode_responses=True,
                    socket_timeout=2,
                    socket_connect_timeout=2,
                )
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info(f"Connected to Redis at {self.url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._client = None
                self._connected = False

        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._get_client() is not None

    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Create a cache key from prefix and arguments."""
        # Create a deterministic hash of the arguments
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"{self.prefix}:{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        client = self._get_client()
        if not client:
            return None

        try:
            value = client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in cache."""
        client = self._get_client()
        if not client:
            return False

        try:
            serialized = json.dumps(value)
            client.setex(key, ttl or self.default_ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        client = self._get_client()
        if not client:
            return False

        try:
            client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        client = self._get_client()
        if not client:
            return 0

        try:
            full_pattern = f"{self.prefix}:{pattern}"
            keys = client.keys(full_pattern)
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache delete pattern error: {e}")
            return 0

    def cached(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None,
    ):
        """
        Decorator to cache function results.

        Args:
            prefix: Cache key prefix
            ttl: TTL in seconds (uses default if not specified)
            key_func: Custom function to generate cache key from args

        Example:
            @cache.cached(prefix="recommendations", ttl=3600)
            def get_recommendations(commander: str) -> list:
                ...
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = f"{self.prefix}:{prefix}:{key_func(*args, **kwargs)}"
                else:
                    cache_key = self._make_key(prefix, *args, **kwargs)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_value

                # Call function and cache result
                result = func(*args, **kwargs)

                if result is not None:
                    self.set(cache_key, result, ttl)
                    logger.debug(f"Cache set: {cache_key}")

                return result

            # Add method to invalidate this specific cache
            wrapper.invalidate = lambda *args, **kwargs: self.delete(
                self._make_key(prefix, *args, **kwargs)
                if not key_func
                else f"{self.prefix}:{prefix}:{key_func(*args, **kwargs)}"
            )

            return wrapper

        return decorator

    def clear_all(self) -> int:
        """Clear all cache entries with our prefix."""
        return self.delete_pattern("*")


# Global cache instance
cache = RedisCache()


# =============================================================================
# Convenience Functions
# =============================================================================


def invalidate_commander_cache(commander: str) -> int:
    """
    Invalidate all cache entries for a commander.

    Call this after syncing EDHREC data.
    """
    # Normalize commander name for pattern matching
    pattern = f"*{commander.lower().replace(' ', '*').replace(',', '')}*"
    deleted = cache.delete_pattern(pattern)
    logger.info(f"Invalidated {deleted} cache entries for {commander}")
    return deleted


def invalidate_all_cache() -> int:
    """
    Invalidate all cache entries.

    Call this after a full data sync.
    """
    deleted = cache.clear_all()
    logger.info(f"Invalidated all {deleted} cache entries")
    return deleted


def get_cache_stats() -> dict:
    """Get cache statistics."""
    client = cache._get_client()
    if not client:
        return {
            "connected": False,
            "error": "Redis not available",
        }

    try:
        info = client.info("stats")
        memory = client.info("memory")
        keyspace = client.info("keyspace")

        # Count our keys
        our_keys = len(client.keys(f"{cache.prefix}:*"))

        return {
            "connected": True,
            "url": cache.url,
            "total_keys": our_keys,
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "memory_used_mb": memory.get("used_memory", 0) / 1024 / 1024,
            "evicted_keys": info.get("evicted_keys", 0),
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
        }
