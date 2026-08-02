"""
Online feature store — Redis-based storage for real-time features.

Stores per-entity:
  - velocity counters (windowed)
  - rolling amount statistics
  - recent transaction timestamps
  - behavioral counters

Supports:
  get_features(entity_type, entity_id)
  update_features(entity_type, entity_id, transaction)
  get_transaction_context(transaction)

TTLs enforce time-windowed features (5m, 1h, 24h, 7d).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from src.config.settings import FeatureStoreConfig, get_settings

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """Redis-based online feature store for real-time fraud detection."""

    def __init__(self, config: Optional[FeatureStoreConfig] = None) -> None:
        self.config = config or get_settings().feature_store
        self._redis = None
        self._connected = False

    def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis

            self._redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            self._redis.ping()
            self._connected = True
            logger.info(
                "Connected to Redis at %s:%d", self.config.redis_host, self.config.redis_port
            )
        except Exception as e:
            logger.warning("Redis not available: %s. Using in-memory fallback.", e)
            self._redis = None
            self._connected = False
            self._memory_store: dict[str, Any] = {}

    def _key(self, entity_type: str, entity_id: str, feature: str) -> str:
        """Generate a Redis key."""
        return f"fs:{entity_type}:{entity_id}:{feature}"

    def get_features(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """Get all features for an entity."""
        prefix = f"fs:{entity_type}:{entity_id}:"

        if self._redis and self._connected:
            try:
                # Use SCAN instead of KEYS to avoid blocking Redis
                keys = []
                cursor = 0
                while True:
                    cursor, batch = self._redis.scan(cursor=cursor, match=f"{prefix}*", count=100)
                    keys.extend(batch)
                    if cursor == 0:
                        break
                if not keys:
                    return {}
                values = self._redis.mget(keys)
                return {
                    k.replace(prefix, ""): self._deserialize(v)
                    for k, v in zip(keys, values)
                    if v is not None
                }
            except Exception as e:
                logger.error("Redis get_features failed: %s", e)
                return {}
        else:
            return {
                k.replace(prefix, ""): v
                for k, v in self._memory_store.items()
                if k.startswith(prefix)
            }

    def update_features(
        self,
        entity_type: str,
        entity_id: str,
        features: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Update features for an entity."""
        if self._redis and self._connected:
            try:
                pipe = self._redis.pipeline()
                for feature_name, value in features.items():
                    key = self._key(entity_type, entity_id, feature_name)
                    pipe.set(key, self._serialize(value))
                    if ttl:
                        pipe.expire(key, ttl)
                pipe.execute()
            except Exception as e:
                logger.error("Redis update_features failed: %s", e)
        else:
            for feature_name, value in features.items():
                key = self._key(entity_type, entity_id, feature_name)
                self._memory_store[key] = value

    def increment_counter(
        self,
        entity_type: str,
        entity_id: str,
        counter_name: str,
        amount: float = 1.0,
        ttl: Optional[int] = None,
    ) -> float:
        """Increment a counter for an entity (e.g., txn_count_1h)."""
        key = self._key(entity_type, entity_id, counter_name)

        if self._redis and self._connected:
            try:
                new_val = self._redis.incrbyfloat(key, amount)
                if ttl:
                    self._redis.expire(key, ttl)
                return float(new_val)
            except Exception as e:
                logger.error("Redis increment failed: %s", e)
                return 0.0
        else:
            current = self._memory_store.get(key, 0.0)
            new_val = current + amount
            self._memory_store[key] = new_val
            return new_val

    def get_transaction_context(self, transaction: dict) -> dict[str, Any]:
        """Get all relevant features for a transaction from the store.

        Gathers features from all entity types referenced in the transaction.
        """
        context = {}

        entity_mapping = {
            "customer": "customer_id",
            "card": "card_id",
            "device": "device_id",
            "ip": "ip_id",
            "merchant": "merchant_id",
            "account": "account_id",
        }

        for entity_type, id_field in entity_mapping.items():
            entity_id = transaction.get(id_field)
            if entity_id:
                features = self.get_features(entity_type, entity_id)
                for feat_name, feat_val in features.items():
                    context[f"{entity_type}_{feat_name}"] = feat_val

        return context

    def update_from_transaction(self, transaction: dict) -> None:
        """Update feature store from a new transaction."""
        amount = transaction.get("amount", 0.0)

        entity_mapping = {
            "customer": "customer_id",
            "card": "card_id",
            "device": "device_id",
            "ip": "ip_id",
            "merchant": "merchant_id",
            "account": "account_id",
        }

        ttl_windows = {
            "5m": self.config.ttl_5m,
            "1h": self.config.ttl_1h,
            "24h": self.config.ttl_24h,
            "7d": self.config.ttl_7d,
        }

        for entity_type, id_field in entity_mapping.items():
            entity_id = transaction.get(id_field)
            if not entity_id:
                continue

            # Update counters for each time window
            for window, ttl in ttl_windows.items():
                self.increment_counter(entity_type, entity_id, f"txn_count_{window}", 1.0, ttl)
                self.increment_counter(entity_type, entity_id, f"amount_sum_{window}", amount, ttl)

            # Update latest timestamp
            self.update_features(
                entity_type,
                entity_id,
                {"last_txn_ts": transaction.get("event_timestamp", "")},
            )

    def health_check(self) -> bool:
        """Check if the feature store is healthy."""
        if self._redis and self._connected:
            try:
                return self._redis.ping()
            except Exception:
                return False
        return True  # In-memory is always "healthy"

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value

    def close(self) -> None:
        """Close the connection."""
        if self._redis:
            self._redis.close()
            logger.info("Redis connection closed")
