"""
Feature definitions registry.

Every production feature has a formal definition including:
  - name, description, data type, source
  - aggregation window, default value
  - transformation logic
  - online/offline implementation parity

This registry prevents training-serving skew by ensuring consistent
feature calculations across batch training and real-time inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class FeatureSource(str, Enum):
    """Where the raw data for a feature originates."""

    TRANSACTION = "transaction"  # Direct transaction field
    CUSTOMER = "customer"  # Customer-level aggregation
    CARD = "card"  # Card-level aggregation
    DEVICE = "device"  # Device-level aggregation
    IP = "ip"  # IP-level aggregation
    MERCHANT = "merchant"  # Merchant-level aggregation
    TERMINAL = "terminal"  # Terminal-level aggregation
    ACCOUNT = "account"  # Account-level aggregation
    DERIVED = "derived"  # Computed from other features
    GEOGRAPHIC = "geographic"  # Geographic comparison
    TEMPORAL = "temporal"  # Time-based extraction


class FeatureType(str, Enum):
    """Data type classification for features."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"


@dataclass
class FeatureDefinition:
    """Formal definition of a production feature."""

    name: str
    description: str
    dtype: FeatureType
    source: FeatureSource
    aggregation_window: Optional[str] = None  # e.g., "5m", "1h", "24h", "7d"
    default_value: Any = 0
    transformation: Optional[str] = None  # e.g., "log", "zscore", "minmax"
    online_available: bool = True  # Whether computable in real-time
    requires_history: bool = False  # Whether needs historical data
    category: str = ""  # Feature group: "static", "entity", "temporal", etc.


class FeatureRegistry:
    """Central registry of all feature definitions.

    Ensures consistency between offline training and online inference.
    """

    def __init__(self) -> None:
        self._features: dict[str, FeatureDefinition] = {}
        self._register_all_features()

    def _register_all_features(self) -> None:
        """Register all production features."""

        # ── A. Static Transaction Features ──
        static_features = [
            FeatureDefinition(
                "log_amount",
                "Natural log of transaction amount",
                FeatureType.NUMERIC,
                FeatureSource.TRANSACTION,
                transformation="log1p",
                category="amount",
            ),
            FeatureDefinition(
                "amount_credit_ratio",
                "Amount / credit_limit",
                FeatureType.NUMERIC,
                FeatureSource.DERIVED,
                category="amount",
            ),
            FeatureDefinition(
                "is_international",
                "International transaction flag",
                FeatureType.BINARY,
                FeatureSource.TRANSACTION,
                category="static",
            ),
            FeatureDefinition(
                "is_installment",
                "Installment payment flag",
                FeatureType.BINARY,
                FeatureSource.TRANSACTION,
                category="static",
            ),
            FeatureDefinition(
                "amount_bucket",
                "Discretized amount range",
                FeatureType.ORDINAL,
                FeatureSource.DERIVED,
                category="amount",
            ),
            FeatureDefinition(
                "is_micro_txn",
                "Amount < 1.0",
                FeatureType.BINARY,
                FeatureSource.DERIVED,
                category="amount",
            ),
            FeatureDefinition(
                "is_high_value_txn",
                "Amount > 5000",
                FeatureType.BINARY,
                FeatureSource.DERIVED,
                category="amount",
            ),
        ]

        # ── Categorical encoding features ──
        categorical_features = [
            FeatureDefinition(
                f"{col}_encoded",
                f"Frequency-encoded {col}",
                FeatureType.NUMERIC,
                FeatureSource.TRANSACTION,
                category="categorical",
            )
            for col in [
                "account_type",
                "card_type",
                "merchant_category",
                "merchant_size",
                "transaction_type",
                "payment_channel",
                "payment_method",
                "authentication_method",
                "connection_type",
                "proxy_type",
                "device_type",
                "device_os",
                "card_status",
                "device_trust_level",
                "currency",
                "settlement_type",
                "processing_route",
                "customer_risk_segment",
                "merchant_risk_segment",
                "network_risk_segment",
            ]
        ]

        # ── B. Entity Features ──
        entity_windows = ["5m", "15m", "1h", "6h", "24h", "7d"]
        entity_types = ["customer", "card", "device", "ip", "merchant", "account"]

        entity_features = []
        for entity in entity_types:
            for window in entity_windows:
                entity_features.extend(
                    [
                        FeatureDefinition(
                            f"{entity}_txn_count_{window}",
                            f"Transaction count for {entity} in last {window}",
                            FeatureType.NUMERIC,
                            FeatureSource[entity.upper()],
                            aggregation_window=window,
                            requires_history=True,
                            category="velocity",
                        ),
                        FeatureDefinition(
                            f"{entity}_amount_sum_{window}",
                            f"Total amount for {entity} in last {window}",
                            FeatureType.NUMERIC,
                            FeatureSource[entity.upper()],
                            aggregation_window=window,
                            requires_history=True,
                            category="velocity",
                        ),
                    ]
                )
            # Non-windowed entity features
            entity_features.extend(
                [
                    FeatureDefinition(
                        f"{entity}_avg_amount",
                        f"Historical average amount for {entity}",
                        FeatureType.NUMERIC,
                        FeatureSource[entity.upper()],
                        requires_history=True,
                        category="entity",
                    ),
                    FeatureDefinition(
                        f"{entity}_std_amount",
                        f"Historical amount std deviation for {entity}",
                        FeatureType.NUMERIC,
                        FeatureSource[entity.upper()],
                        requires_history=True,
                        category="entity",
                    ),
                    FeatureDefinition(
                        f"{entity}_total_txn_count",
                        f"Total historical transaction count for {entity}",
                        FeatureType.NUMERIC,
                        FeatureSource[entity.upper()],
                        requires_history=True,
                        category="entity",
                    ),
                ]
            )

        # ── C. Temporal Features ──
        temporal_features = [
            FeatureDefinition(
                "hour_of_day",
                "Hour (0-23)",
                FeatureType.NUMERIC,
                FeatureSource.TEMPORAL,
                category="temporal",
            ),
            FeatureDefinition(
                "minute_bucket",
                "15-min bucket (0-95)",
                FeatureType.NUMERIC,
                FeatureSource.TEMPORAL,
                category="temporal",
            ),
            FeatureDefinition(
                "day_of_week",
                "Day of week (0=Mon)",
                FeatureType.NUMERIC,
                FeatureSource.TEMPORAL,
                category="temporal",
            ),
            FeatureDefinition(
                "day_of_month",
                "Day of month (1-31)",
                FeatureType.NUMERIC,
                FeatureSource.TEMPORAL,
                category="temporal",
            ),
            FeatureDefinition(
                "is_weekend",
                "Weekend flag",
                FeatureType.BINARY,
                FeatureSource.TEMPORAL,
                category="temporal",
            ),
            FeatureDefinition(
                "is_night",
                "Night hour (22-6) flag",
                FeatureType.BINARY,
                FeatureSource.TEMPORAL,
                category="temporal",
            ),
        ]
        # Time since previous transaction per entity
        for entity in ["customer", "card", "device", "ip"]:
            temporal_features.append(
                FeatureDefinition(
                    f"time_since_prev_{entity}_txn_seconds",
                    f"Seconds since previous {entity} transaction",
                    FeatureType.NUMERIC,
                    FeatureSource.TEMPORAL,
                    requires_history=True,
                    category="temporal",
                    default_value=-1,
                )
            )

        # ── E. Geographic Features ──
        geo_pairs = [
            ("customer_country", "transaction_country"),
            ("billing_country", "transaction_country"),
            ("card_country", "transaction_country"),
            ("device_country", "transaction_country"),
            ("ip_country", "transaction_country"),
            ("merchant_country", "transaction_country"),
        ]
        geographic_features = [
            FeatureDefinition(
                f"{a.split('_')[0]}_txn_country_mismatch",
                f"Mismatch between {a} and {b}",
                FeatureType.BINARY,
                FeatureSource.GEOGRAPHIC,
                category="geographic",
            )
            for a, b in geo_pairs
        ]
        geographic_features.extend(
            [
                FeatureDefinition(
                    "cross_border_flag",
                    "Transaction crosses borders",
                    FeatureType.BINARY,
                    FeatureSource.GEOGRAPHIC,
                    category="geographic",
                ),
                FeatureDefinition(
                    "geo_mismatch_count",
                    "Count of country mismatches",
                    FeatureType.NUMERIC,
                    FeatureSource.GEOGRAPHIC,
                    category="geographic",
                ),
            ]
        )

        # ── F. Identity/Device Risk Features ──
        identity_features = [
            FeatureDefinition(
                "is_new_device",
                "Device age < 7 days",
                FeatureType.BINARY,
                FeatureSource.DERIVED,
                category="identity",
            ),
            FeatureDefinition(
                "device_age_bucket",
                "Discretized device age",
                FeatureType.ORDINAL,
                FeatureSource.DERIVED,
                category="identity",
            ),
            FeatureDefinition(
                "is_proxy",
                "Using any proxy",
                FeatureType.BINARY,
                FeatureSource.DERIVED,
                category="identity",
            ),
            FeatureDefinition(
                "is_tor",
                "TOR exit node connection",
                FeatureType.BINARY,
                FeatureSource.DERIVED,
                category="identity",
            ),
            FeatureDefinition(
                "is_datacenter",
                "Datacenter connection",
                FeatureType.BINARY,
                FeatureSource.DERIVED,
                category="identity",
            ),
            FeatureDefinition(
                "proxy_risk_score",
                "Proxy risk (none=0, proxy=1, tor=2)",
                FeatureType.ORDINAL,
                FeatureSource.DERIVED,
                category="identity",
            ),
            FeatureDefinition(
                "customer_tenure_bucket",
                "Discretized customer tenure",
                FeatureType.ORDINAL,
                FeatureSource.DERIVED,
                category="identity",
            ),
            FeatureDefinition(
                "card_age_bucket",
                "Discretized card age",
                FeatureType.ORDINAL,
                FeatureSource.DERIVED,
                category="identity",
            ),
        ]

        # ── G. Amount Features ──
        amount_features = [
            FeatureDefinition(
                "amount_zscore_customer",
                "Amount z-score vs customer mean",
                FeatureType.NUMERIC,
                FeatureSource.DERIVED,
                requires_history=True,
                category="amount",
            ),
            FeatureDefinition(
                "amount_zscore_card",
                "Amount z-score vs card mean",
                FeatureType.NUMERIC,
                FeatureSource.DERIVED,
                requires_history=True,
                category="amount",
            ),
        ]

        # Register all
        all_features = (
            static_features
            + categorical_features
            + entity_features
            + temporal_features
            + geographic_features
            + identity_features
            + amount_features
        )
        for feat in all_features:
            self._features[feat.name] = feat

    def get(self, name: str) -> FeatureDefinition:
        """Get a feature definition by name."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        return self._features[name]

    def list_features(self, category: Optional[str] = None) -> list[FeatureDefinition]:
        """List all features, optionally filtered by category."""
        if category:
            return [f for f in self._features.values() if f.category == category]
        return list(self._features.values())

    def feature_names(self, category: Optional[str] = None) -> list[str]:
        """List feature names, optionally filtered by category."""
        return [f.name for f in self.list_features(category)]

    @property
    def count(self) -> int:
        return len(self._features)


# Singleton registry
FEATURE_REGISTRY = FeatureRegistry()
