"""
Central configuration — single source of truth for the entire platform.

All modules MUST import configuration from here. No hard-coded dataset paths,
thresholds, or secrets anywhere else in the codebase.

Sub-configs use BaseModel (not BaseSettings) to avoid unintended env var pollution.
Only the master Settings class uses BaseSettings for env var overrides via FRAUD_ prefix.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ──────────────────────────────────────────────────────
# Project Root Detection
# ──────────────────────────────────────────────────────


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: assume two levels up from src/config/settings.py
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _find_project_root()


# ──────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class ModelStage(str, Enum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


# ──────────────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────────────


class DatasetConfig(BaseModel):
    """Immutable dataset configuration. DO NOT add fallback paths."""

    # THE canonical dataset path — the ONLY source of truth.
    path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "CFR_data.csv",
        description="Absolute path to the canonical dataset.",
    )

    expected_sha256: str = "89b2ca3e6791d124e1e739d965b62c95491ad80a67f5d1f77ce7cddd2dd25ad8"
    expected_row_count: int = 999_984
    expected_column_count: int = 54
    expected_fraud_count: int = 3_999
    expected_observed_label_count: int = 13_923
    expected_outlier_count: int = 20_000

    # Expected columns — canonical schema
    expected_columns: list[str] = [
        "transaction_id",
        "event_timestamp",
        "transaction_sequence_id",
        "event_version",
        "customer_id",
        "customer_tenure_days",
        "account_id",
        "account_type",
        "customer_country",
        "customer_risk_segment",
        "card_id",
        "card_type",
        "card_age_days",
        "credit_limit",
        "card_country",
        "card_status",
        "merchant_id",
        "merchant_category",
        "merchant_country",
        "merchant_region",
        "merchant_size",
        "merchant_age_days",
        "merchant_risk_segment",
        "device_id",
        "device_type",
        "device_os",
        "device_age_days",
        "device_country",
        "device_trust_level",
        "ip_id",
        "ip_country",
        "connection_type",
        "network_risk_segment",
        "proxy_type",
        "amount",
        "currency",
        "transaction_type",
        "payment_channel",
        "payment_method",
        "authentication_method",
        "installment_flag",
        "international_flag",
        "transaction_country",
        "billing_country",
        "terminal_id",
        "branch_id",
        "processing_route",
        "settlement_type",
        "batch_window",
        "is_outlier",
        "true_fraud_label",
        "observed_label",
        "label_timestamp",
        "fraud_scenario",
    ]

    # Timestamp range (ISO format)
    expected_timestamp_min: str = "2026-01-01"
    expected_timestamp_max: str = "2026-06-30"


# ──────────────────────────────────────────────────────
# Leakage Configuration
# ──────────────────────────────────────────────────────


class LeakageConfig(BaseModel):
    """Columns that MUST NOT be used as online predictive features."""

    # Post-event / target-related columns — forbidden in online inference
    forbidden_online_columns: list[str] = [
        "true_fraud_label",
        "fraud_scenario",
        "observed_label",
        "label_timestamp",
    ]

    # Columns that should NOT be used as model features (data-quality metadata only)
    excluded_feature_columns: list[str] = [
        "is_outlier",
    ]

    # High-cardinality ID columns — used for aggregation, not one-hot encoding
    id_columns: list[str] = [
        "transaction_id",
        "event_timestamp",
        "transaction_sequence_id",
        "event_version",
        "customer_id",
        "account_id",
        "card_id",
        "device_id",
        "ip_id",
        "merchant_id",
        "terminal_id",
        "branch_id",
    ]


# ──────────────────────────────────────────────────────
# Temporal Split Configuration
# ──────────────────────────────────────────────────────


class TemporalSplitConfig(BaseModel):
    """Chronological train/validation/test split boundaries."""

    train_end: str = "2026-04-30T23:59:59"
    validation_start: str = "2026-05-01T00:00:00"
    val_calib_end: str = "2026-05-15T23:59:59"
    val_opt_start: str = "2026-05-16T00:00:00"
    validation_end: str = "2026-05-31T23:59:59"
    test_start: str = "2026-06-01T00:00:00"


# ──────────────────────────────────────────────────────
# Risk Scoring Configuration
# ──────────────────────────────────────────────────────


class RiskScoringConfig(BaseModel):
    """Risk score mapping and thresholds."""

    # Risk score range
    min_score: int = 0
    max_score: int = 1000

    # Risk level boundaries
    low_max: int = 199
    medium_max: int = 499
    high_max: int = 749
    # 750-1000 = CRITICAL

    # Decision thresholds (on risk_score scale)
    approve_max: int = 299
    review_max: int = 699
    # >= 700 = BLOCK


# ──────────────────────────────────────────────────────
# Cost Function Configuration
# ──────────────────────────────────────────────────────


class CostConfig(BaseModel):
    """Cost parameters for the decision engine."""

    # Average cost when fraud is missed (not blocked)
    avg_fraud_loss: float = 500.0

    # Cost of a false positive (legitimate transaction blocked)
    false_positive_cost: float = 25.0

    # Cost of sending a transaction for manual review
    manual_review_cost: float = 5.0

    # Customer friction cost per unnecessary action
    customer_friction_cost: float = 2.0


# ──────────────────────────────────────────────────────
# Feature Store Configuration
# ──────────────────────────────────────────────────────


class FeatureStoreConfig(BaseModel):
    """Redis-based online feature store."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # TTLs in seconds
    ttl_5m: int = 300
    ttl_15m: int = 900
    ttl_1h: int = 3_600
    ttl_6h: int = 21_600
    ttl_24h: int = 86_400
    ttl_7d: int = 604_800


# ──────────────────────────────────────────────────────
# Kafka Configuration
# ──────────────────────────────────────────────────────


class KafkaConfig(BaseModel):
    """Kafka streaming configuration."""

    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "fraud-detection-consumer"

    # Topics
    topic_raw: str = "transactions.raw"
    topic_validated: str = "transactions.validated"
    topic_scored: str = "transactions.scored"
    topic_alerts: str = "fraud.alerts"
    topic_review: str = "fraud.review"
    topic_blocked: str = "fraud.blocked"
    topic_model_events: str = "model.events"
    topic_dlq: str = "dead_letter.transactions"


# ──────────────────────────────────────────────────────
# MLflow Configuration
# ──────────────────────────────────────────────────────


class MLflowConfig(BaseModel):
    """MLflow tracking and registry."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fraud-detection"
    registry_uri: str | None = None
    artifact_location: str | None = None


# ──────────────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────────────


class APIConfig(BaseModel):
    """FastAPI service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    api_version: str = "v1"
    title: str = "Fraud Detection API"

    # Auth
    api_key_header: str = "X-API-Key"
    jwt_secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60


# ──────────────────────────────────────────────────────
# Monitoring Configuration
# ──────────────────────────────────────────────────────


class MonitoringConfig(BaseModel):
    """Prometheus / Grafana monitoring."""

    prometheus_port: int = 9090
    enable_metrics: bool = True

    # Drift detection thresholds
    psi_warning: float = 0.1
    psi_critical: float = 0.2
    ks_warning: float = 0.05
    ks_critical: float = 0.1
    # JS divergence is bounded [0, ln(2)] ≈ 0.693 — needs own thresholds
    js_warning: float = 0.05
    js_critical: float = 0.1


# ──────────────────────────────────────────────────────
# Database Configuration
# ──────────────────────────────────────────────────────


class DatabaseConfig(BaseModel):
    """PostgreSQL metadata / audit store."""

    url: str = "sqlite:///fraud_platform.db"
    echo: bool = False
    pool_size: int = 5


# ──────────────────────────────────────────────────────
# Model Promotion Gates
# ──────────────────────────────────────────────────────


class PromotionGateConfig(BaseModel):
    """Minimum requirements for promoting a model to production."""

    min_pr_auc: float = 0.30
    min_recall: float = 0.70
    max_fpr: float = 0.10
    max_expected_cost: float = 100_000.0
    max_brier_score: float = 0.05
    require_calibration: bool = True
    require_no_drift: bool = True
    require_no_leakage: bool = True


# ──────────────────────────────────────────────────────
# Master Settings
# ──────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Master settings that composes all sub-configurations.

    Only this top-level class uses BaseSettings for env var overrides.
    Sub-configs use BaseModel to avoid env var pollution (e.g., PATH → path).
    Override sub-config values via FRAUD__<section>__<field> env vars.
    """

    model_config = {"env_prefix": "FRAUD_", "env_nested_delimiter": "__"}

    project_root: Path = PROJECT_ROOT
    environment: str = Field(default="development")

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    temporal_split: TemporalSplitConfig = Field(default_factory=TemporalSplitConfig)
    risk_scoring: RiskScoringConfig = Field(default_factory=RiskScoringConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    promotion_gates: PromotionGateConfig = Field(default_factory=PromotionGateConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
