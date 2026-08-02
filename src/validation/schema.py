"""
Pydantic schemas for training events and online transactions.

Two separate schemas enforce the leakage boundary:
  - TrainingEvent: contains the historical label (for training only)
  - OnlineTransaction: contains ONLY fields available at transaction decision time

The online model must NEVER receive:
  - true_fraud_label
  - fraud_scenario
  - observed_label
  - label_timestamp
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OnlineTransaction(BaseModel):
    """Schema for a transaction at decision time.

    This schema represents ONLY information available when a transaction
    arrives for real-time scoring. Post-event labels are NOT included.
    """

    transaction_id: str
    event_timestamp: str | datetime
    transaction_sequence_id: int
    event_version: str

    # Customer
    customer_id: str
    customer_tenure_days: int
    account_id: str
    account_type: str
    customer_country: str
    customer_risk_segment: str

    # Card
    card_id: str
    card_type: str
    card_age_days: int
    credit_limit: int
    card_country: str
    card_status: str

    # Merchant
    merchant_id: str
    merchant_category: str
    merchant_country: str
    merchant_region: str
    merchant_size: str
    merchant_age_days: int
    merchant_risk_segment: str

    # Device
    device_id: str
    device_type: str
    device_os: str
    device_age_days: int
    device_country: str
    device_trust_level: str

    # Network
    ip_id: str
    ip_country: str
    connection_type: str
    network_risk_segment: str
    proxy_type: str

    # Transaction details
    amount: float
    currency: str
    transaction_type: str
    payment_channel: str
    payment_method: str
    authentication_method: str
    installment_flag: int
    international_flag: int
    transaction_country: str
    billing_country: str

    # Routing
    terminal_id: str
    branch_id: str
    processing_route: str
    settlement_type: str
    batch_window: str

    # Data quality metadata (NOT a model feature)
    is_outlier: int = Field(default=0, description="Data quality flag, not a model feature")

    model_config = {"extra": "forbid"}  # Reject any extra fields


class TrainingEvent(OnlineTransaction):
    """Schema for a historical training event.

    Extends OnlineTransaction with the post-event labels that are
    available only for training and evaluation, never for online inference.
    """

    true_fraud_label: int
    observed_label: int
    label_timestamp: str | datetime
    fraud_scenario: str

    model_config = {"extra": "forbid"}


# ──────────────────────────────────────────────────────
# API Schemas
# ──────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """API request for single transaction scoring."""

    transaction_id: str
    event_timestamp: str | datetime
    transaction_sequence_id: int = 0
    event_version: str = "v2.1"

    customer_id: str
    customer_tenure_days: int = 0
    account_id: str
    account_type: str = "CREDIT"
    customer_country: str = "US"
    customer_risk_segment: str = "LOW"

    card_id: str
    card_type: str = "CREDIT"
    card_age_days: int = 0
    credit_limit: int = 5000
    card_country: str = "US"
    card_status: str = "ACTIVE"

    merchant_id: str
    merchant_category: str = "RETAIL"
    merchant_country: str = "US"
    merchant_region: str = "REG_US_01"
    merchant_size: str = "MEDIUM"
    merchant_age_days: int = 365
    merchant_risk_segment: str = "LOW"

    device_id: str
    device_type: str = "MOBILE_ANDROID"
    device_os: str = "Android_14"
    device_age_days: int = 100
    device_country: str = "US"
    device_trust_level: str = "MEDIUM"

    ip_id: str
    ip_country: str = "US"
    connection_type: str = "residential"
    network_risk_segment: str = "CLEAN"
    proxy_type: str = "none"

    amount: float
    currency: str = "USD"
    transaction_type: str = "purchase"
    payment_channel: str = "WEB"
    payment_method: str = "CARD_PRESENT"
    authentication_method: str = "PIN"
    installment_flag: int = 0
    international_flag: int = 0
    transaction_country: str = "US"
    billing_country: str = "US"

    terminal_id: str = "TERM_0001"
    branch_id: str = "BR_001"
    processing_route: str = "ROUTE_A"
    settlement_type: str = "INSTANT"
    batch_window: str = "BATCH_00"

    is_outlier: int = 0

    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    """API response for transaction scoring."""

    transaction_id: str
    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_score: int = Field(ge=0, le=1000)
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    decision: str  # APPROVE, REVIEW, BLOCK
    model_version: str
    feature_timestamp: str
    inference_latency_ms: float
    explanations: list[str] | None = None
    request_id: str | None = None


class BatchPredictionRequest(BaseModel):
    """API request for batch transaction scoring."""

    transactions: list[PredictionRequest] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """API response for batch transaction scoring."""

    predictions: list[PredictionResponse]
    total_latency_ms: float
    batch_size: int
