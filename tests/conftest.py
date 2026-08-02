"""Shared test fixtures for the fraud detection test suite.

These fixtures provide synthetic data that does NOT require CFR_data.csv.
All unit tests should use these fixtures to avoid dataset dependency.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    """Create a small synthetic transactions DataFrame for testing.

    Contains 100 transactions spanning 3 months with ~5% fraud rate.
    All columns match the canonical schema.
    """
    rng = np.random.RandomState(42)
    n = 100

    base_date = datetime(2026, 1, 1)
    timestamps = [base_date + timedelta(hours=i * 12) for i in range(n)]

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_TEST_{i:04d}" for i in range(n)],
            "event_timestamp": timestamps,
            "transaction_sequence_id": list(range(n)),
            "event_version": ["v2.1"] * n,
            "customer_id": [f"CUST_{rng.randint(0, 10):03d}" for _ in range(n)],
            "customer_tenure_days": rng.randint(10, 1000, n),
            "account_id": [f"ACC_{rng.randint(0, 10):03d}" for _ in range(n)],
            "account_type": rng.choice(["CREDIT", "DEBIT", "SAVINGS"], n),
            "customer_country": rng.choice(
                ["US", "UK", "DE", "FR", "NG"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]
            ),
            "customer_risk_segment": rng.choice(["LOW", "MEDIUM", "HIGH"], n, p=[0.6, 0.3, 0.1]),
            "card_id": [f"CARD_{rng.randint(0, 15):03d}" for _ in range(n)],
            "card_type": rng.choice(["CREDIT", "DEBIT"], n),
            "card_age_days": rng.randint(1, 800, n),
            "credit_limit": rng.choice([1000, 2000, 5000, 10000, 25000], n).astype(float),
            "card_country": rng.choice(
                ["US", "UK", "DE", "FR", "NG"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]
            ),
            "card_status": rng.choice(["ACTIVE", "INACTIVE", "SUSPENDED"], n, p=[0.9, 0.05, 0.05]),
            "merchant_id": [f"MERCH_{rng.randint(0, 20):03d}" for _ in range(n)],
            "merchant_category": rng.choice(
                ["RETAIL", "ELECTRONICS", "FOOD", "TRAVEL", "GAMBLING"], n
            ),
            "merchant_country": rng.choice(
                ["US", "UK", "DE", "FR", "NG"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]
            ),
            "merchant_region": [f"REG_{rng.randint(0, 5):02d}" for _ in range(n)],
            "merchant_size": rng.choice(["SMALL", "MEDIUM", "LARGE"], n),
            "merchant_age_days": rng.randint(30, 2000, n),
            "merchant_risk_segment": rng.choice(["LOW", "MEDIUM", "HIGH"], n, p=[0.6, 0.3, 0.1]),
            "device_id": [f"DEV_{rng.randint(0, 12):03d}" for _ in range(n)],
            "device_type": rng.choice(["MOBILE_ANDROID", "MOBILE_IOS", "DESKTOP_WINDOWS"], n),
            "device_os": rng.choice(["Android_14", "iOS_17", "Windows_11"], n),
            "device_age_days": rng.randint(1, 500, n),
            "device_country": rng.choice(
                ["US", "UK", "DE", "FR", "NG"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]
            ),
            "device_trust_level": rng.choice(["LOW", "MEDIUM", "HIGH"], n),
            "ip_id": [f"IP_{rng.randint(0, 15):03d}" for _ in range(n)],
            "ip_country": rng.choice(
                ["US", "UK", "DE", "FR", "NG"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]
            ),
            "connection_type": rng.choice(["residential", "corporate", "datacenter", "mobile"], n),
            "network_risk_segment": rng.choice(
                ["CLEAN", "SUSPICIOUS", "BLOCKED"], n, p=[0.8, 0.15, 0.05]
            ),
            "proxy_type": rng.choice(
                ["none", "unknown", "proxy", "tor"], n, p=[0.85, 0.05, 0.05, 0.05]
            ),
            "amount": np.abs(rng.lognormal(4, 1.5, n)).round(2),
            "currency": rng.choice(["USD", "EUR", "GBP"], n, p=[0.6, 0.25, 0.15]),
            "transaction_type": rng.choice(["purchase", "transfer", "withdrawal"], n),
            "payment_channel": rng.choice(["WEB", "MOBILE", "POS"], n),
            "payment_method": rng.choice(["CARD_PRESENT", "CARD_NOT_PRESENT"], n),
            "authentication_method": rng.choice(["PIN", "3DS", "BIOMETRIC", "NONE"], n),
            "installment_flag": rng.choice([0, 1], n, p=[0.9, 0.1]),
            "international_flag": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "transaction_country": rng.choice(
                ["US", "UK", "DE", "FR", "NG"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]
            ),
            "billing_country": rng.choice(
                ["US", "UK", "DE", "FR", "NG"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1]
            ),
            "terminal_id": [f"TERM_{rng.randint(0, 8):03d}" for _ in range(n)],
            "branch_id": [f"BR_{rng.randint(0, 5):03d}" for _ in range(n)],
            "processing_route": rng.choice(["ROUTE_A", "ROUTE_B"], n),
            "settlement_type": rng.choice(["INSTANT", "DEFERRED"], n),
            "batch_window": rng.choice(["BATCH_00", "BATCH_01", "BATCH_02"], n),
            "is_outlier": rng.choice([0, 1], n, p=[0.95, 0.05]),
            # Labels (post-event)
            "true_fraud_label": rng.choice([0, 1], n, p=[0.95, 0.05]),
            "observed_label": rng.choice([0, 1], n, p=[0.90, 0.10]),
            "label_timestamp": [
                (base_date + timedelta(hours=i * 12 + 1)).isoformat() for i in range(n)
            ],
            "fraud_scenario": rng.choice(
                ["NONE", "CARD_NOT_PRESENT", "ACCOUNT_TAKEOVER", "IDENTITY_THEFT"],
                n,
                p=[0.95, 0.02, 0.02, 0.01],
            ),
        }
    )

    # Sort by timestamp
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    return df


@pytest.fixture
def sample_online_txn() -> dict:
    """Single online transaction dict (no labels)."""
    return {
        "transaction_id": "TXN_ONLINE_001",
        "event_timestamp": "2026-03-15T10:30:00",
        "transaction_sequence_id": 1,
        "event_version": "v2.1",
        "customer_id": "CUST_001",
        "customer_tenure_days": 365,
        "account_id": "ACC_001",
        "account_type": "CREDIT",
        "customer_country": "US",
        "customer_risk_segment": "LOW",
        "card_id": "CARD_001",
        "card_type": "CREDIT",
        "card_age_days": 200,
        "credit_limit": 5000,
        "card_country": "US",
        "card_status": "ACTIVE",
        "merchant_id": "MERCH_001",
        "merchant_category": "RETAIL",
        "merchant_country": "US",
        "merchant_region": "REG_US_01",
        "merchant_size": "MEDIUM",
        "merchant_age_days": 500,
        "merchant_risk_segment": "LOW",
        "device_id": "DEV_001",
        "device_type": "MOBILE_ANDROID",
        "device_os": "Android_14",
        "device_age_days": 100,
        "device_country": "US",
        "device_trust_level": "MEDIUM",
        "ip_id": "IP_001",
        "ip_country": "US",
        "connection_type": "residential",
        "network_risk_segment": "CLEAN",
        "proxy_type": "none",
        "amount": 150.00,
        "currency": "USD",
        "transaction_type": "purchase",
        "payment_channel": "WEB",
        "payment_method": "CARD_PRESENT",
        "authentication_method": "PIN",
        "installment_flag": 0,
        "international_flag": 0,
        "transaction_country": "US",
        "billing_country": "US",
        "terminal_id": "TERM_001",
        "branch_id": "BR_001",
        "processing_route": "ROUTE_A",
        "settlement_type": "INSTANT",
        "batch_window": "BATCH_00",
        "is_outlier": 0,
    }


@pytest.fixture
def feature_engineer():
    """Pre-configured FeatureEngineer."""
    from src.features.pipeline import FeatureEngineer

    return FeatureEngineer()


@pytest.fixture
def fitted_feature_engineer(synthetic_transactions):
    """FeatureEngineer fitted on synthetic training data."""
    from src.features.pipeline import FeatureEngineer

    eng = FeatureEngineer()
    eng.fit_categorical_maps(synthetic_transactions)
    return eng
