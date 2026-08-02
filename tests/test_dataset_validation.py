"""
Tests for dataset validation, schema enforcement, and leakage prevention.

These tests verify:
  1. Dataset integrity (checksum, schema, row count, etc.)
  2. Leakage prevention (forbidden columns rejected from online inference)
  3. Schema enforcement (OnlineTransaction rejects post-event fields)
  4. Dataset loader (single source, fail-fast behavior)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from src.config.settings import get_settings
from src.data.dataset_validator import (
    compute_sha256,
    validate_dataset,
)
from src.data.loader import DatasetNotFoundError, get_dataset_path, load_dataset
from src.validation.leakage import (
    LeakageValidator,
    LeakageViolationError,
    filter_online_columns,
    get_safe_feature_columns,
    validate_no_leakage,
)
from src.validation.schema import (
    OnlineTransaction,
    PredictionResponse,
    TrainingEvent,
)

# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────


@pytest.fixture
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def sample_online_txn_data():
    """Minimal valid online transaction data (no post-event fields)."""
    return {
        "transaction_id": "TXN_TEST_001",
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
def sample_training_data(sample_online_txn_data):
    """Training event data — extends online data with labels."""
    return {
        **sample_online_txn_data,
        "true_fraud_label": 0,
        "observed_label": 0,
        "label_timestamp": "2026-03-15T12:00:00",
        "fraud_scenario": "NONE",
    }


@pytest.fixture
def sample_df_with_leakage():
    """DataFrame containing forbidden post-event columns."""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN_001", "TXN_002"],
            "amount": [100.0, 200.0],
            "true_fraud_label": [0, 1],
            "fraud_scenario": ["NONE", "CARD_NOT_PRESENT"],
            "observed_label": [0, 1],
            "label_timestamp": ["2026-01-01", "2026-01-02"],
        }
    )


@pytest.fixture
def sample_df_clean():
    """DataFrame without forbidden columns."""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN_001", "TXN_002"],
            "amount": [100.0, 200.0],
            "currency": ["USD", "EUR"],
        }
    )


# ══════════════════════════════════════════════════════
# DATASET INTEGRITY TESTS (INTEGRATION)
# ══════════════════════════════════════════════════════


@pytest.mark.integration
class TestDatasetExists:
    """Verify the canonical dataset exists and is accessible."""

    def test_dataset_file_exists(self, settings):
        """data/CFR_data.csv must exist."""
        assert settings.dataset.path.exists(), (
            f"Canonical dataset not found at {settings.dataset.path}"
        )

    def test_dataset_is_file(self, settings):
        """data/CFR_data.csv must be a regular file."""
        assert settings.dataset.path.is_file()

    def test_dataset_is_readable(self, settings):
        """Must be able to read the first few rows."""
        df = pd.read_csv(settings.dataset.path, nrows=5)
        assert len(df) == 5


@pytest.mark.integration
class TestDatasetChecksum:
    """Verify SHA-256 integrity."""

    def test_sha256_matches(self, settings):
        """SHA-256 must match the expected checksum."""
        actual = compute_sha256(settings.dataset.path)
        assert actual == settings.dataset.expected_sha256, (
            f"SHA-256 mismatch: expected {settings.dataset.expected_sha256}, got {actual}"
        )


@pytest.mark.integration
class TestDatasetSchema:
    """Verify the dataset has the expected schema."""

    def test_column_count(self, settings):
        """Must have exactly 54 columns."""
        df = pd.read_csv(settings.dataset.path, nrows=0)
        assert len(df.columns) == settings.dataset.expected_column_count

    def test_column_names(self, settings):
        """Column names must match the expected schema exactly."""
        df = pd.read_csv(settings.dataset.path, nrows=0)
        assert list(df.columns) == settings.dataset.expected_columns

    def test_row_count(self, settings):
        """Must have the expected number of rows."""
        df = pd.read_csv(settings.dataset.path, usecols=["transaction_id"])
        assert len(df) == settings.dataset.expected_row_count

    def test_no_null_values(self, settings):
        """No null values in any column."""
        df = pd.read_csv(settings.dataset.path)
        null_count = df.isnull().sum().sum()
        assert null_count == 0, f"Found {null_count} null values"

    def test_transaction_id_unique(self, settings):
        """All transaction IDs must be unique."""
        df = pd.read_csv(settings.dataset.path, usecols=["transaction_id"])
        dup_count = df["transaction_id"].duplicated().sum()
        assert dup_count == 0, f"Found {dup_count} duplicate transaction IDs"


@pytest.mark.integration
class TestDatasetDistribution:
    """Verify target and key column distributions."""

    def test_fraud_count(self, settings):
        """true_fraud_label=1 count must match expected."""
        df = pd.read_csv(settings.dataset.path, usecols=["true_fraud_label"])
        assert df["true_fraud_label"].sum() == settings.dataset.expected_fraud_count

    def test_observed_label_count(self, settings):
        """observed_label=1 count must match expected."""
        df = pd.read_csv(settings.dataset.path, usecols=["observed_label"])
        assert df["observed_label"].sum() == settings.dataset.expected_observed_label_count

    def test_outlier_count(self, settings):
        """is_outlier=1 count must match expected."""
        df = pd.read_csv(settings.dataset.path, usecols=["is_outlier"])
        assert df["is_outlier"].sum() == settings.dataset.expected_outlier_count

    def test_no_fraud_in_outliers(self, settings):
        """Outlier rows must contain zero fraud examples."""
        df = pd.read_csv(
            settings.dataset.path,
            usecols=["is_outlier", "true_fraud_label"],
        )
        fraud_in_outliers = df.loc[df["is_outlier"] == 1, "true_fraud_label"].sum()
        assert fraud_in_outliers == 0, f"Found {fraud_in_outliers} fraud cases in outlier rows"


@pytest.mark.integration
class TestDatasetTimestamps:
    """Verify timestamp range and ordering."""

    def test_timestamp_range(self, settings):
        """Timestamps must be within Jan–Jun 2026."""
        df = pd.read_csv(settings.dataset.path, usecols=["event_timestamp"])
        ts = pd.to_datetime(df["event_timestamp"])
        assert ts.min() >= pd.Timestamp("2025-12-31")
        assert ts.max() <= pd.Timestamp("2026-07-01")


@pytest.mark.integration
class TestDatasetValidator:
    """Test the full validation pipeline."""

    def test_validate_dataset_passes(self, settings):
        """Full validation should pass on the canonical dataset."""
        report = validate_dataset(fail_fast=True, config=settings.dataset)
        assert report.is_valid, report.summary()

    def test_validate_report_contents(self, settings):
        """Report should contain accurate statistics."""
        report = validate_dataset(fail_fast=False, config=settings.dataset)
        assert report.row_count == settings.dataset.expected_row_count
        assert report.column_count == settings.dataset.expected_column_count
        assert report.fraud_count == settings.dataset.expected_fraud_count
        assert report.null_count == 0
        assert report.duplicate_txn_count == 0


# ══════════════════════════════════════════════════════
# DATASET LOADER TESTS
# ══════════════════════════════════════════════════════


class TestDatasetLoader:
    """Test the single-source dataset loader."""

    @pytest.mark.integration
    def test_load_returns_dataframe(self):
        """load_dataset() must return a valid DataFrame."""
        df = load_dataset(nrows=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    @pytest.mark.integration
    def test_load_specific_columns(self):
        """Loading specific columns should work."""
        cols = ["transaction_id", "amount", "true_fraud_label"]
        df = load_dataset(columns=cols, nrows=5)
        assert list(df.columns) == cols

    def test_load_nonexistent_raises_error(self):
        """Loading from a missing path must raise DatasetNotFoundError."""
        fake_settings = MagicMock()
        fake_settings.dataset.path = Path("/nonexistent/path/data.csv")

        with (
            patch("src.data.loader.get_settings", return_value=fake_settings),
            pytest.raises(DatasetNotFoundError),
        ):
            load_dataset(nrows=1)

    @pytest.mark.integration
    def test_get_dataset_path(self):
        """get_dataset_path() must return the canonical path."""
        path = get_dataset_path()
        assert path.name == "CFR_data.csv"
        assert path.exists()


# ══════════════════════════════════════════════════════
# LEAKAGE PREVENTION TESTS
# ══════════════════════════════════════════════════════


class TestLeakagePrevention:
    """Tests that deliberately attempt to pass forbidden columns to inference."""

    def test_detect_true_fraud_label_in_dataframe(self, sample_df_with_leakage):
        """Must reject DataFrame containing true_fraud_label."""
        with pytest.raises(LeakageViolationError) as exc_info:
            validate_no_leakage(sample_df_with_leakage)
        assert "true_fraud_label" in str(exc_info.value)

    def test_detect_fraud_scenario_in_dataframe(self):
        """Must reject DataFrame containing fraud_scenario."""
        df = pd.DataFrame({"fraud_scenario": ["NONE"], "amount": [100.0]})
        with pytest.raises(LeakageViolationError):
            validate_no_leakage(df)

    def test_detect_observed_label_in_dataframe(self):
        """Must reject DataFrame containing observed_label."""
        df = pd.DataFrame({"observed_label": [0], "amount": [100.0]})
        with pytest.raises(LeakageViolationError):
            validate_no_leakage(df)

    def test_detect_label_timestamp_in_dataframe(self):
        """Must reject DataFrame containing label_timestamp."""
        df = pd.DataFrame({"label_timestamp": ["2026-01-01"], "amount": [100.0]})
        with pytest.raises(LeakageViolationError):
            validate_no_leakage(df)

    def test_clean_dataframe_passes(self, sample_df_clean):
        """DataFrame without forbidden columns should pass."""
        validate_no_leakage(sample_df_clean)  # Should not raise

    def test_detect_leakage_in_dict(self):
        """Must reject dictionary containing forbidden keys."""
        validator = LeakageValidator()
        data = {"amount": 100.0, "true_fraud_label": 1}
        with pytest.raises(LeakageViolationError):
            validator.validate_no_leakage_dict(data)

    def test_filter_removes_forbidden_columns(self, sample_df_with_leakage):
        """filter_online_columns must strip all forbidden columns."""
        filtered = filter_online_columns(sample_df_with_leakage)
        forbidden = {"true_fraud_label", "fraud_scenario", "observed_label", "label_timestamp"}
        remaining_forbidden = forbidden & set(filtered.columns)
        assert len(remaining_forbidden) == 0, (
            f"Forbidden columns still present: {remaining_forbidden}"
        )

    def test_safe_feature_columns_exclude_forbidden(self, settings):
        """get_safe_feature_columns must exclude all forbidden and ID columns."""
        all_cols = settings.dataset.expected_columns
        safe_cols = get_safe_feature_columns(all_cols)

        for col in settings.leakage.forbidden_online_columns:
            assert col not in safe_cols, f"Forbidden column '{col}' found in safe columns"

        for col in settings.leakage.excluded_feature_columns:
            assert col not in safe_cols, f"Excluded column '{col}' found in safe columns"

        for col in settings.leakage.id_columns:
            assert col not in safe_cols, f"ID column '{col}' found in safe columns"


# ══════════════════════════════════════════════════════
# SCHEMA ENFORCEMENT TESTS
# ══════════════════════════════════════════════════════


class TestSchemaEnforcement:
    """Test Pydantic schema validation for leakage boundaries."""

    def test_online_transaction_rejects_true_fraud_label(self, sample_online_txn_data):
        """OnlineTransaction must reject true_fraud_label."""
        data = {**sample_online_txn_data, "true_fraud_label": 1}
        with pytest.raises(ValidationError):  # Pydantic ValidationError (extra='forbid')
            OnlineTransaction.model_validate(data)

    def test_online_transaction_rejects_fraud_scenario(self, sample_online_txn_data):
        """OnlineTransaction must reject fraud_scenario."""
        data = {**sample_online_txn_data, "fraud_scenario": "CARD_NOT_PRESENT"}
        with pytest.raises(ValidationError):
            OnlineTransaction.model_validate(data)

    def test_online_transaction_rejects_observed_label(self, sample_online_txn_data):
        """OnlineTransaction must reject observed_label."""
        data = {**sample_online_txn_data, "observed_label": 0}
        with pytest.raises(ValidationError):
            OnlineTransaction.model_validate(data)

    def test_online_transaction_rejects_label_timestamp(self, sample_online_txn_data):
        """OnlineTransaction must reject label_timestamp."""
        data = {**sample_online_txn_data, "label_timestamp": "2026-01-01"}
        with pytest.raises(ValidationError):
            OnlineTransaction.model_validate(data)

    def test_online_transaction_accepts_valid_data(self, sample_online_txn_data):
        """OnlineTransaction must accept valid data without post-event fields."""
        txn = OnlineTransaction.model_validate(sample_online_txn_data)
        assert txn.transaction_id == "TXN_TEST_001"
        assert txn.amount == 150.00

    def test_training_event_accepts_labels(self, sample_training_data):
        """TrainingEvent must accept data WITH labels."""
        event = TrainingEvent.model_validate(sample_training_data)
        assert event.true_fraud_label == 0
        assert event.fraud_scenario == "NONE"

    def test_prediction_response_valid(self):
        """PredictionResponse must validate correctly."""
        resp = PredictionResponse(
            transaction_id="TXN_001",
            fraud_probability=0.85,
            risk_score=850,
            risk_level="CRITICAL",
            decision="BLOCK",
            model_version="v1.0.0",
            feature_timestamp="2026-03-15T10:30:00",
            inference_latency_ms=12.5,
        )
        assert resp.risk_score == 850

    def test_prediction_response_rejects_invalid_probability(self):
        """fraud_probability must be between 0 and 1."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                transaction_id="TXN_001",
                fraud_probability=1.5,  # Invalid
                risk_score=500,
                risk_level="HIGH",
                decision="BLOCK",
                model_version="v1.0.0",
                feature_timestamp="2026-03-15T10:30:00",
                inference_latency_ms=12.5,
            )


# ══════════════════════════════════════════════════════
# CONFIGURATION TESTS
# ══════════════════════════════════════════════════════


class TestConfiguration:
    """Test that central configuration is consistent."""

    def test_dataset_path_is_canonical(self, settings):
        """Dataset path must point to data/CFR_data.csv."""
        assert settings.dataset.path.name == "CFR_data.csv"
        assert "data" in str(settings.dataset.path)

    def test_forbidden_columns_defined(self, settings):
        """All 4 forbidden columns must be in the leakage config."""
        forbidden = set(settings.leakage.forbidden_online_columns)
        expected = {"true_fraud_label", "fraud_scenario", "observed_label", "label_timestamp"}
        assert forbidden == expected

    def test_is_outlier_excluded(self, settings):
        """is_outlier must be in excluded feature columns."""
        assert "is_outlier" in settings.leakage.excluded_feature_columns

    def test_risk_score_ranges_valid(self, settings):
        """Risk score boundaries must be consistent."""
        rs = settings.risk_scoring
        assert rs.min_score == 0
        assert rs.max_score == 1000
        assert rs.low_max < rs.medium_max < rs.high_max < rs.max_score

    def test_temporal_split_ordered(self, settings):
        """Train end must be before validation start, which is before test start."""
        ts = settings.temporal_split
        assert ts.train_end < ts.validation_start
        assert ts.validation_end < ts.test_start
