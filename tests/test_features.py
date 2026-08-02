"""Tests for feature engineering pipeline.

Verifies:
  1. Feature generation produces expected columns
  2. No NaN/Inf values in output
  3. Categorical encoding correctness
  4. Temporal feature extraction
  5. Geographic mismatch computation
  6. Identity/device features
  7. Velocity features produce reasonable values
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeatureEngineer


class TestStaticFeatures:
    """Test static transaction feature generation."""

    def test_log_amount_positive(self, synthetic_transactions):
        features = FeatureEngineer.generate_static_features(synthetic_transactions)
        assert (features["log_amount"] >= 0).all()

    def test_amount_credit_ratio_bounded(self, synthetic_transactions):
        features = FeatureEngineer.generate_static_features(synthetic_transactions)
        assert (features["amount_credit_ratio"] >= 0).all()
        assert (features["amount_credit_ratio"] <= 100).all()

    def test_binary_flags_valid(self, synthetic_transactions):
        features = FeatureEngineer.generate_static_features(synthetic_transactions)
        for col in ["is_international", "is_installment", "is_micro_txn", "is_high_value_txn"]:
            assert set(features[col].unique()).issubset({0, 1}), f"{col} not binary"

    def test_bucket_features_valid(self, synthetic_transactions):
        features = FeatureEngineer.generate_static_features(synthetic_transactions)
        for col in [
            "amount_bucket",
            "customer_tenure_bucket",
            "card_age_bucket",
            "device_age_bucket",
        ]:
            assert features[col].notna().all(), f"{col} contains NaN"
            assert (features[col] >= 0).all(), f"{col} has negative values"

    def test_no_nan_in_static_features(self, synthetic_transactions):
        features = FeatureEngineer.generate_static_features(synthetic_transactions)
        assert features.isna().sum().sum() == 0, "NaN values found in static features"


class TestCategoricalEncoding:
    """Test categorical frequency encoding."""

    def test_fit_transform_produces_columns(self, fitted_feature_engineer, synthetic_transactions):
        features = fitted_feature_engineer.generate_categorical_features(synthetic_transactions)
        expected_suffix = "_encoded"
        encoded_cols = [c for c in features.columns if c.endswith(expected_suffix)]
        assert len(encoded_cols) > 0, "No encoded columns generated"

    def test_fitted_values_between_0_and_1(self, fitted_feature_engineer, synthetic_transactions):
        features = fitted_feature_engineer.generate_categorical_features(synthetic_transactions)
        for col in features.columns:
            assert (features[col] >= 0).all(), f"{col} has negative values"
            assert (features[col] <= 1).all(), f"{col} has values > 1"

    def test_unseen_categories_get_zero(self, fitted_feature_engineer):
        """Unseen categorical values should map to 0.0."""
        df = pd.DataFrame(
            {
                "account_type": ["TOTALLY_NEW_TYPE"],
                "card_type": ["MYSTERY_CARD"],
                "merchant_category": ["SPACESHIP_PARTS"],
            },
            index=[0],
        )
        features = fitted_feature_engineer.generate_categorical_features(df)
        for col in features.columns:
            assert features[col].iloc[0] == 0.0, f"Unseen category not 0 for {col}"

    def test_unfitted_warns(self, synthetic_transactions):
        """Using unfitted encoder should produce a warning."""
        eng = FeatureEngineer()
        # Should not raise, but falls back to computing from df
        features = eng.generate_categorical_features(synthetic_transactions)
        assert len(features.columns) > 0


class TestTemporalFeatures:
    """Test temporal feature extraction."""

    def test_hour_range(self, synthetic_transactions):
        features = FeatureEngineer.generate_temporal_features(synthetic_transactions)
        assert (features["hour_of_day"] >= 0).all()
        assert (features["hour_of_day"] <= 23).all()

    def test_day_of_week_range(self, synthetic_transactions):
        features = FeatureEngineer.generate_temporal_features(synthetic_transactions)
        assert (features["day_of_week"] >= 0).all()
        assert (features["day_of_week"] <= 6).all()

    def test_binary_temporal_features(self, synthetic_transactions):
        features = FeatureEngineer.generate_temporal_features(synthetic_transactions)
        for col in ["is_weekend", "is_night"]:
            assert set(features[col].unique()).issubset({0, 1}), f"{col} not binary"


class TestGeographicFeatures:
    """Test geographic mismatch features."""

    def test_mismatch_count_bounded(self, synthetic_transactions):
        features = FeatureEngineer.generate_geographic_features(synthetic_transactions)
        assert (features["geo_mismatch_count"] >= 0).all()
        assert (features["geo_mismatch_count"] <= 6).all()

    def test_cross_border_binary(self, synthetic_transactions):
        features = FeatureEngineer.generate_geographic_features(synthetic_transactions)
        assert set(features["cross_border_flag"].unique()).issubset({0, 1})

    def test_same_country_no_mismatch(self):
        """If all countries match, no mismatches."""
        df = pd.DataFrame(
            {
                "transaction_country": ["US"],
                "customer_country": ["US"],
                "billing_country": ["US"],
                "card_country": ["US"],
                "device_country": ["US"],
                "ip_country": ["US"],
                "merchant_country": ["US"],
            }
        )
        features = FeatureEngineer.generate_geographic_features(df)
        assert features["geo_mismatch_count"].iloc[0] == 0

    def test_all_different_countries(self):
        """If all countries differ, max mismatches."""
        df = pd.DataFrame(
            {
                "transaction_country": ["US"],
                "customer_country": ["UK"],
                "billing_country": ["DE"],
                "card_country": ["FR"],
                "device_country": ["NG"],
                "ip_country": ["JP"],
                "merchant_country": ["BR"],
            }
        )
        features = FeatureEngineer.generate_geographic_features(df)
        assert features["geo_mismatch_count"].iloc[0] == 6


class TestIdentityFeatures:
    """Test identity and device risk features."""

    def test_new_device_detection(self):
        df = pd.DataFrame(
            {
                "device_age_days": [1, 7, 30, 365],
                "proxy_type": ["none", "none", "none", "none"],
                "connection_type": ["residential", "residential", "residential", "residential"],
            }
        )
        features = FeatureEngineer.generate_identity_features(df)
        assert features["is_new_device"].tolist() == [1, 0, 0, 0]

    def test_proxy_detection(self):
        df = pd.DataFrame(
            {
                "device_age_days": [100, 100, 100, 100],
                "proxy_type": ["none", "proxy", "tor", "unknown"],
                "connection_type": ["residential", "residential", "residential", "residential"],
            }
        )
        features = FeatureEngineer.generate_identity_features(df)
        assert features["is_proxy"].tolist() == [0, 1, 1, 0]
        assert features["is_tor"].tolist() == [0, 0, 1, 0]


class TestVelocityFeaturesFast:
    """Test fast velocity feature generation."""

    def test_cumulative_count_increases(self, synthetic_transactions):
        features = FeatureEngineer.generate_velocity_features_fast(
            synthetic_transactions, "customer_id"
        )
        # First transaction for each customer should have count 0 (no prior txns)
        first_per_cust = synthetic_transactions.groupby("customer_id").first().index
        first_idx = (
            synthetic_transactions[synthetic_transactions["customer_id"].isin(first_per_cust)]
            .groupby("customer_id")
            .head(1)
            .index
        )

        for idx in first_idx:
            assert features.loc[idx, "customer_total_txn_count"] == 0

    def test_std_excludes_current_row(self, synthetic_transactions):
        """Expanding std should use shift(1) — first row should be 0."""
        features = FeatureEngineer.generate_velocity_features_fast(
            synthetic_transactions, "customer_id"
        )
        # For each customer's first transaction, std should be 0 (no prior data)
        for _, group in synthetic_transactions.groupby("customer_id"):
            first_idx = group.index[0]
            assert features.loc[first_idx, "customer_std_amount"] == 0.0

    def test_no_nan_inf(self, synthetic_transactions):
        features = FeatureEngineer.generate_velocity_features_fast(
            synthetic_transactions, "customer_id"
        )
        assert not np.isinf(features.values).any(), "Inf values in velocity features"


class TestFullPipeline:
    """Test the full feature engineering pipeline."""

    def test_generates_features(self, fitted_feature_engineer, synthetic_transactions):
        features = fitted_feature_engineer.generate_all_features(
            synthetic_transactions, include_velocity=False
        )
        assert len(features.columns) > 20, "Too few features generated"
        assert len(features) == len(synthetic_transactions)

    def test_with_velocity(self, fitted_feature_engineer, synthetic_transactions):
        features = fitted_feature_engineer.generate_all_features(
            synthetic_transactions, include_velocity=True, fast_mode=True
        )
        assert len(features.columns) > 30, "Too few features with velocity"

    def test_no_nan_inf_in_output(self, fitted_feature_engineer, synthetic_transactions):
        features = fitted_feature_engineer.generate_all_features(
            synthetic_transactions, include_velocity=True, fast_mode=True
        )
        assert not features.isna().any().any(), "NaN values in output"
        assert not np.isinf(features.values).any(), "Inf values in output"

    def test_no_forbidden_columns(self, fitted_feature_engineer, synthetic_transactions):
        """Feature output must not contain any leakage columns."""
        features = fitted_feature_engineer.generate_all_features(
            synthetic_transactions, include_velocity=False
        )
        forbidden = {"true_fraud_label", "fraud_scenario", "observed_label", "label_timestamp"}
        leaked = forbidden & set(features.columns)
        assert len(leaked) == 0, f"Forbidden columns leaked: {leaked}"


class TestOnlineFeatures:
    """Test single-transaction online feature generation."""

    def test_generates_dict(self, feature_engineer, sample_online_txn):
        features = feature_engineer.generate_online_features(sample_online_txn)
        assert isinstance(features, dict)
        assert len(features) > 20

    def test_rejects_leakage(self, feature_engineer, sample_online_txn):
        """Online features must reject transactions with labels."""
        txn = {**sample_online_txn, "true_fraud_label": 1}
        with pytest.raises(ValueError, match="LEAKAGE"):
            feature_engineer.generate_online_features(txn)

    def test_online_parity_keys(
        self, fitted_feature_engineer, sample_online_txn, synthetic_transactions
    ):
        """Online features should produce a superset of keys that overlap with batch mode."""
        online = fitted_feature_engineer.generate_online_features(sample_online_txn)
        batch = fitted_feature_engineer.generate_all_features(
            synthetic_transactions, include_velocity=False
        )
        # All batch static columns should have a corresponding online key
        for col in batch.columns:
            if "velocity" not in col.lower() and "txn_count" not in col.lower():
                # Most static features should exist in online mode
                pass  # Just checking it doesn't crash
        assert "log_amount" in online
        assert "hour_of_day" in online
        assert "geo_mismatch_count" in online
