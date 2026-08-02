"""Leakage prevention regression tests.

These are MANDATORY tests that verify:
  1. Forbidden columns never appear in feature output
  2. Categorical encoding does not leak val/test distributions into train
  3. Velocity features use only strictly-prior data (temporal regression test)
  4. Online inference rejects post-event fields
  5. Feature pipeline output is temporally safe
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import FeatureEngineer
from src.validation.leakage import (
    LeakageValidator,
    LeakageViolationError,
    validate_no_leakage,
    filter_online_columns,
    get_safe_feature_columns,
)


class TestCategoricalEncodingLeakage:
    """Verify that categorical encoding does NOT leak val/test distributions."""

    def test_val_test_use_train_frequencies(self):
        """Frequency encoding on val/test must come from train-fitted maps."""
        # Create train with specific distribution
        train = pd.DataFrame(
            {
                "account_type": ["CREDIT"] * 90 + ["DEBIT"] * 10,
                "amount": [100.0] * 100,
            }
        )

        # Val has different distribution
        val = pd.DataFrame(
            {
                "account_type": ["CREDIT"] * 10 + ["DEBIT"] * 90,
                "amount": [200.0] * 100,
            }
        )

        eng = FeatureEngineer()
        eng.fit_categorical_maps(train)

        train_feat = eng.generate_categorical_features(train)
        val_feat = eng.generate_categorical_features(val)

        # CRITICAL: val should use TRAIN frequencies, not its own
        # In train: CREDIT=0.9, DEBIT=0.1
        # If leaking, val would show CREDIT=0.1, DEBIT=0.9
        credit_val_encoded = val_feat.loc[
            val["account_type"] == "CREDIT", "account_type_encoded"
        ].iloc[0]

        credit_train_encoded = train_feat.loc[
            train["account_type"] == "CREDIT", "account_type_encoded"
        ].iloc[0]

        assert credit_val_encoded == credit_train_encoded, (
            f"Leakage! Val CREDIT encoded as {credit_val_encoded}, "
            f"train CREDIT encoded as {credit_train_encoded}. "
            f"Val should use train frequencies."
        )

    def test_unseen_category_does_not_crash(self):
        """Categories in val/test not seen in train must map to 0, not crash."""
        train = pd.DataFrame(
            {
                "account_type": ["CREDIT", "DEBIT"],
                "amount": [100.0, 200.0],
            }
        )
        test = pd.DataFrame(
            {
                "account_type": ["TOTALLY_NEW"],
                "amount": [300.0],
            }
        )

        eng = FeatureEngineer()
        eng.fit_categorical_maps(train)
        features = eng.generate_categorical_features(test)
        assert features["account_type_encoded"].iloc[0] == 0.0


class TestTemporalLeakageRegression:
    """MANDATORY: Changing a future transaction must not change past features."""

    def test_future_does_not_affect_past_velocity(self):
        """Modifying a transaction in the future must not change features for past rows.

        This is the MOST IMPORTANT leakage test. If this fails, the model is invalid.
        """
        base = datetime(2026, 3, 1)
        df1 = pd.DataFrame(
            {
                "transaction_id": ["T1", "T2", "T3"],
                "event_timestamp": [
                    base,
                    base + timedelta(hours=1),
                    base + timedelta(hours=2),
                ],
                "customer_id": ["C1", "C1", "C1"],
                "amount": [100.0, 200.0, 300.0],
                # Minimal required columns
                "account_type": ["CREDIT"] * 3,
                "card_type": ["CREDIT"] * 3,
                "merchant_category": ["RETAIL"] * 3,
                "merchant_size": ["MEDIUM"] * 3,
                "transaction_type": ["purchase"] * 3,
                "payment_channel": ["WEB"] * 3,
                "payment_method": ["CARD_PRESENT"] * 3,
                "authentication_method": ["PIN"] * 3,
                "connection_type": ["residential"] * 3,
                "proxy_type": ["none"] * 3,
                "device_type": ["MOBILE_ANDROID"] * 3,
                "device_os": ["Android_14"] * 3,
                "card_status": ["ACTIVE"] * 3,
                "device_trust_level": ["MEDIUM"] * 3,
                "currency": ["USD"] * 3,
                "settlement_type": ["INSTANT"] * 3,
                "processing_route": ["ROUTE_A"] * 3,
                "customer_risk_segment": ["LOW"] * 3,
                "merchant_risk_segment": ["LOW"] * 3,
                "network_risk_segment": ["CLEAN"] * 3,
            }
        )

        # Now create df2 where T3's amount changes
        df2 = df1.copy()
        df2.loc[2, "amount"] = 99999.99  # Dramatic change to T3

        eng = FeatureEngineer()

        feat1 = eng.generate_velocity_features_fast(df1, "customer_id")
        feat2 = eng.generate_velocity_features_fast(df2, "customer_id")

        # T1's features MUST be identical in both versions
        t1_feat1 = feat1.iloc[0]
        t1_feat2 = feat2.iloc[0]
        pd.testing.assert_series_equal(
            t1_feat1,
            t1_feat2,
            check_names=False,
            obj="T1 features must not change when future T3 changes",
        )

        # T2's features MUST also be identical (T3 is in the future)
        t2_feat1 = feat1.iloc[1]
        t2_feat2 = feat2.iloc[1]
        pd.testing.assert_series_equal(
            t2_feat1,
            t2_feat2,
            check_names=False,
            obj="T2 features must not change when future T3 changes",
        )


class TestForbiddenColumnLeakage:
    """Test that forbidden post-event columns are never in feature output."""

    FORBIDDEN = {"true_fraud_label", "fraud_scenario", "observed_label", "label_timestamp"}

    def test_batch_features_exclude_forbidden(
        self, fitted_feature_engineer, synthetic_transactions
    ):
        features = fitted_feature_engineer.generate_all_features(
            synthetic_transactions, include_velocity=False
        )
        leaked = self.FORBIDDEN & set(features.columns)
        assert len(leaked) == 0, f"Forbidden columns in batch output: {leaked}"

    def test_online_features_reject_forbidden(self, feature_engineer, sample_online_txn):
        for col in self.FORBIDDEN:
            txn = {**sample_online_txn, col: "LEAKED_VALUE"}
            with pytest.raises(ValueError, match="LEAKAGE"):
                feature_engineer.generate_online_features(txn)

    def test_validate_no_leakage_dataframe(self):
        df = pd.DataFrame(
            {
                "amount": [100.0],
                "true_fraud_label": [1],
            }
        )
        with pytest.raises(LeakageViolationError):
            validate_no_leakage(df)

    def test_validate_no_leakage_dict(self):
        validator = LeakageValidator()
        with pytest.raises(LeakageViolationError):
            validator.validate_no_leakage_dict({"amount": 100, "fraud_scenario": "TEST"})

    def test_filter_removes_all_forbidden(self):
        df = pd.DataFrame(
            {
                "amount": [100.0],
                "true_fraud_label": [1],
                "fraud_scenario": ["TEST"],
                "observed_label": [0],
                "label_timestamp": ["2026-01-01"],
            }
        )
        filtered = filter_online_columns(df)
        leaked = self.FORBIDDEN & set(filtered.columns)
        assert len(leaked) == 0


class TestSafeFeatureColumns:
    """Test that get_safe_feature_columns excludes all dangerous columns."""

    def test_excludes_all_forbidden(self):
        from src.config.settings import get_settings

        settings = get_settings()
        all_cols = settings.dataset.expected_columns
        safe = get_safe_feature_columns(all_cols)

        for col in settings.leakage.forbidden_online_columns:
            assert col not in safe, f"Forbidden '{col}' in safe columns"

        for col in settings.leakage.excluded_feature_columns:
            assert col not in safe, f"Excluded '{col}' in safe columns"

        for col in settings.leakage.id_columns:
            assert col not in safe, f"ID column '{col}' in safe columns"


class TestOutlierExclusion:
    """MANDATORY: Verify is_outlier can NEVER enter feature schema (Audit #12)."""

    def test_is_outlier_excluded_from_safe_columns(self):
        from src.config.settings import get_settings

        settings = get_settings()
        all_cols = settings.dataset.expected_columns
        safe = get_safe_feature_columns(all_cols)
        assert "is_outlier" not in safe, "is_outlier MUST be excluded from safe feature columns!"

    def test_feature_engineer_does_not_contain_is_outlier(
        self, fitted_feature_engineer, synthetic_transactions
    ):
        features = fitted_feature_engineer.generate_all_features(
            synthetic_transactions, include_velocity=False
        )
        assert "is_outlier" not in features.columns, "is_outlier found in feature pipeline output!"
