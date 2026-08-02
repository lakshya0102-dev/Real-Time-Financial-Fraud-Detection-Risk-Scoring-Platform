"""End-to-end training and model evaluation test on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from src.features.pipeline import FeatureEngineer
from src.calibration.calibrator import ProbabilityCalibrator
from src.scoring.decision_engine import DecisionEngine, RiskScorer


class TestModelTrainingAndInference:
    """Test model fitting, probability calibration, and decision scoring."""

    def test_end_to_end_model_pipeline(self, synthetic_transactions):
        # 1. Feature Engineering
        engineer = FeatureEngineer()
        engineer.fit_categorical_maps(synthetic_transactions)
        features = engineer.generate_all_features(synthetic_transactions, include_velocity=False)

        y = synthetic_transactions["true_fraud_label"].values
        X = features.values

        # 2. Fit Model
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        raw_probs = model.predict_proba(X)[:, 1]
        assert len(raw_probs) == len(synthetic_transactions)
        assert (raw_probs >= 0.0).all() and (raw_probs <= 1.0).all()

        # 3. Fit Calibrator
        calibrator = ProbabilityCalibrator()
        calibrator.fit_platt(y, raw_probs)
        cal_probs = calibrator.calibrate(raw_probs)
        assert len(cal_probs) == len(synthetic_transactions)
        assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()

        # 4. Risk Scorer & Decision Engine
        scorer = RiskScorer()
        engine = DecisionEngine()

        for idx, prob in enumerate(cal_probs[:10]):
            score = scorer.probability_to_score(prob)
            assert 0 <= score <= 1000

            level = scorer.score_to_level(score)
            assert level.value in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

            decision = engine.decide(f"TXN_{idx}", prob)
            assert decision.risk_score == score
            assert decision.decision.value in ["APPROVE", "REVIEW", "BLOCK"]


class TestArtifactBundle:
    """Test ProductionArtifactBundle creation, serialization, and compatibility verification."""

    def test_bundle_save_and_load(self, tmp_path, synthetic_transactions):
        from src.models.artifact_bundle import (
            ProductionArtifactBundle,
            ArtifactIncompatibilityError,
        )

        engineer = FeatureEngineer()
        engineer.fit_categorical_maps(synthetic_transactions)
        features = engineer.generate_all_features(synthetic_transactions, include_velocity=False)
        feature_names = list(features.columns)

        X = features.values
        y = synthetic_transactions["true_fraud_label"].values

        model = LogisticRegression(max_iter=100)
        model.fit(X, y)

        calibrator = ProbabilityCalibrator()
        calibrator.fit_platt(y, model.predict_proba(X)[:, 1])

        bundle = ProductionArtifactBundle(
            model=model,
            calibrator=calibrator,
            feature_engineer=engineer,
            scaler=None,
            approve_threshold=0.1,
            block_threshold=0.7,
            feature_names=feature_names,
            model_version="test-v1",
        )

        # Verification passes
        assert bundle.verify_compatibility()

        # Save & load
        save_file = tmp_path / "bundle.pkl"
        bundle.save(save_file)

        loaded = ProductionArtifactBundle.load(save_file, verify=True)
        assert loaded.model_version == "test-v1"
        assert loaded.feature_names == feature_names

    def test_bundle_detects_feature_mismatch(self, synthetic_transactions):
        from src.models.artifact_bundle import (
            ProductionArtifactBundle,
            ArtifactIncompatibilityError,
        )

        engineer = FeatureEngineer()
        features = engineer.generate_all_features(synthetic_transactions, include_velocity=False)
        feature_names = list(features.columns)
        X = features.values
        y = synthetic_transactions["true_fraud_label"].values

        model = LogisticRegression(max_iter=100)
        model.fit(X, y)

        bundle = ProductionArtifactBundle(
            model=model,
            calibrator=None,
            feature_engineer=engineer,
            scaler=None,
            approve_threshold=0.1,
            block_threshold=0.7,
            feature_names=feature_names,
            model_version="test-v1",
        )

        # Passing mismatched feature list should raise error
        wrong_features = feature_names[:-1]  # Missing one feature
        with pytest.raises(ArtifactIncompatibilityError, match="Feature count mismatch"):
            bundle.verify_compatibility(input_features=wrong_features)
