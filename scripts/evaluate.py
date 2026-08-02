"""Evaluate a trained fraud detection model on the test set."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from src.data.loader import load_dataset
from src.features.pipeline import FeatureEngineer
from src.models.metrics import compute_fraud_metrics, compute_metrics_at_thresholds
from src.models.temporal_split import temporal_train_val_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    bundle_path = artifacts_dir / "production_bundle.pkl"

    if bundle_path.exists():
        from src.models.artifact_bundle import ProductionArtifactBundle

        bundle = ProductionArtifactBundle.load(bundle_path, verify=True)
        model = bundle.model
        calibrator = bundle.calibrator
        feature_names = bundle.feature_names
        feature_type = bundle.feature_type
        scaler = bundle.scaler
        logger.info("Loaded ProductionArtifactBundle (%s)", bundle.model_version)
    else:
        model_path = artifacts_dir / "production_model.pkl"
        if not model_path.exists():
            logger.error("No production model found at %s. Run training first.", model_path)
            sys.exit(1)

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", model_path)

        cal_path = artifacts_dir / "production_calibrator.pkl"
        calibrator = None
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                calibrator = pickle.load(f)

        scaler_path = artifacts_dir / "production_scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        manifest_path = artifacts_dir / "production_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

        feature_names = manifest.get("feature_names", [])
        feature_type = manifest.get("feature_type", "engineered")

    # Load and split data
    df = load_dataset(parse_dates=True)
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    split = temporal_train_val_test_split(df)

    train_df = df.loc[split.train_idx].copy()
    test_df = df.loc[split.test_idx].copy()
    y_test = np.asarray(test_df["true_fraud_label"].values, dtype=np.int32)
    test_amounts = np.asarray(test_df["amount"].values, dtype=np.float64)

    # Feature engineering
    engineer = FeatureEngineer()
    engineer.fit_categorical_maps(train_df)

    include_velocity = feature_type == "engineered"
    test_features = engineer.generate_all_features(
        test_df, include_velocity=include_velocity, fast_mode=True
    )

    # Align features
    if feature_names:
        available = [f for f in feature_names if f in test_features.columns]
        x_test = test_features[available].values
    else:
        x_test = test_features.values

    # Apply scaler if present and model is not a Pipeline
    if scaler is not None and not hasattr(model, "named_steps"):
        x_test = scaler.transform(x_test)

    # Predict
    y_prob = model.predict_proba(x_test)[:, 1]

    y_prob_cal = calibrator.calibrate(y_prob) if calibrator is not None else y_prob

    # Metrics
    metrics = compute_fraud_metrics(y_test, y_prob_cal, amounts=test_amounts)
    logger.info("Test Evaluation:\n%s", metrics.summary())

    # Multi-threshold analysis
    threshold_metrics = compute_metrics_at_thresholds(y_test, y_prob_cal, amounts=test_amounts)
    logger.info("\nThreshold Analysis:")
    for m in threshold_metrics:
        logger.info(
            "  t=%.2f: P=%.3f R=%.3f F1=%.3f PR-AUC=%.3f Cost=$%.0f",
            m.threshold,
            m.precision,
            m.recall,
            m.f1,
            m.pr_auc,
            m.total_expected_cost,
        )


if __name__ == "__main__":
    main()
