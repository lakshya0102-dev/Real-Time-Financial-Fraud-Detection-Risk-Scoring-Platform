"""Score sample transactions using the production model."""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import FeatureEngineer
from src.scoring.decision_engine import DecisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_TRANSACTIONS = [
    {
        "transaction_id": "TXN_PREDICT_001",
        "event_timestamp": "2026-06-15T14:30:00",
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
        "amount": 45.99,
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
    },
    {
        "transaction_id": "TXN_PREDICT_002",
        "event_timestamp": "2026-06-15T03:15:00",
        "transaction_sequence_id": 2,
        "event_version": "v2.1",
        "customer_id": "CUST_002",
        "customer_tenure_days": 15,
        "account_id": "ACC_002",
        "account_type": "DEBIT",
        "customer_country": "US",
        "customer_risk_segment": "HIGH",
        "card_id": "CARD_002",
        "card_type": "DEBIT",
        "card_age_days": 10,
        "credit_limit": 2000,
        "card_country": "US",
        "card_status": "ACTIVE",
        "merchant_id": "MERCH_002",
        "merchant_category": "ELECTRONICS",
        "merchant_country": "NG",
        "merchant_region": "REG_NG_01",
        "merchant_size": "SMALL",
        "merchant_age_days": 30,
        "merchant_risk_segment": "HIGH",
        "device_id": "DEV_002",
        "device_type": "DESKTOP_WINDOWS",
        "device_os": "Windows_11",
        "device_age_days": 2,
        "device_country": "RU",
        "device_trust_level": "LOW",
        "ip_id": "IP_002",
        "ip_country": "RU",
        "connection_type": "datacenter",
        "network_risk_segment": "SUSPICIOUS",
        "proxy_type": "tor",
        "amount": 4999.99,
        "currency": "USD",
        "transaction_type": "purchase",
        "payment_channel": "WEB",
        "payment_method": "CARD_NOT_PRESENT",
        "authentication_method": "NONE",
        "installment_flag": 0,
        "international_flag": 1,
        "transaction_country": "NG",
        "billing_country": "US",
        "terminal_id": "TERM_002",
        "branch_id": "BR_002",
        "processing_route": "ROUTE_B",
        "settlement_type": "DEFERRED",
        "batch_window": "BATCH_03",
        "is_outlier": 0,
    },
]


def main() -> None:
    artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    bundle_path = artifacts_dir / "production_bundle.pkl"

    if bundle_path.exists():
        from src.models.artifact_bundle import ProductionArtifactBundle

        bundle = ProductionArtifactBundle.load(bundle_path, verify=True)
        model = bundle.model
        calibrator = bundle.calibrator
        engineer = bundle.feature_engineer
        scaler = bundle.scaler
        feature_names = bundle.feature_names
        model_version = bundle.model_version
    else:
        model_path = artifacts_dir / "production_model.pkl"
        if not model_path.exists():
            logger.error("No production model found at %s. Run training first.", model_path)
            sys.exit(1)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

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
        model_version = manifest.get("model_version", "unknown")
        engineer = FeatureEngineer()

    decision_engine = DecisionEngine()

    logger.info("=" * 60)
    logger.info("INFERENCE — %d sample transactions", len(SAMPLE_TRANSACTIONS))
    logger.info("Model: %s", model_version)
    logger.info("=" * 60)

    for txn in SAMPLE_TRANSACTIONS:
        start = time.perf_counter()

        # Generate features
        features = engineer.generate_online_features(txn)

        # Align with model features
        if feature_names:
            feature_values = [features.get(f, 0) for f in feature_names]
        else:
            feature_values = list(features.values())

        X = np.array([feature_values], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply scaler if present and model is not a Pipeline
        if scaler is not None and not hasattr(model, "named_steps"):
            X = scaler.transform(X)

        # Predict
        raw_prob = float(model.predict_proba(X)[:, 1][0])

        if calibrator is not None:
            cal_prob = float(calibrator.calibrate(np.array([raw_prob]))[0])
        else:
            cal_prob = raw_prob

        # Decision
        decision = decision_engine.decide(txn["transaction_id"], cal_prob)
        latency_ms = (time.perf_counter() - start) * 1000

        result = {
            "transaction_id": str(txn["transaction_id"]),
            "fraud_probability": round(cal_prob, 6),
            "risk_score": int(decision.risk_score),
            "risk_level": str(decision.risk_level.value)
            if hasattr(decision.risk_level, "value")
            else str(decision.risk_level),
            "decision": str(decision.decision.value)
            if hasattr(decision.decision, "value")
            else str(decision.decision),
            "model_version": str(model_version),
            "inference_latency_ms": round(latency_ms, 2),
        }

        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    main()
