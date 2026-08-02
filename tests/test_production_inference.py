"""Real deterministic production inference integration test (Audit #9).

Tests:
  1. Loads production_bundle.pkl from disk
  2. Scores sample raw transaction through online feature extraction
  3. Validates return schema: fraud_probability, risk_score, decision, risk_level
  4. Runs sub-process prediction test to prove deterministic reproducibility
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.features.pipeline import FeatureEngineer
from src.models.artifact_bundle import ProductionArtifactBundle
from src.scoring.decision_engine import DecisionEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_TXN = {
    "transaction_id": "TXN_AUDIT_DETERMINISTIC_001",
    "event_timestamp": "2026-06-15T12:00:00",
    "transaction_sequence_id": 99,
    "customer_id": "CUST_DETERMINISTIC",
    "customer_tenure_days": 180,
    "account_id": "ACC_DETERMINISTIC",
    "account_type": "CHECKING",
    "customer_country": "US",
    "customer_risk_segment": "LOW",
    "card_id": "CARD_DETERMINISTIC",
    "card_type": "DEBIT",
    "card_age_days": 120,
    "credit_limit": 3000,
    "card_country": "US",
    "card_status": "ACTIVE",
    "merchant_id": "MERCH_DETERMINISTIC",
    "merchant_category": "GROCERY",
    "merchant_country": "US",
    "merchant_region": "REG_US_01",
    "merchant_size": "LARGE",
    "merchant_age_days": 1000,
    "merchant_risk_segment": "LOW",
    "device_id": "DEV_DETERMINISTIC",
    "device_type": "MOBILE_IOS",
    "device_os": "iOS_17",
    "device_age_days": 50,
    "device_country": "US",
    "device_trust_level": "HIGH",
    "ip_id": "IP_DETERMINISTIC",
    "ip_country": "US",
    "connection_type": "residential",
    "network_risk_segment": "CLEAN",
    "proxy_type": "none",
    "amount": 75.50,
    "currency": "USD",
    "transaction_type": "purchase",
    "payment_channel": "MOBILE_APP",
    "payment_method": "CARD_PRESENT",
    "authentication_method": "BIOMETRIC",
    "installment_flag": 0,
    "international_flag": 0,
    "transaction_country": "US",
    "billing_country": "US",
    "terminal_id": "TERM_DET",
    "branch_id": "BR_DET",
    "processing_route": "ROUTE_A",
    "settlement_type": "INSTANT",
    "batch_window": "BATCH_01",
    "is_outlier": 0,
}


def run_single_inference(txn: dict) -> dict:
    """Perform end-to-end inference using ProductionArtifactBundle."""
    artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    bundle_path = artifacts_dir / "production_bundle.pkl"

    bundle = ProductionArtifactBundle.load(bundle_path, verify=True)
    engineer = bundle.feature_engineer or FeatureEngineer()
    decision_engine = DecisionEngine()

    features = engineer.generate_online_features(txn)
    feature_values = [features.get(f, 0) for f in bundle.feature_names]

    x = np.array([feature_values], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if bundle.scaler is not None and not hasattr(bundle.model, "named_steps"):
        x = bundle.scaler.transform(x)

    raw_prob = float(bundle.model.predict_proba(x)[:, 1][0])
    cal_prob = (
        float(bundle.calibrator.calibrate(np.array([raw_prob]))[0])
        if bundle.calibrator
        else raw_prob
    )

    decision = decision_engine.decide(txn["transaction_id"], cal_prob)

    return {
        "transaction_id": txn["transaction_id"],
        "fraud_probability": round(cal_prob, 6),
        "risk_score": int(decision.risk_score),
        "risk_level": str(decision.risk_level),
        "decision": str(decision.decision.value),
        "model_version": bundle.model_version,
    }


@pytest.mark.integration
class TestProductionInferenceDeterminism:
    """Test real production inference loading and determinism across subprocesses."""

    def test_direct_inference(self):
        res = run_single_inference(SAMPLE_TXN)
        assert "fraud_probability" in res
        assert 0.0 <= res["fraud_probability"] <= 1.0
        assert 0 <= res["risk_score"] <= 1000
        assert res["decision"] in ["APPROVE", "REVIEW", "BLOCK"]

    def test_deterministic_across_subprocess_restarts(self):
        code = (
            "import json, sys; sys.path.insert(0, '.'); "
            "from tests.test_production_inference import run_single_inference, SAMPLE_TXN; "
            "res = run_single_inference(SAMPLE_TXN); "
            "print(json.dumps(res))"
        )

        p1 = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        out1 = json.loads(p1.stdout.strip().split("\n")[-1])

        p2 = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        out2 = json.loads(p2.stdout.strip().split("\n")[-1])

        assert out1 == out2, (
            f"Inference mismatch across Python restarts!\nRun 1: {out1}\nRun 2: {out2}"
        )
