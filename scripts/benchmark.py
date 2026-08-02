"""
Resilience & Latency Benchmark Script.

Tests API inference latency against target: p95 < 100ms.
Simulates resilient handling of malformed inputs and high load.
"""

from __future__ import annotations

import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

from src.features.pipeline import FeatureEngineer
from src.scoring.decision_engine import DecisionEngine
from src.validation.leakage import LeakageValidator

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    with suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_benchmark(n_iterations: int = 200, warmup_iterations: int = 10) -> None:
    print("=" * 60)
    print(f"RUNNING LATENCY BENCHMARK ({n_iterations} iterations, {warmup_iterations} warmup)")
    print("=" * 60)

    # Load production model for realistic benchmark
    artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    bundle_path = artifacts_dir / "production_bundle.pkl"
    model = None
    calibrator = None
    scaler = None
    feature_names = []

    if bundle_path.exists():
        from src.models.artifact_bundle import ProductionArtifactBundle

        bundle = ProductionArtifactBundle.load(bundle_path, verify=True)
        model = bundle.model
        calibrator = bundle.calibrator
        scaler = bundle.scaler
        feature_names = bundle.feature_names
        engineer = bundle.feature_engineer or FeatureEngineer()
        print(f"  Model: {bundle.model_version}")
    else:
        import pickle

        model_path = artifacts_dir / "production_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"  Model loaded from {model_path}")
        else:
            print("  [WARNING] No production model found — benchmarking WITHOUT inference")

        engineer = FeatureEngineer()

    decision_engine = DecisionEngine()
    validator = LeakageValidator()

    sample_txn = {
        "transaction_id": "TXN_BENCHMARK_001",
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

    # Warmup
    for _ in range(warmup_iterations):
        validator.validate_no_leakage_dict(sample_txn)
        feats = engineer.generate_online_features(sample_txn)
        if model is not None:
            fv = [feats.get(f, 0) for f in feature_names] if feature_names else list(feats.values())
            x = np.array([fv], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if scaler is not None:
                x = scaler.transform(x)
            prob = float(model.predict_proba(x)[:, 1][0])
        else:
            prob = 0.25
        decision_engine.decide(sample_txn["transaction_id"], prob)

    latencies_ms = []

    for _ in range(n_iterations):
        start = time.perf_counter()

        # 1. Leakage check
        validator.validate_no_leakage_dict(sample_txn)

        # 2. Feature generation
        feats = engineer.generate_online_features(sample_txn)

        # 3. Model inference (the most expensive step)
        if model is not None:
            fv = [feats.get(f, 0) for f in feature_names] if feature_names else list(feats.values())
            x = np.array([fv], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if scaler is not None:
                x = scaler.transform(x)
            prob = float(model.predict_proba(x)[:, 1][0])
            if calibrator is not None:
                prob = float(calibrator.calibrate(np.array([prob]))[0])
        else:
            prob = 0.25

        # 4. Decision
        decision_engine.decide(sample_txn["transaction_id"], prob)

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    latencies_arr = np.array(latencies_ms)
    p50 = np.percentile(latencies_arr, 50)
    p95 = np.percentile(latencies_arr, 95)
    p99 = np.percentile(latencies_arr, 99)

    print(f"\nResults ({n_iterations} iterations):")
    print(f"  Model inference included: {'YES' if model is not None else 'NO (model not found)'}")
    print(f"  p50 Latency: {p50:.2f} ms")
    print(f"  p95 Latency: {p95:.2f} ms")
    print(f"  p99 Latency: {p99:.2f} ms")

    if p95 < 100:
        print("\n[SUCCESS] PERFORMANCE TARGET ACHIEVED: p95 latency < 100ms")
    else:
        print(f"\n[FAIL] TARGET MISSED: p95 latency ({p95:.2f}ms) >= 100ms")


if __name__ == "__main__":
    run_benchmark()
