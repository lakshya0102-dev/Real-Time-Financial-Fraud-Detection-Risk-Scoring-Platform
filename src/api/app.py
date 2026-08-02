"""
FastAPI production inference service.

Endpoints:
  GET  /health        — health check
  GET  /ready         — readiness check
  GET  /version       — service version
  POST /predict       — single transaction scoring
  POST /predict/batch — batch scoring
  GET  /model/info    — model metadata
  GET  /metrics       — Prometheus metrics
  POST /admin/reload-model — hot-reload model
"""

from __future__ import annotations

import logging
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader

from src.config.settings import get_settings
from src.features.pipeline import FeatureEngineer
from src.monitoring.prometheus_metrics import (
    get_metrics,
    record_prediction,
    set_api_up,
    set_model_info,
)
from src.scoring.decision_engine import DecisionEngine
from src.security.auth import audit_logger
from src.validation.leakage import LeakageValidator
from src.validation.schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────


class ModelState:
    """Holds the loaded model and associated components."""

    def __init__(self) -> None:
        self.model = None
        self.calibrator = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer()
        self.decision_engine = DecisionEngine()
        self.leakage_validator = LeakageValidator()
        self.model_version: str = "unknown"
        self.feature_names: list[str] = []
        self.is_ready: bool = False
        self.load_time: str | None = None

    def load(self) -> None:
        """Load production model, calibrator, and scaler from artifacts."""
        settings = get_settings()
        artifacts_dir = settings.project_root / "artifacts" / "models"

        try:
            bundle_path = artifacts_dir / "production_bundle.pkl"
            if bundle_path.exists():
                from src.models.artifact_bundle import ProductionArtifactBundle

                bundle = ProductionArtifactBundle.load(bundle_path, verify=True)
                self.model = bundle.model
                self.calibrator = bundle.calibrator
                self.feature_engineer = bundle.feature_engineer
                self.scaler = bundle.scaler
                self.feature_names = bundle.feature_names
                self.model_version = bundle.model_version
                logger.info("Successfully loaded ProductionArtifactBundle (%s)", self.model_version)
            else:
                # Fallback to individual files
                model_path = artifacts_dir / "production_model.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info("Loaded production model from %s", model_path)
                else:
                    logger.warning("No production model found at %s", model_path)
                    return

                cal_path = artifacts_dir / "production_calibrator.pkl"
                if cal_path.exists():
                    with open(cal_path, "rb") as f:
                        self.calibrator = pickle.load(f)
                    logger.info("Loaded calibrator from %s", cal_path)

                scaler_path = artifacts_dir / "production_scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, "rb") as f:
                        self.scaler = pickle.load(f)
                    logger.info("Loaded scaler from %s", scaler_path)

                import json

                manifest_path = artifacts_dir / "production_manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    self.feature_names = manifest.get("feature_names", [])
                    self.model_version = manifest.get("model_name", "unknown")
                    logger.info(
                        "Loaded manifest: %s, %d features",
                        self.model_version,
                        len(self.feature_names),
                    )

            self.is_ready = True
            self.load_time = datetime.now(timezone.utc).isoformat()
            set_api_up(True)
            set_model_info(
                version=self.model_version,
                features=len(self.feature_names),
                calibration="platt" if self.calibrator else "none",
            )
            logger.info("Model state ready")

        except Exception as e:
            logger.error("Failed to load model: %s", e)
            self.is_ready = False
            set_api_up(False)


state = ModelState()


# ──────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Verify API key. In development, allow empty keys."""
    settings = get_settings()
    if settings.environment == "development":
        return api_key or "dev-key"

    valid_keys = os.environ.get("FRAUD_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


# ──────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting Fraud Detection API...")
    state.load()
    yield
    logger.info("Shutting down Fraud Detection API...")


# ──────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-Time Financial Fraud Detection & Risk Scoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID and timing to every response."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"

    return response


# ──────────────────────────────────────────────────────
# Health/Readiness Endpoints
# ──────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
async def ready():
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ready",
        "model_version": state.model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/version")
async def version():
    return {
        "api_version": "1.0.0",
        "model_version": state.model_version,
        "load_time": state.load_time,
    }


# ──────────────────────────────────────────────────────
# Prediction Endpoints
# ──────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key),
):
    """Score a single transaction for fraud risk."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()

    try:
        # Convert to dict and validate no leakage
        txn_dict = request.model_dump()
        state.leakage_validator.validate_no_leakage_dict(txn_dict)

        # Generate features
        features = state.feature_engineer.generate_online_features(txn_dict)

        # Align features with model's expected feature set
        if state.feature_names:
            feature_values = [features.get(f, 0) for f in state.feature_names]
        else:
            feature_values = list(features.values())

        x = np.array([feature_values], dtype=np.float64)

        # Replace NaN/Inf
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply scaler if loaded and model is not a Pipeline
        if state.scaler is not None and not hasattr(state.model, "named_steps"):
            x = state.scaler.transform(x)

        # Predict
        raw_prob = float(state.model.predict_proba(x)[:, 1][0])

        # Calibrate
        if state.calibrator is not None:
            cal_prob = float(state.calibrator.calibrate(np.array([raw_prob]))[0])
        else:
            cal_prob = raw_prob

        # Decision
        decision = state.decision_engine.decide(request.transaction_id, cal_prob)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics & audit log
        record_prediction(
            probability=cal_prob,
            risk_score=decision.risk_score,
            decision=decision.decision.value,
            inference_latency_s=latency_ms / 1000.0,
        )
        audit_logger.log_prediction(
            transaction_id=request.transaction_id,
            request_id=str(uuid.uuid4()),
            model_version=state.model_version,
            fraud_probability=cal_prob,
            risk_score=decision.risk_score,
            decision=decision.decision.value,
            latency_ms=latency_ms,
        )

        return PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=round(cal_prob, 6),
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            decision=decision.decision.value,
            model_version=state.model_version,
            feature_timestamp=datetime.now(timezone.utc).isoformat(),
            inference_latency_ms=round(latency_ms, 2),
            explanations=decision.explanations if decision.explanations else None,
            request_id=None,
        )

    except Exception as e:
        logger.error("Prediction failed for %s: %s", request.transaction_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key),
):
    """Score a batch of transactions."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    predictions = []

    for txn in request.transactions:
        try:
            resp = await predict(txn, api_key)
            predictions.append(resp)
        except HTTPException as exc:
            # Fail-safe: send failed transactions to REVIEW, never auto-approve
            logger.warning(
                "Batch item %s failed (%s), routing to REVIEW",
                txn.transaction_id,
                exc.detail,
            )
            predictions.append(
                PredictionResponse(
                    transaction_id=txn.transaction_id,
                    fraud_probability=0.0,
                    risk_score=0,
                    risk_level="UNKNOWN",
                    decision="REVIEW",
                    model_version=state.model_version,
                    feature_timestamp=datetime.now(timezone.utc).isoformat(),
                    inference_latency_ms=0.0,
                    explanations=[f"Scoring failed: {exc.detail}"],
                )
            )

    total_latency = (time.perf_counter() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_latency_ms=round(total_latency, 2),
        batch_size=len(predictions),
    )


# ──────────────────────────────────────────────────────
# Model Info & Admin
# ──────────────────────────────────────────────────────


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    return {
        "model_version": state.model_version,
        "feature_count": len(state.feature_names),
        "is_ready": state.is_ready,
        "load_time": state.load_time,
        "calibrator": state.calibrator is not None,
    }


@app.post("/admin/reload-model")
async def reload_model(api_key: str = Depends(verify_api_key)):
    """Hot-reload the production model."""
    state.load()
    audit_logger.log_admin_action(
        action="reload_model",
        user_id=api_key,
        details=f"Model version: {state.model_version}",
    )
    return {
        "status": "reloaded",
        "model_version": state.model_version,
        "is_ready": state.is_ready,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(
        get_metrics(),
        media_type="text/plain",
    )
