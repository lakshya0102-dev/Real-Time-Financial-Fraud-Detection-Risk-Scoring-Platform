"""
Security module — authentication, authorization, and audit logging.

Implements:
  - API key authentication
  - JWT token generation/validation
  - Role-based access control (RBAC)
  - Structured audit logging (never logs secrets)
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class Role(str, Enum):
    READER = "reader"  # Can read predictions
    SCORER = "scorer"  # Can score transactions
    ADMIN = "admin"  # Can reload models, configure
    AUDITOR = "auditor"  # Can view audit logs


# Default API keys for development (MUST be overridden in production)
DEV_API_KEYS = {
    "dev-key-001": {"role": Role.ADMIN, "name": "Development Admin"},
    "dev-key-002": {"role": Role.SCORER, "name": "Development Scorer"},
}


def validate_api_key(api_key: str) -> dict | None:
    """Validate an API key and return associated metadata.

    In production, keys should be stored in a secure vault.
    """
    settings = get_settings()

    if settings.environment == "development":
        if api_key in DEV_API_KEYS:
            return DEV_API_KEYS[api_key]
        # In dev, accept any key
        return {"role": Role.SCORER, "name": "dev-user"}

    # Production: check environment-provided keys
    valid_keys_str = os.environ.get("FRAUD_API_KEYS", "")
    if valid_keys_str:
        valid_keys = [k.strip() for k in valid_keys_str.split(",")]
        if api_key in valid_keys:
            return {"role": Role.SCORER, "name": "api-user"}

    return None


def generate_jwt(user_id: str, role: Role) -> str:
    """Generate a JWT token."""
    try:
        from jose import jwt

        settings = get_settings()
        payload = {
            "sub": user_id,
            "role": role.value,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.api.jwt_expiry_minutes),
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(
            payload, settings.api.jwt_secret_key, algorithm=settings.api.jwt_algorithm
        )
    except ImportError:
        logger.warning("python-jose not installed, returning dummy token")
        return f"dev-token-{user_id}"


def verify_jwt(token: str) -> dict | None:
    """Verify a JWT token and return payload."""
    try:
        from jose import jwt

        settings = get_settings()
        payload = jwt.decode(
            token,
            settings.api.jwt_secret_key,
            algorithms=[settings.api.jwt_algorithm],
        )
        return payload
    except ImportError:
        if token.startswith("dev-token-"):
            return {"sub": token.replace("dev-token-", ""), "role": "admin"}
        return None
    except Exception:
        return None


class AuditLogger:
    """Structured audit logging for predictions and admin actions.

    NEVER logs:
      - Passwords
      - Authentication credentials
      - Raw secrets
      - Sensitive tokens
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("fraud.audit")

    def log_prediction(
        self,
        transaction_id: str,
        request_id: str,
        model_version: str,
        fraud_probability: float,
        risk_score: int,
        decision: str,
        latency_ms: float,
    ) -> None:
        """Log a prediction audit record."""
        self._logger.info(
            "PREDICTION | txn_id=%s | req_id=%s | model=%s | "
            "prob=%.6f | score=%d | decision=%s | latency_ms=%.2f",
            transaction_id,
            request_id,
            model_version,
            fraud_probability,
            risk_score,
            decision,
            latency_ms,
        )

    def log_admin_action(
        self,
        action: str,
        user_id: str,
        details: str = "",
    ) -> None:
        """Log an admin action."""
        self._logger.info(
            "ADMIN | action=%s | user=%s | details=%s",
            action,
            user_id,
            details,
        )

    def log_model_event(
        self,
        event: str,
        model_version: str,
        details: str = "",
    ) -> None:
        """Log a model lifecycle event."""
        self._logger.info(
            "MODEL | event=%s | version=%s | details=%s",
            event,
            model_version,
            details,
        )


# Singleton
audit_logger = AuditLogger()
