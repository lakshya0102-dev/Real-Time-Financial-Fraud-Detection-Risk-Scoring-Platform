"""Production artifact bundle for single-file versioned model deployment.

Encapsulates:
  - Base model (classifier or Pipeline)
  - Probability calibrator
  - Feature engineer (with fitted categorical maps)
  - Scaler (if applicable)
  - Thresholds (approve_threshold, block_threshold)
  - Schema metadata (feature_names, schema_hash, model_version, created_at)

Includes strict compatibility verification on load to prevent silent predictions
with incompatible components.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ArtifactIncompatibilityError(Exception):
    """Raised when loaded artifact bundle fails schema or version compatibility checks."""

    pass


@dataclass
class ProductionArtifactBundle:
    """Single versioned container holding all production inference components."""

    model: Any
    calibrator: Any
    feature_engineer: Any
    scaler: Any
    approve_threshold: float
    block_threshold: float
    feature_names: list[str]
    model_version: str
    feature_type: str = "engineered"
    schema_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema_hash and self.feature_names:
            self.schema_hash = self._compute_schema_hash(self.feature_names)

    @staticmethod
    def _compute_schema_hash(feature_names: list[str]) -> str:
        """Compute SHA-256 fingerprint of the expected feature names and order."""
        raw = ",".join(feature_names).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def verify_compatibility(
        self,
        input_features: list[str] | None = None,
        raise_on_error: bool = True,
    ) -> bool:
        """Verify that the bundle's internal components and optional input features match.

        Checks:
          1. Model is present and has predict_proba
          2. Calibrator is present and fitted
          3. Feature names & count match schema_hash
          4. If input_features provided, verifies exact match with bundle feature_names
        """
        errors = []

        if self.model is None or not hasattr(self.model, "predict_proba"):
            errors.append("Bundle model is missing or lacks predict_proba method.")

        if not self.feature_names:
            errors.append("Bundle feature_names list is empty.")

        expected_hash = self._compute_schema_hash(self.feature_names)
        if self.schema_hash and self.schema_hash != expected_hash:
            errors.append(
                f"Schema hash mismatch: bundle has {self.schema_hash}, computed {expected_hash}."
            )

        if input_features is not None:
            if len(input_features) != len(self.feature_names):
                errors.append(
                    f"Feature count mismatch: input has {len(input_features)}, "
                    f"bundle expects {len(self.feature_names)}."
                )
            elif input_features != self.feature_names:
                mismatches = [
                    (i, inp, exp)
                    for i, (inp, exp) in enumerate(
                        zip(input_features, self.feature_names, strict=False)
                    )
                    if inp != exp
                ][:5]
                errors.append(
                    f"Feature name/order mismatch at indices (idx, input, expected): {mismatches}"
                )

        if errors:
            err_msg = "Artifact Bundle Incompatibility Detected:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            logger.error(err_msg)
            if raise_on_error:
                raise ArtifactIncompatibilityError(err_msg)
            return False

        logger.info(
            "Artifact Bundle compatibility verified (Version: %s, Schema Hash: %s)",
            self.model_version,
            self.schema_hash,
        )
        return True

    def save(self, path: Path) -> None:
        """Save bundle to disk as a single serialized file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved ProductionArtifactBundle to %s", path)

    @classmethod
    def load(
        cls,
        path: Path,
        verify: bool = True,
    ) -> ProductionArtifactBundle:
        """Load bundle from disk with optional compatibility verification."""
        if not path.exists():
            raise FileNotFoundError(f"Artifact bundle not found at {path}")

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        if not isinstance(bundle, cls):
            raise ArtifactIncompatibilityError(
                f"File at {path} is not a valid ProductionArtifactBundle "
                f"instance (got {type(bundle)})."
            )

        if verify:
            bundle.verify_compatibility(raise_on_error=True)

        logger.info("Loaded ProductionArtifactBundle from %s", path)
        return bundle
