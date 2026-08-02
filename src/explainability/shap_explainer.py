"""
SHAP-based model explainability.

Provides:
  - Global feature importance
  - Local per-transaction explanations (top risk contributors)
  - Compatible with XGBoost, LightGBM, and tree-based models

IMPORTANT: Never expose misleading explanations. If the model/features
cannot support meaningful explanations, this module will report that clearly.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FraudExplainer:
    """SHAP-based explainer for fraud detection models.

    Supports TreeExplainer for gradient boosting models.
    Falls back gracefully if SHAP is unavailable.
    """

    def __init__(self, model: object, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self._explainer: object | None = None
        self._shap_available = False
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialize SHAP TreeExplainer if available."""
        try:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
            self._shap_available = True
            logger.info("SHAP TreeExplainer initialized with %d features", len(self.feature_names))
        except ImportError:
            logger.warning("SHAP not installed — explainability disabled")
        except Exception as e:
            logger.warning("SHAP initialization failed: %s — explainability disabled", e)

    def explain_single(
        self,
        feature_values: np.ndarray,
        top_n: int = 5,
    ) -> list[str]:
        """Explain a single prediction — return top N risk contributors.

        Args:
            feature_values: 1D or 2D array of feature values for one transaction.
            top_n: Number of top contributors to return.

        Returns:
            List of human-readable explanation strings.
        """
        if not self._shap_available or self._explainer is None:
            return ["Explainability unavailable (SHAP not loaded)"]

        try:
            x = np.asarray(feature_values)
            if x.ndim == 1:
                x = x.reshape(1, -1)

            shap_values = self._explainer.shap_values(x)

            # For binary classification, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                # Use class 1 (fraud) SHAP values
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                sv = shap_values[0]

            # Get top N features by absolute SHAP value
            abs_shap = np.abs(sv)
            top_indices = np.argsort(-abs_shap)[:top_n]

            explanations = []
            for idx in top_indices:
                feat_name = (
                    self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                )
                direction = "↑" if sv[idx] > 0 else "↓"
                explanations.append(f"{feat_name} ({direction} risk, SHAP={sv[idx]:.4f})")

            return explanations

        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            return [f"Explanation failed: {e}"]

    def global_feature_importance(
        self,
        x: np.ndarray,
        max_samples: int = 1000,
    ) -> dict[str, float]:
        """Compute global feature importance via mean |SHAP|.

        Args:
            x: Feature matrix (may be sampled for speed).
            max_samples: Maximum rows to use for SHAP computation.

        Returns:
            Dict of feature_name → mean absolute SHAP value, sorted descending.
        """
        if not self._shap_available or self._explainer is None:
            logger.warning("SHAP not available — returning empty importance")
            return {}

        try:
            # Sample if too large
            if len(x) > max_samples:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(x), max_samples, replace=False)
                x_sample = x[indices]
            else:
                x_sample = x

            shap_values = self._explainer.shap_values(x_sample)

            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            mean_abs = np.abs(sv).mean(axis=0)

            importance = {}
            for i, val in enumerate(mean_abs):
                name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                importance[name] = float(val)

            # Sort descending
            importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

            logger.info("Top 10 features by SHAP importance:")
            for i, (name, val) in enumerate(importance.items()):
                if i >= 10:
                    break
                logger.info("  %2d. %s: %.6f", i + 1, name, val)

            return importance

        except Exception as e:
            logger.error("Global SHAP computation failed: %s", e)
            return {}
