"""
Probability calibration for fraud models.

Compares raw model probabilities with:
  - Platt scaling (logistic regression calibration)
  - Isotonic regression calibration

Evaluates: Brier score, calibration curve, expected calibration error (ECE).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of probability calibration."""

    method: str
    brier_score_raw: float
    brier_score_calibrated: float
    ece_raw: float
    ece_calibrated: float
    calibrated_probs: np.ndarray
    raw_probs: np.ndarray


class ProbabilityCalibrator:
    """Calibrates model probabilities for reliable risk assessment.

    The final API should return a calibrated fraud probability.
    """

    def __init__(self) -> None:
        self._platt_model: LogisticRegression | None = None
        self._isotonic_model: IsotonicRegression | None = None
        self._active_method: str = "platt"

    def fit_platt(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression on raw probabilities)."""
        y_prob = np.asarray(y_prob).reshape(-1, 1)
        y_true = np.asarray(y_true)

        self._platt_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._platt_model.fit(y_prob, y_true)
        logger.info("Platt scaling fitted")

    def fit_isotonic(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit isotonic regression calibration."""
        y_prob = np.asarray(y_prob).ravel()
        y_true = np.asarray(y_true)

        self._isotonic_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._isotonic_model.fit(y_prob, y_true)
        logger.info("Isotonic regression fitted")

    def calibrate(
        self,
        y_prob: np.ndarray,
        method: str | None = None,
    ) -> np.ndarray:
        """Calibrate probabilities using the fitted model.

        Args:
            y_prob: Raw predicted probabilities.
            method: 'platt' or 'isotonic' (uses active method if None).

        Returns:
            Calibrated probabilities.
        """
        method = method or self._active_method
        y_prob = np.asarray(y_prob)

        if method == "platt":
            if self._platt_model is None:
                raise ValueError("Platt model not fitted. Call fit_platt first.")
            return self._platt_model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            if self._isotonic_model is None:
                raise ValueError("Isotonic model not fitted. Call fit_isotonic first.")
            return self._isotonic_model.predict(y_prob.ravel())
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob_raw: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, CalibrationResult]:
        """Evaluate all calibration methods.

        Returns:
            Dict of method → CalibrationResult.
        """
        y_true = np.asarray(y_true)
        y_prob_raw = np.asarray(y_prob_raw)

        results = {}

        # Raw (uncalibrated)
        brier_raw = float(brier_score_loss(y_true, y_prob_raw))
        ece_raw = self._expected_calibration_error(y_true, y_prob_raw, n_bins)

        # Platt scaling
        if self._platt_model is not None:
            cal_platt = self.calibrate(y_prob_raw, method="platt")
            brier_platt = float(brier_score_loss(y_true, cal_platt))
            ece_platt = self._expected_calibration_error(y_true, cal_platt, n_bins)
            results["platt"] = CalibrationResult(
                method="platt",
                brier_score_raw=brier_raw,
                brier_score_calibrated=brier_platt,
                ece_raw=ece_raw,
                ece_calibrated=ece_platt,
                calibrated_probs=cal_platt,
                raw_probs=y_prob_raw,
            )

        # Isotonic
        if self._isotonic_model is not None:
            cal_iso = self.calibrate(y_prob_raw, method="isotonic")
            brier_iso = float(brier_score_loss(y_true, cal_iso))
            ece_iso = self._expected_calibration_error(y_true, cal_iso, n_bins)
            results["isotonic"] = CalibrationResult(
                method="isotonic",
                brier_score_raw=brier_raw,
                brier_score_calibrated=brier_iso,
                ece_raw=ece_raw,
                ece_calibrated=ece_iso,
                calibrated_probs=cal_iso,
                raw_probs=y_prob_raw,
            )

        # Log comparison
        for method, result in results.items():
            logger.info(
                "Calibration [%s]: Brier %.6f → %.6f, ECE %.6f → %.6f",
                method,
                result.brier_score_raw,
                result.brier_score_calibrated,
                result.ece_raw,
                result.ece_calibrated,
            )

        return results

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_true)

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

            bin_count = mask.sum()
            if bin_count > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += (bin_count / total) * abs(bin_acc - bin_conf)

        return float(ece)

    def set_active_method(self, method: str) -> None:
        """Set the default calibration method."""
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown method: {method}")
        self._active_method = method
        logger.info("Active calibration method set to: %s", method)
