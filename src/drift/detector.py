"""
Drift detection — data and concept drift monitoring.

Monitors distribution changes using:
  - Population Stability Index (PSI)
  - Kolmogorov-Smirnov test (KS)
  - Jensen-Shannon divergence (JS)

Sets configurable warning/critical thresholds and triggers
retraining when conditions are met.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats

from src.config.settings import get_settings, MonitoringConfig

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a drift check for a single feature."""

    feature_name: str
    metric: str  # "psi", "ks", "js"
    value: float
    severity: DriftSeverity
    threshold_warning: float
    threshold_critical: float


@dataclass
class DriftReport:
    """Full drift report across all monitored features."""

    results: list[DriftResult]
    overall_severity: DriftSeverity
    drift_detected: bool
    summary: str

    @property
    def critical_features(self) -> list[str]:
        return [r.feature_name for r in self.results if r.severity == DriftSeverity.CRITICAL]

    @property
    def warning_features(self) -> list[str]:
        return [r.feature_name for r in self.results if r.severity == DriftSeverity.WARNING]


class DriftDetector:
    """Detects data drift between reference and current distributions."""

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self.config = config or get_settings().monitoring

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index (PSI)."""
        from src.monitoring.drift import calculate_psi

        return calculate_psi(reference, current, num_bins=n_bins)

    def compute_ks(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        statistic, _ = stats.ks_2samp(reference, current)
        return float(statistic)

    def compute_js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 50,
    ) -> float:
        """Compute Jensen-Shannon divergence."""
        from src.monitoring.drift import calculate_js_divergence

        return calculate_js_divergence(reference, current, num_bins=n_bins)

    def check_drift(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        metric: str = "psi",
    ) -> DriftResult:
        """Check drift for a single feature."""
        if metric == "psi":
            value = self.compute_psi(reference, current)
            warn_t = self.config.psi_warning
            crit_t = self.config.psi_critical
        elif metric == "ks":
            value = self.compute_ks(reference, current)
            warn_t = self.config.ks_warning
            crit_t = self.config.ks_critical
        elif metric == "js":
            value = self.compute_js_divergence(reference, current)
            warn_t = self.config.js_warning
            crit_t = self.config.js_critical
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if value >= crit_t:
            severity = DriftSeverity.CRITICAL
        elif value >= warn_t:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE

        return DriftResult(
            feature_name=feature_name,
            metric=metric,
            value=value,
            severity=severity,
            threshold_warning=warn_t,
            threshold_critical=crit_t,
        )

    def full_drift_check(
        self,
        reference_data: dict[str, np.ndarray],
        current_data: dict[str, np.ndarray],
        metrics: list[str] | None = None,
    ) -> DriftReport:
        """Check drift across all features."""
        if metrics is None:
            metrics = ["psi", "ks"]

        results = []
        for feature_name in reference_data:
            if feature_name not in current_data:
                continue

            ref = reference_data[feature_name]
            cur = current_data[feature_name]

            if len(ref) < 10 or len(cur) < 10:
                continue

            for metric in metrics:
                result = self.check_drift(feature_name, ref, cur, metric)
                results.append(result)

        # Overall severity
        if any(r.severity == DriftSeverity.CRITICAL for r in results):
            overall = DriftSeverity.CRITICAL
        elif any(r.severity == DriftSeverity.WARNING for r in results):
            overall = DriftSeverity.WARNING
        else:
            overall = DriftSeverity.NONE

        critical = [r for r in results if r.severity == DriftSeverity.CRITICAL]
        warning = [r for r in results if r.severity == DriftSeverity.WARNING]

        summary = (
            f"Drift Report: {len(results)} checks, {len(critical)} critical, {len(warning)} warning"
        )

        return DriftReport(
            results=results,
            overall_severity=overall,
            drift_detected=overall != DriftSeverity.NONE,
            summary=summary,
        )
