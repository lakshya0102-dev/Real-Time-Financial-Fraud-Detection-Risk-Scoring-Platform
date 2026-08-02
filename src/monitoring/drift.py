"""Temporal data and model drift monitoring suite.

Computes:
  - Population Stability Index (PSI)
  - Jensen-Shannon (JS) divergence
  - Feature & score distribution shifts between TRAIN, VAL, and TEST splits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


@dataclass
class DriftMetric:
    """Drift result for a single feature or variable."""

    feature_name: str
    psi: float
    js_divergence: float
    drift_status: str  # "NO_DRIFT", "SLIGHT_DRIFT", "SIGNIFICANT_DRIFT"


@dataclass
class PopulationDriftReport:
    """Summary report of temporal dataset drift."""

    baseline_name: str
    target_name: str
    feature_metrics: list[DriftMetric] = field(default_factory=list)
    overall_drift_score: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"TEMPORAL DRIFT REPORT: {self.baseline_name} → {self.target_name}",
            "=" * 60,
            f"Overall Drift Score (Mean PSI): {self.overall_drift_score:.4f}",
            "-" * 60,
        ]
        for m in self.feature_metrics:
            lines.append(
                f"  {m.feature_name:<35} | PSI: {m.psi:.4f} | JS: {m.js_divergence:.4f} | Status: {m.drift_status}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def calculate_psi(
    baseline: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two 1D numeric arrays.

    PSI < 0.1  : No significant change
    0.1 <= PSI < 0.2: Slight change / drift
    PSI >= 0.2 : Significant drift
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Filter out NaNs/Infs
    baseline = baseline[np.isfinite(baseline)]
    target = target[np.isfinite(target)]

    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    # Determine quantile bin edges based on baseline
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, percentiles)
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    b_counts, _ = np.histogram(baseline, bins=bin_edges)
    t_counts, _ = np.histogram(target, bins=bin_edges)

    b_pct = b_counts / len(baseline)
    t_pct = t_counts / len(target)

    # Clip to avoid div by zero / log(0)
    b_pct = np.clip(b_pct, eps, 1.0)
    t_pct = np.clip(t_pct, eps, 1.0)

    psi_val = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
    return float(psi_val)


def calculate_js_divergence(
    baseline: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculate Jensen-Shannon Divergence between two distributions (0 to 1)."""
    baseline = np.asarray(baseline, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    baseline = baseline[np.isfinite(baseline)]
    target = target[np.isfinite(target)]

    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    min_val = min(baseline.min(), target.min())
    max_val = max(baseline.max(), target.max())

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val - 1e-5, max_val + 1e-5, num_bins + 1)

    b_counts, _ = np.histogram(baseline, bins=bins)
    t_counts, _ = np.histogram(target, bins=bins)

    b_dist = (b_counts / len(baseline)) + eps
    t_dist = (t_counts / len(target)) + eps

    b_dist /= b_dist.sum()
    t_dist /= t_dist.sum()

    return float(jensenshannon(b_dist, t_dist))


def compute_temporal_drift(
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list[str],
    baseline_name: str = "TRAIN",
    target_name: str = "TEST",
) -> PopulationDriftReport:
    """Compute drift report comparing feature distributions between baseline and target splits."""
    metrics = []
    psi_list = []

    for f in features:
        if f not in baseline_df.columns or f not in target_df.columns:
            continue

        b_vals = baseline_df[f].values
        t_vals = target_df[f].values

        psi = calculate_psi(b_vals, t_vals)
        js = calculate_js_divergence(b_vals, t_vals)

        if psi >= 0.2:
            status = "SIGNIFICANT_DRIFT"
        elif psi >= 0.1:
            status = "SLIGHT_DRIFT"
        else:
            status = "NO_DRIFT"

        metrics.append(
            DriftMetric(
                feature_name=f,
                psi=round(psi, 4),
                js_divergence=round(js, 4),
                drift_status=status,
            )
        )
        psi_list.append(psi)

    overall_score = float(np.mean(psi_list)) if psi_list else 0.0

    return PopulationDriftReport(
        baseline_name=baseline_name,
        target_name=target_name,
        feature_metrics=metrics,
        overall_drift_score=round(overall_score, 4),
    )
