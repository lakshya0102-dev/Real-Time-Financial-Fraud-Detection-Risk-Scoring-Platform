"""
Comprehensive fraud-appropriate metrics suite.

Reports all metrics required for imbalanced fraud classification:
  - PR-AUC, ROC-AUC, precision, recall, F1
  - Fraud recall, false-positive rate, specificity
  - Confusion matrix
  - Precision@K, Recall@K
  - Expected financial loss / cost metrics
  - Performance at operational thresholds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config.settings import get_settings, CostConfig

logger = logging.getLogger(__name__)


@dataclass
class FraudMetrics:
    """Comprehensive fraud detection metrics."""

    # Core ML metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    random_baseline_pr_auc: float = 0.0
    pr_auc_lift: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fraud_recall: float = 0.0  # Same as recall (for fraud = positive class)
    specificity: float = 0.0
    false_positive_rate: float = 0.0

    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Calibration
    brier_score: float = 0.0

    # At-K metrics
    precision_at_100: float = 0.0
    precision_at_500: float = 0.0
    recall_at_100: float = 0.0
    recall_at_500: float = 0.0

    # Cost metrics
    fraud_loss_prevented: float = 0.0
    fraud_loss_missed: float = 0.0
    false_positive_cost: float = 0.0
    total_expected_cost: float = 0.0
    net_benefit: float = 0.0

    # Threshold
    threshold: float = 0.5

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "FRAUD DETECTION METRICS",
            "=" * 60,
            f"Threshold:        {self.threshold:.4f}",
            f"ROC-AUC:          {self.roc_auc:.4f}",
            f"PR-AUC:           {self.pr_auc:.4f}",
            f"Precision:        {self.precision:.4f}",
            f"Recall (Fraud):   {self.recall:.4f}",
            f"F1:               {self.f1:.4f}",
            f"Specificity:      {self.specificity:.4f}",
            f"FPR:              {self.false_positive_rate:.4f}",
            f"Brier Score:      {self.brier_score:.6f}",
            "-" * 60,
            f"Confusion Matrix:",
            f"  TP: {self.true_positives:>8,}  |  FP: {self.false_positives:>8,}",
            f"  FN: {self.false_negatives:>8,}  |  TN: {self.true_negatives:>8,}",
            "-" * 60,
            f"P@100:            {self.precision_at_100:.4f}",
            f"P@500:            {self.precision_at_500:.4f}",
            f"R@100:            {self.recall_at_100:.4f}",
            f"R@500:            {self.recall_at_500:.4f}",
            "-" * 60,
            f"Fraud Prevented:  ${self.fraud_loss_prevented:>12,.2f}",
            f"Fraud Missed:     ${self.fraud_loss_missed:>12,.2f}",
            f"FP Cost:          ${self.false_positive_cost:>12,.2f}",
            f"Total Cost:       ${self.total_expected_cost:>12,.2f}",
            f"Net Benefit:      ${self.net_benefit:>12,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_fraud_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    amounts: Optional[np.ndarray] = None,
    cost_config: Optional[CostConfig] = None,
) -> FraudMetrics:
    """Compute comprehensive fraud detection metrics.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted fraud probabilities.
        y_pred: Binary predictions (optional, derived from threshold).
        threshold: Classification threshold.
        amounts: Transaction amounts (for cost calculation).
        cost_config: Cost parameters.

    Returns:
        FraudMetrics with all computed values.
    """
    if cost_config is None:
        cost_config = get_settings().cost

    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(np.int32)

    metrics = FraudMetrics(threshold=threshold)

    # Core metrics
    if len(np.unique(y_true)) > 1:
        metrics.roc_auc = float(roc_auc_score(y_true, y_prob))
        metrics.pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        logger.warning("Only one class present in y_true, AUC metrics set to 0.0")

    metrics.precision = float(precision_score(y_true, y_pred, zero_division=0))
    metrics.recall = float(recall_score(y_true, y_pred, zero_division=0))
    metrics.fraud_recall = metrics.recall
    metrics.f1 = float(f1_score(y_true, y_pred, zero_division=0))
    metrics.brier_score = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics.true_negatives = int(cm[0, 0])
    metrics.false_positives = int(cm[0, 1])
    metrics.false_negatives = int(cm[1, 0])
    metrics.true_positives = int(cm[1, 1])

    # Specificity and FPR
    if (metrics.true_negatives + metrics.false_positives) > 0:
        metrics.specificity = metrics.true_negatives / (
            metrics.true_negatives + metrics.false_positives
        )
        metrics.false_positive_rate = metrics.false_positives / (
            metrics.true_negatives + metrics.false_positives
        )

    # Precision@K, Recall@K
    sorted_idx = np.argsort(-y_prob)
    total_fraud = y_true.sum()

    for k in [100, 500]:
        if k <= len(y_true):
            top_k_true = y_true[sorted_idx[:k]]
            pk = top_k_true.sum() / k
            rk = top_k_true.sum() / max(total_fraud, 1)
        else:
            pk = rk = 0.0
        if k == 100:
            metrics.precision_at_100 = float(pk)
            metrics.recall_at_100 = float(rk)
        elif k == 500:
            metrics.precision_at_500 = float(pk)
            metrics.recall_at_500 = float(rk)

    # Cost metrics
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=np.float64)
        # Fraud caught (blocked or reviewed)
        caught_mask = (y_true == 1) & (y_pred == 1)
        missed_mask = (y_true == 1) & (y_pred == 0)

        metrics.fraud_loss_prevented = float(amounts[caught_mask].sum())
        metrics.fraud_loss_missed = float(amounts[missed_mask].sum())
    else:
        metrics.fraud_loss_prevented = float(metrics.true_positives * cost_config.avg_fraud_loss)
        metrics.fraud_loss_missed = float(metrics.false_negatives * cost_config.avg_fraud_loss)

    metrics.false_positive_cost = float(metrics.false_positives * cost_config.false_positive_cost)

    metrics.total_expected_cost = metrics.fraud_loss_missed + metrics.false_positive_cost
    metrics.net_benefit = metrics.fraud_loss_prevented - metrics.total_expected_cost

    # PR-AUC Baseline & Lift (Audit #15)
    prevalence = float(y_true.mean()) if len(y_true) > 0 else 0.0
    metrics.random_baseline_pr_auc = prevalence
    metrics.pr_auc_lift = metrics.pr_auc / prevalence if prevalence > 0 else 0.0

    return metrics


def compute_metrics_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[list[float]] = None,
    amounts: Optional[np.ndarray] = None,
) -> list[FraudMetrics]:
    """Compute metrics at multiple thresholds for threshold optimization."""
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for t in thresholds:
        results.append(compute_fraud_metrics(y_true, y_prob, threshold=t, amounts=amounts))
    return results


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Compute 95% bootstrap confidence intervals for key metrics (Audit #14).

    Returns:
        Dict mapping metric_name → (point_estimate, ci_lower_95, ci_upper_95)
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64) if amounts is not None else None

    point_metrics = compute_fraud_metrics(y_true, y_prob, threshold=threshold, amounts=amounts)

    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    metric_samples: dict[str, list[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "p_at_100": [],
        "p_at_500": [],
        "net_benefit": [],
    }

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample_y_true = y_true[idx]
        sample_y_prob = y_prob[idx]
        sample_amounts = amounts[idx] if amounts is not None else None

        if len(np.unique(sample_y_true)) < 2:
            continue

        m = compute_fraud_metrics(
            sample_y_true, sample_y_prob, threshold=threshold, amounts=sample_amounts
        )
        metric_samples["pr_auc"].append(m.pr_auc)
        metric_samples["roc_auc"].append(m.roc_auc)
        metric_samples["precision"].append(m.precision)
        metric_samples["recall"].append(m.recall)
        metric_samples["f1"].append(m.f1)
        metric_samples["p_at_100"].append(m.precision_at_100)
        metric_samples["p_at_500"].append(m.precision_at_500)
        metric_samples["net_benefit"].append(m.net_benefit)

    ci_results = {}
    for name, samples in metric_samples.items():
        point_est = getattr(
            point_metrics,
            "precision_at_100"
            if name == "p_at_100"
            else ("precision_at_500" if name == "p_at_500" else name),
        )
        if len(samples) > 0:
            low = float(np.percentile(samples, 2.5))
            high = float(np.percentile(samples, 97.5))
        else:
            low = high = point_est
        ci_results[name] = (float(point_est), low, high)

    return ci_results
