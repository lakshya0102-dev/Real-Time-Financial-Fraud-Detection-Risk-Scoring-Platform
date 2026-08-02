"""
Automated retraining pipeline.

Pipeline:
  new labeled data → data validation → feature generation →
  temporal split → training → calibration → evaluation →
  cost optimization → drift evaluation → model validation →
  MLflow registry → promotion decision

A failed model MUST NEVER replace the production model.
Always retains the last known-good model.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config.settings import get_settings, PromotionGateConfig
from src.models.metrics import FraudMetrics

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    """Result of model promotion gate evaluation."""

    approved: bool
    model_version: str
    metrics: FraudMetrics
    checks_passed: list[str]
    checks_failed: list[str]
    reason: str


class ModelPromotionValidator:
    """Validates whether a candidate model can be promoted to production.

    Requires meeting ALL gates:
      - Minimum PR-AUC
      - Minimum recall
      - Maximum false-positive rate
      - Maximum expected cost
      - Calibration requirement
      - No severe feature drift
      - No data leakage
    """

    def __init__(self, config: Optional[PromotionGateConfig] = None) -> None:
        self.config = config or get_settings().promotion_gates

    def evaluate(
        self,
        model_version: str,
        metrics: FraudMetrics,
        has_calibration: bool = True,
        has_drift: bool = False,
        has_leakage: bool = False,
    ) -> PromotionDecision:
        """Evaluate whether a model passes all promotion gates."""
        passed = []
        failed = []

        # PR-AUC gate
        if metrics.pr_auc >= self.config.min_pr_auc:
            passed.append(f"PR-AUC {metrics.pr_auc:.4f} >= {self.config.min_pr_auc}")
        else:
            failed.append(f"PR-AUC {metrics.pr_auc:.4f} < {self.config.min_pr_auc}")

        # Recall gate
        if metrics.recall >= self.config.min_recall:
            passed.append(f"Recall {metrics.recall:.4f} >= {self.config.min_recall}")
        else:
            failed.append(f"Recall {metrics.recall:.4f} < {self.config.min_recall}")

        # FPR gate
        if metrics.false_positive_rate <= self.config.max_fpr:
            passed.append(f"FPR {metrics.false_positive_rate:.4f} <= {self.config.max_fpr}")
        else:
            failed.append(f"FPR {metrics.false_positive_rate:.4f} > {self.config.max_fpr}")

        # Brier score gate
        if metrics.brier_score <= self.config.max_brier_score:
            passed.append(f"Brier {metrics.brier_score:.6f} <= {self.config.max_brier_score}")
        else:
            failed.append(f"Brier {metrics.brier_score:.6f} > {self.config.max_brier_score}")

        # Cost gate
        if metrics.total_expected_cost <= self.config.max_expected_cost:
            passed.append(
                f"Cost ${metrics.total_expected_cost:,.0f} <= ${self.config.max_expected_cost:,.0f}"
            )
        else:
            failed.append(
                f"Cost ${metrics.total_expected_cost:,.0f} > ${self.config.max_expected_cost:,.0f}"
            )

        # Calibration gate
        if self.config.require_calibration:
            if has_calibration:
                passed.append("Calibration present")
            else:
                failed.append("Calibration required but not present")

        # Drift gate
        if self.config.require_no_drift:
            if not has_drift:
                passed.append("No severe drift detected")
            else:
                failed.append("Severe drift detected")

        # Leakage gate
        if self.config.require_no_leakage:
            if not has_leakage:
                passed.append("No data leakage")
            else:
                failed.append("DATA LEAKAGE DETECTED")

        approved = len(failed) == 0

        reason = "APPROVED" if approved else f"REJECTED: {'; '.join(failed)}"

        decision = PromotionDecision(
            approved=approved,
            model_version=model_version,
            metrics=metrics,
            checks_passed=passed,
            checks_failed=failed,
            reason=reason,
        )

        if approved:
            logger.info("✅ Model %s APPROVED for promotion", model_version)
            for p in passed:
                logger.info("  ✅ %s", p)
        else:
            logger.warning("❌ Model %s REJECTED for promotion", model_version)
            for f in failed:
                logger.warning("  ❌ %s", f)
            for p in passed:
                logger.info("  ✅ %s", p)

        return decision


class RetrainingPipeline:
    """Orchestrates the full retraining pipeline.

    Ensures a failed model never replaces the production model.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.validator = ModelPromotionValidator()
        self.artifacts_dir = self.settings.project_root / "artifacts" / "models"

    def should_retrain(
        self,
        current_metrics: FraudMetrics,
        baseline_metrics: FraudMetrics,
        drift_detected: bool = False,
    ) -> bool:
        """Determine if retraining should be triggered.

        Does NOT retrain blindly on every drift signal.
        """
        # Only retrain if:
        # 1. Significant performance degradation
        pr_auc_drop = baseline_metrics.pr_auc - current_metrics.pr_auc
        recall_drop = baseline_metrics.recall - current_metrics.recall

        if pr_auc_drop > 0.05:
            logger.info("PR-AUC dropped by %.4f, recommending retraining", pr_auc_drop)
            return True

        if recall_drop > 0.1:
            logger.info("Recall dropped by %.4f, recommending retraining", recall_drop)
            return True

        # 2. Drift + performance degradation (not drift alone)
        if drift_detected and (pr_auc_drop > 0.02 or recall_drop > 0.05):
            logger.info("Drift detected + performance degradation, recommending retraining")
            return True

        logger.info("No retraining needed")
        return False

    def backup_current_model(self) -> None:
        """Backup the current production model before replacement."""
        production_path = self.artifacts_dir / "production_model.pkl"
        backup_path = self.artifacts_dir / "production_model_backup.pkl"

        if production_path.exists():
            import shutil

            shutil.copy2(production_path, backup_path)
            logger.info("Backed up production model to %s", backup_path)

    def rollback(self) -> None:
        """Rollback to the previous production model."""
        backup_path = self.artifacts_dir / "production_model_backup.pkl"
        production_path = self.artifacts_dir / "production_model.pkl"

        if backup_path.exists():
            import shutil

            shutil.copy2(backup_path, production_path)
            logger.info("Rolled back to backup model")
        else:
            logger.error("No backup model found for rollback!")
