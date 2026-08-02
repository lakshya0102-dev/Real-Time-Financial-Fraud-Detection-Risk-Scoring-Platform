"""
Cost-sensitive decision engine.

Makes APPROVE/REVIEW/BLOCK decisions based on configurable thresholds
and explicit cost function optimization.

Does NOT simply use probability > 0.5. Instead optimizes against:
  - fraud_loss_prevented
  - false_positive_cost
  - manual_review_cost
  - customer_friction_cost
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config.settings import (
    CostConfig,
    Decision,
    RiskScoringConfig,
    get_settings,
)
from src.scoring.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)


@dataclass
class TransactionDecision:
    """Result of the decision engine for a single transaction."""

    transaction_id: str
    fraud_probability: float
    risk_score: int
    risk_level: str
    decision: Decision
    explanations: list[str]


class DecisionEngine:
    """Cost-sensitive decision engine for fraud transactions.

    Makes three-tier decisions:
      APPROVE:  risk < threshold_low → pass transaction
      REVIEW:   threshold_low ≤ risk < threshold_high → manual review
      BLOCK:    risk ≥ threshold_high → block transaction
    """

    def __init__(
        self,
        risk_config: Optional[RiskScoringConfig] = None,
        cost_config: Optional[CostConfig] = None,
    ) -> None:
        settings = get_settings()
        self.risk_config = risk_config or settings.risk_scoring
        self.cost_config = cost_config or settings.cost
        self.risk_scorer = RiskScorer(self.risk_config)

    def decide(
        self,
        transaction_id: str,
        fraud_probability: float,
        explanations: Optional[list[str]] = None,
    ) -> TransactionDecision:
        """Make a decision for a single transaction.

        Args:
            transaction_id: Transaction identifier.
            fraud_probability: Calibrated fraud probability.
            explanations: Optional SHAP-based explanations.

        Returns:
            TransactionDecision with score, level, and decision.
        """
        score, level = self.risk_scorer.assess(fraud_probability)

        if score <= self.risk_config.approve_max:
            decision = Decision.APPROVE
        elif score <= self.risk_config.review_max:
            decision = Decision.REVIEW
        else:
            decision = Decision.BLOCK

        return TransactionDecision(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            risk_score=score,
            risk_level=level.value,
            decision=decision,
            explanations=explanations or [],
        )

    def batch_decide(
        self,
        transaction_ids: list[str],
        fraud_probabilities: np.ndarray,
    ) -> list[TransactionDecision]:
        """Make decisions for a batch of transactions."""
        return [
            self.decide(tid, float(prob)) for tid, prob in zip(transaction_ids, fraud_probabilities)
        ]


@dataclass
class CostAnalysis:
    """Result of cost function analysis at a given set of thresholds."""

    approve_threshold: float
    review_threshold: float
    fraud_caught: int
    fraud_missed: int
    false_declines: int
    manual_reviews: int
    fraud_loss_prevented: float
    fraud_loss_missed: float
    false_positive_cost: float
    review_cost: float
    total_cost: float
    net_benefit: float


class ThresholdOptimizer:
    """Optimizes APPROVE/REVIEW/BLOCK thresholds against cost function."""

    def __init__(self, cost_config: Optional[CostConfig] = None) -> None:
        self.cost_config = cost_config or get_settings().cost

    def compute_cost(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        approve_threshold: float,
        block_threshold: float,
        amounts: Optional[np.ndarray] = None,
    ) -> CostAnalysis:
        """Compute business cost at given thresholds.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            approve_threshold: Below this → APPROVE.
            block_threshold: Above this → BLOCK. Between → REVIEW.
            amounts: Transaction amounts.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        approve_mask = y_prob < approve_threshold
        review_mask = (y_prob >= approve_threshold) & (y_prob < block_threshold)
        block_mask = y_prob >= block_threshold

        # Fraud outcomes
        fraud_caught = int(((block_mask | review_mask) & (y_true == 1)).sum())
        fraud_missed = int((approve_mask & (y_true == 1)).sum())
        false_declines = int((block_mask & (y_true == 0)).sum())
        manual_reviews = int(review_mask.sum())

        # Cost calculation
        if amounts is not None:
            amounts = np.asarray(amounts)
            fraud_loss_prevented = float(amounts[(block_mask | review_mask) & (y_true == 1)].sum())
            fraud_loss_missed = float(amounts[approve_mask & (y_true == 1)].sum())
        else:
            fraud_loss_prevented = fraud_caught * self.cost_config.avg_fraud_loss
            fraud_loss_missed = fraud_missed * self.cost_config.avg_fraud_loss

        fp_cost = false_declines * self.cost_config.false_positive_cost
        review_cost = manual_reviews * self.cost_config.manual_review_cost

        total_cost = fraud_loss_missed + fp_cost + review_cost
        net_benefit = fraud_loss_prevented - total_cost

        return CostAnalysis(
            approve_threshold=approve_threshold,
            review_threshold=block_threshold,
            fraud_caught=fraud_caught,
            fraud_missed=fraud_missed,
            false_declines=false_declines,
            manual_reviews=manual_reviews,
            fraud_loss_prevented=fraud_loss_prevented,
            fraud_loss_missed=fraud_loss_missed,
            false_positive_cost=fp_cost,
            review_cost=review_cost,
            total_cost=total_cost,
            net_benefit=net_benefit,
        )

    def optimize(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        amounts: Optional[np.ndarray] = None,
        n_steps: int = 50,
    ) -> tuple[CostAnalysis, float, float]:
        """Find optimal APPROVE/BLOCK thresholds by maximizing net benefit.

        Returns:
            (best_analysis, best_approve_threshold, best_block_threshold)
        """
        best_analysis = None
        best_benefit = float("-inf")
        best_at = 0.0
        best_bt = 0.5

        approve_thresholds = np.linspace(0.01, 0.5, n_steps)
        block_thresholds = np.linspace(0.1, 0.95, n_steps)

        for at in approve_thresholds:
            for bt in block_thresholds:
                if bt <= at:
                    continue

                analysis = self.compute_cost(y_true, y_prob, at, bt, amounts)
                if analysis.net_benefit > best_benefit:
                    best_benefit = analysis.net_benefit
                    best_analysis = analysis
                    best_at = at
                    best_bt = bt

        if best_analysis is None:
            raise ValueError(
                "ThresholdOptimizer found no valid threshold pair. "
                "Check that approve_thresholds < block_thresholds."
            )

        logger.info(
            "Optimal thresholds: APPROVE < %.4f, BLOCK >= %.4f, Net benefit: $%.2f",
            best_at,
            best_bt,
            best_benefit,
        )

        return best_analysis, best_at, best_bt
