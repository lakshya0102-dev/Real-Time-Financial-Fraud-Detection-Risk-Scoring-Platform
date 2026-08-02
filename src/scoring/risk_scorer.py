"""
Risk scorer — converts calibrated probability to risk score and level.

Maps: calibrated_probability → risk_score (0–1000) → risk_level

Levels:
  0–199:   LOW
  200–499: MEDIUM
  500–749: HIGH
  750–1000: CRITICAL
"""

from __future__ import annotations

import numpy as np

from src.config.settings import get_settings, RiskLevel, RiskScoringConfig


class RiskScorer:
    """Converts calibrated fraud probability into a risk score and level."""

    def __init__(self, config: RiskScoringConfig | None = None) -> None:
        self.config = config or get_settings().risk_scoring

    def probability_to_score(self, probability: float) -> int:
        """Convert probability [0, 1] to risk score [0, 1000].

        Uses a non-linear (sqrt) mapping that amplifies low
        probabilities to spread scores across the full range.
        Without this, a 0.4% fraud rate would cluster 99.6% of
        scores in 0-10 on a linear scale.

        Examples (sqrt mapping):
          p=0.01  → score ~100
          p=0.05  → score ~224
          p=0.10  → score ~316
          p=0.50  → score ~707
          p=1.00  → score  1000
        """
        p = np.clip(probability, 0.0, 1.0)
        # Non-linear sqrt mapping: spreads low probabilities across score range
        score = int(np.sqrt(p) * self.config.max_score)
        return int(np.clip(score, self.config.min_score, self.config.max_score))

    def score_to_level(self, score: int) -> RiskLevel:
        """Map risk score to risk level."""
        if score <= self.config.low_max:
            return RiskLevel.LOW
        elif score <= self.config.medium_max:
            return RiskLevel.MEDIUM
        elif score <= self.config.high_max:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def assess(self, probability: float) -> tuple[int, RiskLevel]:
        """Full assessment: probability → (score, level)."""
        score = self.probability_to_score(probability)
        level = self.score_to_level(score)
        return score, level

    def batch_assess(self, probabilities: np.ndarray) -> tuple[np.ndarray, list[RiskLevel]]:
        """Batch assessment for multiple probabilities."""
        scores = np.clip(
            (np.sqrt(probabilities) * self.config.max_score).astype(int),
            self.config.min_score,
            self.config.max_score,
        )
        levels = [self.score_to_level(int(s)) for s in scores]
        return scores, levels
