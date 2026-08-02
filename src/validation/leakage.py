"""
Leakage control module.

Prevents target leakage by enforcing strict separation between:
  - Training data (contains historical labels)
  - Online inference data (contains ONLY pre-decision information)

The following columns are POST-EVENT and MUST NEVER be used as predictive features:
  - true_fraud_label  (the target variable)
  - fraud_scenario    (reveals the fraud type)
  - observed_label    (delayed observation)
  - label_timestamp   (timestamp of label assignment)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class LeakageViolationError(Exception):
    """Raised when forbidden post-event columns are detected in online data."""

    def __init__(self, forbidden_columns_found: list[str]) -> None:
        self.forbidden_columns = forbidden_columns_found
        super().__init__(
            f"LEAKAGE VIOLATION: Post-event columns detected in online data: "
            f"{forbidden_columns_found}. These columns MUST NOT be used for "
            f"real-time prediction. Remove them before inference."
        )


class LeakageValidator:
    """Validates that online inference data contains no post-event leakage."""

    def __init__(self, forbidden_columns: Optional[list[str]] = None) -> None:
        settings = get_settings()
        self.forbidden_columns = set(forbidden_columns or settings.leakage.forbidden_online_columns)
        self.excluded_features = set(settings.leakage.excluded_feature_columns)
        self.id_columns = set(settings.leakage.id_columns)

    def validate_no_leakage(self, df: pd.DataFrame) -> None:
        """Check that no forbidden columns exist in the DataFrame.

        Args:
            df: DataFrame to validate.

        Raises:
            LeakageViolationError: If any forbidden columns are found.
        """
        found = [col for col in self.forbidden_columns if col in df.columns]
        if found:
            raise LeakageViolationError(found)
        logger.debug("Leakage validation passed: no forbidden columns detected")

    def validate_no_leakage_dict(self, data: dict) -> None:
        """Check that no forbidden keys exist in a dictionary (single transaction).

        Args:
            data: Dictionary of feature values.

        Raises:
            LeakageViolationError: If any forbidden keys are found.
        """
        found = [key for key in self.forbidden_columns if key in data]
        if found:
            raise LeakageViolationError(found)

    def get_safe_feature_columns(self, all_columns: list[str]) -> list[str]:
        """Return only columns that are safe to use as model features.

        Removes:
          - Forbidden post-event columns
          - Excluded feature columns (e.g., is_outlier)
          - High-cardinality ID columns (used for aggregation, not as features)

        Args:
            all_columns: List of all available columns.

        Returns:
            List of columns safe for model training/inference.
        """
        unsafe = self.forbidden_columns | self.excluded_features | self.id_columns
        safe = [col for col in all_columns if col not in unsafe]
        logger.info(
            "Safe feature columns: %d / %d (removed %d forbidden, %d excluded, %d IDs)",
            len(safe),
            len(all_columns),
            len(self.forbidden_columns & set(all_columns)),
            len(self.excluded_features & set(all_columns)),
            len(self.id_columns & set(all_columns)),
        )
        return safe

    def filter_online_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove all forbidden columns from a DataFrame.

        This is a safety measure — even if forbidden columns somehow end up
        in the data pipeline, this ensures they are stripped before inference.

        Args:
            df: DataFrame potentially containing forbidden columns.

        Returns:
            DataFrame with forbidden columns removed.
        """
        cols_to_drop = [c for c in self.forbidden_columns if c in df.columns]
        if cols_to_drop:
            logger.warning("Stripping forbidden columns from online data: %s", cols_to_drop)
            df = df.drop(columns=cols_to_drop)
        return df


# ──────────────────────────────────────────────────────
# Module-level convenience functions
# ──────────────────────────────────────────────────────

_validator: Optional[LeakageValidator] = None


def _get_validator() -> LeakageValidator:
    global _validator
    if _validator is None:
        _validator = LeakageValidator()
    return _validator


def validate_no_leakage(df: pd.DataFrame) -> None:
    """Convenience function: validate a DataFrame has no leakage.

    Raises:
        LeakageViolationError: If forbidden columns are found.
    """
    _get_validator().validate_no_leakage(df)


def filter_online_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function: strip forbidden columns from a DataFrame."""
    return _get_validator().filter_online_data(df)


def get_safe_feature_columns(all_columns: list[str]) -> list[str]:
    """Convenience function: return columns safe for model features."""
    return _get_validator().get_safe_feature_columns(all_columns)
