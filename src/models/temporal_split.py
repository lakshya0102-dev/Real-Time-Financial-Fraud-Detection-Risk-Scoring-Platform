"""
Temporal train/validation/test split.

CRITICAL: No random splitting. Uses chronological boundaries derived
from event_timestamp to prevent temporal leakage.

Split:
  TRAIN:      Jan 1 → Apr 30, 2026
  VALIDATION: May 1 → May 31, 2026
  TEST:       Jun 1 → Jun 29, 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    """Result of a temporal train/validation/test split."""

    train_idx: list[int]
    val_idx: list[int]
    val_calib_idx: list[int]
    val_opt_idx: list[int]
    test_idx: list[int]
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp

    @property
    def summary(self) -> str:
        return (
            f"Temporal Split:\n"
            f"  TRAIN:     {len(self.train_idx):,} rows (→ {self.train_end})\n"
            f"  VAL:       {len(self.val_idx):,} rows ({self.val_start} → {self.val_end})\n"
            f"    VAL_CALIB: {len(self.val_calib_idx):,} rows\n"
            f"    VAL_OPT:   {len(self.val_opt_idx):,} rows\n"
            f"  TEST:      {len(self.test_idx):,} rows ({self.test_start} →)\n"
        )


def temporal_train_val_test_split(
    df: pd.DataFrame,
    timestamp_col: str = "event_timestamp",
    config: Optional[object] = None,
) -> TemporalSplit:
    """Split dataset chronologically into train/validation/test sets.

    Args:
        df: DataFrame with timestamps.
        timestamp_col: Column containing event timestamps.
        config: Optional TemporalSplitConfig override.

    Returns:
        TemporalSplit with indices for each set.
    """
    if config is None:
        config = get_settings().temporal_split

    ts = pd.to_datetime(df[timestamp_col])

    train_end = pd.Timestamp(config.train_end)
    val_start = pd.Timestamp(config.validation_start)
    val_calib_end = pd.Timestamp(getattr(config, "val_calib_end", "2026-05-15T23:59:59"))
    val_opt_start = pd.Timestamp(getattr(config, "val_opt_start", "2026-05-16T00:00:00"))
    val_end = pd.Timestamp(config.validation_end)
    test_start = pd.Timestamp(config.test_start)

    train_mask = ts <= train_end
    val_mask = (ts >= val_start) & (ts <= val_end)
    val_calib_mask = (ts >= val_start) & (ts <= val_calib_end)
    val_opt_mask = (ts >= val_opt_start) & (ts <= val_end)
    test_mask = ts >= test_start

    split = TemporalSplit(
        train_idx=df.index[train_mask].tolist(),
        val_idx=df.index[val_mask].tolist(),
        val_calib_idx=df.index[val_calib_mask].tolist(),
        val_opt_idx=df.index[val_opt_mask].tolist(),
        test_idx=df.index[test_mask].tolist(),
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
    )

    logger.info(split.summary)

    # Verify no overlap across primary splits & sub-splits
    train_set = set(split.train_idx)
    val_calib_set = set(split.val_calib_idx)
    val_opt_set = set(split.val_opt_idx)
    test_set = set(split.test_idx)

    assert train_set.isdisjoint(val_calib_set), "Train and val_calib sets overlap!"
    assert train_set.isdisjoint(val_opt_set), "Train and val_opt sets overlap!"
    assert train_set.isdisjoint(test_set), "Train and test sets overlap!"
    assert val_calib_set.isdisjoint(val_opt_set), "val_calib and val_opt sets overlap!"
    assert val_calib_set.isdisjoint(test_set), "val_calib and test sets overlap!"
    assert val_opt_set.isdisjoint(test_set), "val_opt and test sets overlap!"

    # Log fraud distribution per split
    if "true_fraud_label" in df.columns:
        for name, idx in [
            ("TRAIN", split.train_idx),
            ("VAL_CALIB", split.val_calib_idx),
            ("VAL_OPT", split.val_opt_idx),
            ("TEST", split.test_idx),
        ]:
            subset = df.loc[idx, "true_fraud_label"]
            fraud_count = subset.sum()
            fraud_rate = fraud_count / len(subset) * 100 if len(subset) > 0 else 0
            logger.info(
                "  %s fraud: %d / %d (%.3f%%)",
                name,
                fraud_count,
                len(subset),
                fraud_rate,
            )

    return split
