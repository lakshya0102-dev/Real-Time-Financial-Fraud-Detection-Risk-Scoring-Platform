"""
Dataset loader — single source of truth for loading CFR_data.csv.

This module enforces the ABSOLUTE DATASET RULE:
  - There is exactly ONE dataset: data/CFR_data.csv
  - If it doesn't exist → DatasetNotFoundError (never search for alternatives)
  - If it can't be read → fail fast
  - The raw dataset is NEVER modified
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the canonical dataset is not found at the expected path.

    The application MUST NOT:
      - Search for alternative CSV files
      - Download a replacement dataset
      - Generate synthetic data as a fallback
      - Silently switch to any other data source
    """

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"FATAL: Canonical dataset not found at '{path}'. "
            f"The application requires exactly this file. "
            f"DO NOT substitute another dataset. "
            f"Place the dataset at the expected path and retry."
        )


def load_dataset(
    columns: list[str] | None = None,
    nrows: int | None = None,
    parse_dates: bool = True,
    dtype_backend: str = "pyarrow",
) -> pd.DataFrame:
    """Load the canonical dataset from data/CFR_data.csv.

    This is the ONLY function that should ever load the raw dataset.
    All training, evaluation, replay, and development workflows must use this.

    Args:
        columns: Specific columns to load (None = all).
        nrows: Number of rows to load (None = all).
        parse_dates: Whether to parse event_timestamp as datetime.
        dtype_backend: Backend for dtype inference ('pyarrow' for efficiency).

    Returns:
        DataFrame containing the requested data.

    Raises:
        DatasetNotFoundError: If the canonical file does not exist.
        pd.errors.ParserError: If the file is corrupted or unreadable.
    """
    settings = get_settings()
    dataset_path = settings.dataset.path

    if not dataset_path.exists():
        raise DatasetNotFoundError(dataset_path)

    if not dataset_path.is_file():
        raise DatasetNotFoundError(dataset_path)

    logger.info("Loading dataset from %s", dataset_path)

    date_columns = ["event_timestamp", "label_timestamp"] if parse_dates else None

    # Filter date_columns to only include requested columns
    if date_columns and columns:
        date_columns = [c for c in date_columns if c in columns]
        if not date_columns:
            date_columns = None

    try:
        df = pd.read_csv(
            dataset_path,
            usecols=columns,
            nrows=nrows,
            parse_dates=date_columns,
            low_memory=False,
        )
    except Exception as e:
        logger.error("Failed to read dataset at %s: %s", dataset_path, e)
        raise

    logger.info(
        "Loaded dataset: %d rows, %d columns",
        len(df),
        len(df.columns),
    )

    return df


def get_dataset_path() -> Path:
    """Return the canonical dataset path (for hashing, validation, etc.)."""
    settings = get_settings()
    path = settings.dataset.path
    if not path.exists():
        raise DatasetNotFoundError(path)
    return path
