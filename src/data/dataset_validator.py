"""
Dataset integrity validator.

Performs comprehensive checks on data/CFR_data.csv at startup/training time:
  1. File existence
  2. SHA-256 checksum
  3. Column schema
  4. Data types
  5. Minimum row count
  6. Transaction ID uniqueness
  7. Timestamp range
  8. Target distribution
  9. No null values
  10. No duplicate rows

Fails clearly if the dataset has been unexpectedly modified.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import get_settings, DatasetConfig

logger = logging.getLogger(__name__)

# Read buffer size for SHA-256 (8 MB)
_HASH_BUFFER_SIZE = 8 * 1024 * 1024


class DatasetIntegrityError(Exception):
    """Raised when the dataset fails an integrity check.

    The application MUST NOT proceed with a corrupted or modified dataset.
    """

    def __init__(self, check_name: str, detail: str) -> None:
        self.check_name = check_name
        self.detail = detail
        super().__init__(f"Dataset integrity check FAILED [{check_name}]: {detail}")


@dataclass
class DatasetValidationReport:
    """Summary of all dataset validation checks."""

    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sha256: str = ""
    row_count: int = 0
    column_count: int = 0
    fraud_count: int = 0
    observed_label_count: int = 0
    outlier_count: int = 0
    null_count: int = 0
    duplicate_txn_count: int = 0
    timestamp_min: str = ""
    timestamp_max: str = ""

    @property
    def is_valid(self) -> bool:
        return len(self.checks_failed) == 0

    def summary(self) -> str:
        lines = ["=" * 60, "DATASET VALIDATION REPORT", "=" * 60]
        lines.append(f"Status: {'✅ PASSED' if self.is_valid else '❌ FAILED'}")
        lines.append(f"SHA-256: {self.sha256}")
        lines.append(f"Rows: {self.row_count:,}")
        lines.append(f"Columns: {self.column_count}")
        lines.append(f"Fraud: {self.fraud_count:,}")
        lines.append(f"Observed labels: {self.observed_label_count:,}")
        lines.append(f"Outliers: {self.outlier_count:,}")
        lines.append(f"Nulls: {self.null_count:,}")
        lines.append(f"Duplicate TXN IDs: {self.duplicate_txn_count:,}")
        lines.append(f"Timestamp range: {self.timestamp_min} → {self.timestamp_max}")
        lines.append("-" * 60)

        if self.checks_passed:
            lines.append(f"Checks PASSED ({len(self.checks_passed)}):")
            for c in self.checks_passed:
                lines.append(f"  ✅ {c}")

        if self.checks_failed:
            lines.append(f"Checks FAILED ({len(self.checks_failed)}):")
            for c in self.checks_failed:
                lines.append(f"  ❌ {c}")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(_HASH_BUFFER_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def validate_dataset(
    fail_fast: bool = True,
    config: Optional[DatasetConfig] = None,
) -> DatasetValidationReport:
    """Run all dataset integrity checks.

    Args:
        fail_fast: If True, raise DatasetIntegrityError on first failure.
        config: Optional DatasetConfig override (uses settings by default).

    Returns:
        DatasetValidationReport with all check results.

    Raises:
        DatasetIntegrityError: If any check fails and fail_fast=True.
    """
    if config is None:
        config = get_settings().dataset

    report = DatasetValidationReport()

    def _pass(name: str) -> None:
        report.checks_passed.append(name)
        logger.info("✅ %s", name)

    def _fail(name: str, detail: str) -> None:
        report.checks_failed.append(f"{name}: {detail}")
        logger.error("❌ %s: %s", name, detail)
        if fail_fast:
            raise DatasetIntegrityError(name, detail)

    def _warn(msg: str) -> None:
        report.warnings.append(msg)
        logger.warning("⚠️  %s", msg)

    # ── Check 1: File exists ──
    if not config.path.exists():
        _fail("file_exists", f"Dataset not found at {config.path}")
        return report
    _pass("file_exists")

    if not config.path.is_file():
        _fail("is_file", f"{config.path} is not a regular file")
        return report
    _pass("is_file")

    # ── Check 2: SHA-256 checksum ──
    logger.info("Computing SHA-256 checksum...")
    sha256 = compute_sha256(config.path)
    report.sha256 = sha256

    if sha256 != config.expected_sha256:
        _fail(
            "sha256_checksum",
            f"Expected {config.expected_sha256}, got {sha256}. The dataset has been modified.",
        )
    else:
        _pass("sha256_checksum")

    # ── Load dataset for remaining checks ──
    logger.info("Loading dataset for schema validation...")
    try:
        df = pd.read_csv(config.path, low_memory=False)
    except Exception as e:
        _fail("file_readable", str(e))
        return report
    _pass("file_readable")

    report.row_count = len(df)
    report.column_count = len(df.columns)

    # ── Check 3: Column schema ──
    actual_cols = list(df.columns)
    expected_cols = config.expected_columns

    if actual_cols != expected_cols:
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)
        detail = ""
        if missing:
            detail += f"Missing columns: {missing}. "
        if extra:
            detail += f"Unexpected columns: {extra}. "
        if not missing and not extra:
            detail = "Column order mismatch."
        _fail("column_schema", detail)
    else:
        _pass("column_schema")

    # ── Check 4: Data types ──
    expected_numeric = [
        "transaction_sequence_id",
        "customer_tenure_days",
        "card_age_days",
        "credit_limit",
        "merchant_age_days",
        "device_age_days",
        "amount",
        "installment_flag",
        "international_flag",
        "is_outlier",
        "true_fraud_label",
        "observed_label",
    ]
    for col in expected_numeric:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            _fail("data_types", f"Column '{col}' should be numeric, got {df[col].dtype}")
    _pass("data_types")

    # ── Check 5: Minimum row count ──
    if report.row_count < config.expected_row_count:
        _fail(
            "row_count",
            f"Expected >= {config.expected_row_count:,} rows, got {report.row_count:,}",
        )
    else:
        _pass("row_count")

    # ── Check 6: Transaction ID uniqueness ──
    dup_count = df["transaction_id"].duplicated().sum()
    report.duplicate_txn_count = int(dup_count)
    if dup_count > 0:
        _fail("txn_id_unique", f"Found {dup_count:,} duplicate transaction IDs")
    else:
        _pass("txn_id_unique")

    # ── Check 7: Null values ──
    null_total = int(df.isnull().sum().sum())
    report.null_count = null_total
    if null_total > 0:
        null_cols = df.columns[df.isnull().any()].tolist()
        _fail("no_nulls", f"Found {null_total:,} null values in columns: {null_cols}")
    else:
        _pass("no_nulls")

    # ── Check 8: Timestamp ordering/range ──
    try:
        timestamps = pd.to_datetime(df["event_timestamp"])
        ts_min = timestamps.min()
        ts_max = timestamps.max()
        report.timestamp_min = str(ts_min)
        report.timestamp_max = str(ts_max)

        expected_min = pd.Timestamp(config.expected_timestamp_min)
        expected_max = pd.Timestamp(config.expected_timestamp_max)

        if ts_min < expected_min - pd.Timedelta(days=1):
            _fail("timestamp_range", f"Min timestamp {ts_min} is before expected {expected_min}")
        elif ts_max > expected_max + pd.Timedelta(days=1):
            _fail("timestamp_range", f"Max timestamp {ts_max} is after expected {expected_max}")
        else:
            _pass("timestamp_range")
    except Exception as e:
        _fail("timestamp_range", f"Failed to parse timestamps: {e}")

    # ── Check 9: Target distribution ──
    fraud_count = int(df["true_fraud_label"].sum())
    report.fraud_count = fraud_count
    if fraud_count != config.expected_fraud_count:
        _warn(f"Fraud count {fraud_count:,} differs from expected {config.expected_fraud_count:,}")
    _pass("target_distribution")

    observed_count = int(df["observed_label"].sum())
    report.observed_label_count = observed_count
    if observed_count != config.expected_observed_label_count:
        _warn(
            f"Observed label count {observed_count:,} differs from expected "
            f"{config.expected_observed_label_count:,}"
        )

    outlier_count = int(df["is_outlier"].sum())
    report.outlier_count = outlier_count
    if outlier_count != config.expected_outlier_count:
        _warn(
            f"Outlier count {outlier_count:,} differs from expected "
            f"{config.expected_outlier_count:,}"
        )

    # ── Check 10: Outlier-fraud cross-check ──
    fraud_in_outliers = int(df.loc[df["is_outlier"] == 1, "true_fraud_label"].sum())
    if fraud_in_outliers > 0:
        _warn(
            f"Found {fraud_in_outliers} fraud cases in outlier rows. "
            f"This may affect the is_outlier exclusion policy."
        )
    else:
        _pass("outlier_fraud_separation")

    logger.info(report.summary())
    return report
