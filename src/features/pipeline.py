"""
Feature engineering pipeline — orchestrates all feature groups.

Supports two modes:
  - Offline (batch): Process entire DataFrame for training
  - Online (single): Process one transaction with feature store context

All feature calculations are deterministic and respect temporal ordering.

LEAKAGE PREVENTION:
  - Categorical frequency encoding is FIT on training data only, then APPLIED to val/test.
  - Velocity features use only strictly-prior transactions.
  - Expanding statistics exclude the current row.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Production feature engineering pipeline.

    Generates all feature categories:
      A. Static transaction features
      B. Entity aggregate features
      C. Temporal features
      D. Velocity features
      E. Geographic consistency features
      F. Identity/device risk features
      G. Amount features

    IMPORTANT: Call fit_categorical_maps(train_df) before transforming any data.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._categorical_freq_maps: dict[str, dict] = {}
        self._is_fitted: bool = False

    CATEGORICAL_COLS: list[str] = [
        "account_type",
        "card_type",
        "merchant_category",
        "merchant_size",
        "transaction_type",
        "payment_channel",
        "payment_method",
        "authentication_method",
        "connection_type",
        "proxy_type",
        "device_type",
        "device_os",
        "card_status",
        "device_trust_level",
        "currency",
        "settlement_type",
        "processing_route",
        "customer_risk_segment",
        "merchant_risk_segment",
        "network_risk_segment",
    ]

    # ──────────────────────────────────────────────────
    # A. Static Transaction Features
    # ──────────────────────────────────────────────────

    @staticmethod
    def generate_static_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate features directly from transaction fields."""
        result = pd.DataFrame(index=df.index)

        # Amount features
        result["log_amount"] = np.log1p(df["amount"].clip(lower=0))
        result["amount_credit_ratio"] = (df["amount"] / df["credit_limit"].replace(0, 1)).clip(
            0, 100
        )

        # Flags
        result["is_international"] = df["international_flag"].astype(np.int8)
        result["is_installment"] = df["installment_flag"].astype(np.int8)

        # Amount buckets
        result["amount_bucket"] = (
            pd.cut(
                df["amount"],
                bins=[-np.inf, 1, 10, 50, 100, 500, 1000, 5000, np.inf],
                labels=[0, 1, 2, 3, 4, 5, 6, 7],
            )
            .astype(float)
            .fillna(0)
            .astype(np.int8)
        )

        result["is_micro_txn"] = (df["amount"] < 1.0).astype(np.int8)
        result["is_high_value_txn"] = (df["amount"] > 5000).astype(np.int8)

        # Tenure and age features
        result["customer_tenure_bucket"] = (
            pd.cut(
                df["customer_tenure_days"],
                bins=[-1, 30, 90, 365, 730, np.inf],
                labels=[0, 1, 2, 3, 4],
            )
            .astype(float)
            .fillna(0)
            .astype(np.int8)
        )

        result["card_age_bucket"] = (
            pd.cut(
                df["card_age_days"],
                bins=[-1, 30, 90, 365, 730, np.inf],
                labels=[0, 1, 2, 3, 4],
            )
            .astype(float)
            .fillna(0)
            .astype(np.int8)
        )

        result["device_age_bucket"] = (
            pd.cut(
                df["device_age_days"],
                bins=[-1, 7, 30, 90, 365, np.inf],
                labels=[0, 1, 2, 3, 4],
            )
            .astype(float)
            .fillna(0)
            .astype(np.int8)
        )

        return result

    # ──────────────────────────────────────────────────
    # B. Categorical Encoding (FIT/TRANSFORM pattern)
    # ──────────────────────────────────────────────────

    def fit_categorical_maps(self, train_df: pd.DataFrame) -> None:
        """Fit frequency encoding maps on TRAINING data only.

        MUST be called before generate_categorical_features() on any split.
        This prevents val/test distribution leakage.
        """
        self._categorical_freq_maps = {}
        for col in self.CATEGORICAL_COLS:
            if col in train_df.columns:
                self._categorical_freq_maps[col] = (
                    train_df[col].value_counts(normalize=True).to_dict()
                )
        self._is_fitted = True
        logger.info(
            "Fitted categorical frequency maps on %d columns from %d training rows",
            len(self._categorical_freq_maps),
            len(train_df),
        )

    def generate_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Frequency-encode categorical columns using pre-fitted maps.

        If not fitted, falls back to computing frequencies from the passed
        DataFrame (for backward compatibility / online mode), but logs a warning.
        """
        result = pd.DataFrame(index=df.index)

        if not self._is_fitted:
            logger.warning(
                "Categorical maps not fitted — computing from passed DataFrame. "
                "This is acceptable for online inference but NOT for training/evaluation."
            )

        for col in self.CATEGORICAL_COLS:
            if col not in df.columns:
                continue

            if self._is_fitted and col in self._categorical_freq_maps:
                # Use pre-fitted map — unseen categories get 0.0
                freq_map = self._categorical_freq_maps[col]
                result[f"{col}_encoded"] = df[col].map(freq_map).fillna(0).astype(np.float32)
            else:
                # Fallback: compute from this DataFrame
                freq = df[col].value_counts(normalize=True)
                result[f"{col}_encoded"] = df[col].map(freq).fillna(0).astype(np.float32)

        return result

    # ──────────────────────────────────────────────────
    # C. Temporal Features
    # ──────────────────────────────────────────────────

    @staticmethod
    def generate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from event_timestamp."""
        result = pd.DataFrame(index=df.index)

        ts = pd.to_datetime(df["event_timestamp"])

        result["hour_of_day"] = ts.dt.hour.astype(np.int8)
        result["minute_bucket"] = (ts.dt.hour * 4 + ts.dt.minute // 15).astype(np.int8)
        result["day_of_week"] = ts.dt.dayofweek.astype(np.int8)
        result["day_of_month"] = ts.dt.day.astype(np.int8)
        result["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(np.int8)
        result["is_night"] = ts.dt.hour.isin(list(range(0, 7)) + [22, 23]).astype(np.int8)

        return result

    # ──────────────────────────────────────────────────
    # D. Velocity Features (offline batch mode)
    # ──────────────────────────────────────────────────

    @staticmethod
    def generate_velocity_features(
        df: pd.DataFrame,
        entity_col: str,
        timestamp_col: str = "event_timestamp",
        windows: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Generate velocity features for an entity using temporal-safe rolling windows.

        CRITICAL: Each row only uses data from BEFORE its timestamp (no future leakage).

        Args:
            df: DataFrame sorted by timestamp.
            entity_col: Column to group by (e.g., 'customer_id').
            timestamp_col: Timestamp column.
            windows: Rolling window sizes (e.g., ['5min', '1h', '24h']).

        Returns:
            DataFrame with velocity features for this entity.
        """
        if windows is None:
            windows = ["5min", "15min", "1h", "6h", "24h", "7D"]

        result = pd.DataFrame(index=df.index)
        entity_short = entity_col.replace("_id", "")

        ts = pd.to_datetime(df[timestamp_col])

        for window in windows:
            window_label = window.replace("min", "m").replace("h", "h").replace("D", "d")

            # Transaction count in window
            count_col = f"{entity_short}_txn_count_{window_label}"
            amount_col = f"{entity_short}_amount_sum_{window_label}"

            # Group by entity for efficient processing
            grouped = df.groupby(entity_col)
            window_td = pd.Timedelta(window)

            # For each entity group, compute rolling counts/sums
            count_series = pd.Series(0, index=df.index, dtype=np.int32)
            amount_series = pd.Series(0.0, index=df.index, dtype=np.float64)

            for entity_val, group in grouped:
                group_ts = ts.loc[group.index]
                group_amounts = df.loc[group.index, "amount"]

                for i, (idx, t) in enumerate(group_ts.items()):
                    # Only look at PREVIOUS transactions (strict temporal ordering)
                    prev = group_ts.iloc[:i]
                    mask = prev >= (t - window_td)
                    count_series.at[idx] = mask.sum()
                    if mask.sum() > 0:
                        prev_amounts = group_amounts.iloc[:i]
                        amount_series.at[idx] = float(prev_amounts.loc[mask.index[mask]].sum())

            result[count_col] = count_series
            result[amount_col] = amount_series

        return result

    @staticmethod
    def generate_velocity_features_fast(
        df: pd.DataFrame,
        entity_col: str,
        timestamp_col: str = "event_timestamp",
    ) -> pd.DataFrame:
        """Fast approximate velocity features using pandas groupby + cumcount.

        Uses expanding windows per entity (temporal-safe since data is sorted).
        Much faster than exact rolling windows for large datasets.

        CRITICAL FIX: expanding().std() and cumulative mean now exclude the
        current row to prevent self-leakage.
        """
        result = pd.DataFrame(index=df.index)
        entity_short = entity_col.replace("_id", "")

        # Ensure sorted by timestamp
        ts = pd.to_datetime(df[timestamp_col])

        # Cumulative transaction count per entity (temporal-safe: count of PRIOR txns)
        result[f"{entity_short}_total_txn_count"] = (
            df.groupby(entity_col).cumcount().astype(np.int32)
        )

        # Cumulative average amount per entity (excluding current row)
        cum_sum = df.groupby(entity_col)["amount"].cumsum() - df["amount"]
        cum_count = df.groupby(entity_col).cumcount()
        result[f"{entity_short}_avg_amount"] = (
            (cum_sum / cum_count.replace(0, 1)).fillna(0).astype(np.float32)
        )

        # Cumulative std deviation (excluding current row via shift)
        # We use shift(1) on the expanding std to exclude current row
        entity_std = (
            df.groupby(entity_col)["amount"].expanding().std().reset_index(level=0, drop=True)
        )
        # Shift within each group so row i gets std of rows [0..i-1]
        result[f"{entity_short}_std_amount"] = (
            df.groupby(entity_col)["amount"]
            .transform(lambda x: x.expanding().std().shift(1))
            .fillna(0)
            .astype(np.float32)
        )

        # Time since previous transaction for this entity
        prev_ts = df.groupby(entity_col)[timestamp_col].shift(1)
        prev_ts = pd.to_datetime(prev_ts)
        result[f"time_since_prev_{entity_short}_txn_seconds"] = (
            (ts - prev_ts).dt.total_seconds().fillna(-1).astype(np.float64)
        )

        # Amount deviation from entity mean (using prior-only mean and std)
        result[f"amount_zscore_{entity_short}"] = (
            (
                (df["amount"] - result[f"{entity_short}_avg_amount"])
                / result[f"{entity_short}_std_amount"].replace(0, 1)
            )
            .fillna(0)
            .clip(-10, 10)
            .astype(np.float32)
        )

        return result

    # ──────────────────────────────────────────────────
    # E. Geographic Consistency Features
    # ──────────────────────────────────────────────────

    @staticmethod
    def generate_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate country-mismatch and cross-border features."""
        result = pd.DataFrame(index=df.index)

        txn_country = df["transaction_country"]

        # Country mismatches
        pairs = {
            "customer": "customer_country",
            "billing": "billing_country",
            "card": "card_country",
            "device": "device_country",
            "ip": "ip_country",
            "merchant": "merchant_country",
        }

        mismatch_count = pd.Series(0, index=df.index, dtype=np.int8)
        for prefix, col in pairs.items():
            if col in df.columns:
                mismatch = (df[col] != txn_country).astype(np.int8)
                result[f"{prefix}_txn_country_mismatch"] = mismatch
                mismatch_count += mismatch

        result["geo_mismatch_count"] = mismatch_count

        # Cross-border: customer_country != transaction_country
        result["cross_border_flag"] = (df["customer_country"] != txn_country).astype(np.int8)

        return result

    # ──────────────────────────────────────────────────
    # F. Identity / Device Risk Features
    # ──────────────────────────────────────────────────

    @staticmethod
    def generate_identity_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate device/identity risk features."""
        result = pd.DataFrame(index=df.index)

        # New device
        result["is_new_device"] = (df["device_age_days"] < 7).astype(np.int8)

        # Proxy/TOR/datacenter
        proxy = df["proxy_type"].str.lower().fillna("none")
        result["is_proxy"] = (~proxy.isin(["none", "unknown"])).astype(np.int8)
        result["is_tor"] = proxy.str.contains("tor", case=False, na=False).astype(np.int8)
        result["is_datacenter"] = (
            df["connection_type"].str.lower().str.contains("datacenter", na=False)
        ).astype(np.int8)

        # Proxy risk score: none=0, unknown=1, proxy=2, tor=3
        proxy_risk = pd.Series(0, index=df.index, dtype=np.int8)
        proxy_risk = proxy_risk.where(proxy.isin(["none"]), 1)
        proxy_risk = proxy_risk.where(~result["is_proxy"].astype(bool), 2)
        proxy_risk = proxy_risk.where(~result["is_tor"].astype(bool), 3)
        result["proxy_risk_score"] = proxy_risk

        return result

    # ──────────────────────────────────────────────────
    # Full Pipeline
    # ──────────────────────────────────────────────────

    def _generate_features_single(
        self,
        df: pd.DataFrame,
        include_velocity: bool = True,
        fast_mode: bool = True,
    ) -> pd.DataFrame:
        """Internal helper to generate features for a single DataFrame."""
        logger.info("Starting feature engineering pipeline...")

        # Ensure sorted by timestamp
        if "event_timestamp" in df.columns:
            ts = pd.to_datetime(df["event_timestamp"])
            if not ts.is_monotonic_increasing:
                logger.warning("DataFrame not sorted by timestamp, sorting now...")
                sort_idx = ts.argsort()
                df = df.iloc[sort_idx].reset_index(drop=True)

        features = []

        # A. Static features
        logger.info("  → Static features...")
        features.append(self.generate_static_features(df))

        # B. Categorical encoding
        logger.info("  → Categorical encoding...")
        features.append(self.generate_categorical_features(df))

        # C. Temporal features
        logger.info("  → Temporal features...")
        features.append(self.generate_temporal_features(df))

        # D. Velocity features (per entity)
        if include_velocity:
            entity_cols = [
                "customer_id",
                "card_id",
                "device_id",
                "ip_id",
                "merchant_id",
                "account_id",
            ]
            for entity_col in entity_cols:
                if entity_col in df.columns:
                    logger.info("  → Velocity features for %s...", entity_col)
                    if fast_mode:
                        features.append(self.generate_velocity_features_fast(df, entity_col))
                    else:
                        features.append(self.generate_velocity_features(df, entity_col))

        # E. Geographic features
        logger.info("  → Geographic features...")
        features.append(self.generate_geographic_features(df))

        # F. Identity features
        logger.info("  → Identity/device risk features...")
        features.append(self.generate_identity_features(df))

        # Combine all
        result = pd.concat(features, axis=1)

        # Replace inf/nan
        result = result.replace([np.inf, -np.inf], 0).fillna(0)

        # Validate no leakage columns leaked into output
        forbidden = set(self.settings.leakage.forbidden_online_columns)
        leaked = forbidden & set(result.columns)
        if leaked:
            raise ValueError(
                f"LEAKAGE DETECTED: Forbidden columns found in feature output: {leaked}"
            )

        logger.info(
            "Feature engineering complete: %d features, %d rows",
            len(result.columns),
            len(result),
        )

        return result

    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_velocity: bool = True,
        fast_mode: bool = True,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate all features for the dataset.

        Args:
            df: Target DataFrame (must be sorted by event_timestamp).
            include_velocity: Whether to compute velocity features.
            fast_mode: Use fast approximate velocity computation.
            historical_df: Optional prior historical DataFrame (e.g. TRAIN when computing VAL)
                to allow velocity features to leverage prior activity without future leakage.

        Returns:
            DataFrame with all engineered features.
        """
        if historical_df is not None and not historical_df.empty:
            logger.info(
                "Prepending historical state (%d rows) for velocity feature generation...",
                len(historical_df),
            )
            combined_df = pd.concat([historical_df, df], ignore_index=True)
            if "event_timestamp" in combined_df.columns:
                ts = pd.to_datetime(combined_df["event_timestamp"])
                if not ts.is_monotonic_increasing:
                    combined_df = combined_df.sort_values("event_timestamp").reset_index(drop=True)

            combined_features = self._generate_features_single(
                combined_df, include_velocity=include_velocity, fast_mode=fast_mode
            )
            return combined_features.iloc[len(historical_df) :].reset_index(drop=True)
        else:
            return self._generate_features_single(
                df, include_velocity=include_velocity, fast_mode=fast_mode
            )

    def generate_online_features(self, transaction: dict) -> dict:
        """Generate features for a single online transaction (ultra-fast direct dict calculation).

        In online inference, velocity features come from the Redis feature store.
        """
        # Validate no leakage
        forbidden = set(self.settings.leakage.forbidden_online_columns)
        leaked = [k for k in forbidden if k in transaction]
        if leaked:
            raise ValueError(
                f"LEAKAGE VIOLATION: Post-event columns in online transaction: {leaked}"
            )

        res: dict = {}

        # A. Amount & static
        amount = float(transaction.get("amount", 0.0))
        credit_limit = float(transaction.get("credit_limit", 5000))
        res["log_amount"] = float(np.log1p(max(0.0, amount)))
        res["amount_credit_ratio"] = min(
            100.0, amount / (credit_limit if credit_limit != 0 else 1.0)
        )

        res["is_international"] = int(transaction.get("international_flag", 0))
        res["is_installment"] = int(transaction.get("installment_flag", 0))

        # Amount buckets: [-inf, 1, 10, 50, 100, 500, 1000, 5000, inf]
        bins_amount = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
        res["amount_bucket"] = int(np.searchsorted(bins_amount, amount))

        res["is_micro_txn"] = 1 if amount < 1.0 else 0
        res["is_high_value_txn"] = 1 if amount > 5000 else 0

        cust_tenure = float(transaction.get("customer_tenure_days", 0))
        res["customer_tenure_bucket"] = int(np.searchsorted([30, 90, 365, 730], cust_tenure))

        card_age = float(transaction.get("card_age_days", 0))
        res["card_age_bucket"] = int(np.searchsorted([30, 90, 365, 730], card_age))

        dev_age = float(transaction.get("device_age_days", 0))
        res["device_age_bucket"] = int(np.searchsorted([7, 30, 90, 365], dev_age))

        # B. Categorical encoding (use fitted maps if available, else default 0.0)
        for col in self.CATEGORICAL_COLS:
            if self._is_fitted and col in self._categorical_freq_maps:
                val = transaction.get(col, "")
                res[f"{col}_encoded"] = self._categorical_freq_maps[col].get(val, 0.0)
            else:
                res[f"{col}_encoded"] = 0.0

        # C. Temporal features
        raw_ts = transaction.get("event_timestamp", "")
        if isinstance(raw_ts, str):
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now()
        elif isinstance(raw_ts, datetime):
            dt = raw_ts
        else:
            dt = datetime.now()

        res["hour_of_day"] = dt.hour
        res["minute_bucket"] = dt.hour * 4 + dt.minute // 15
        res["day_of_week"] = dt.weekday()
        res["day_of_month"] = dt.day
        res["is_weekend"] = 1 if dt.weekday() in (5, 6) else 0
        res["is_night"] = 1 if (dt.hour >= 22 or dt.hour < 7) else 0

        # E. Geographic features
        txn_country = transaction.get("transaction_country", "")
        pairs = {
            "customer": "customer_country",
            "billing": "billing_country",
            "card": "card_country",
            "device": "device_country",
            "ip": "ip_country",
            "merchant": "merchant_country",
        }
        mismatch_count = 0
        for prefix, col in pairs.items():
            val = transaction.get(col, "")
            mism = 1 if (val != txn_country) else 0
            res[f"{prefix}_txn_country_mismatch"] = mism
            mismatch_count += mism

        res["geo_mismatch_count"] = mismatch_count
        res["cross_border_flag"] = (
            1 if (transaction.get("customer_country", "") != txn_country) else 0
        )

        # F. Identity features
        res["is_new_device"] = 1 if dev_age < 7 else 0

        proxy = str(transaction.get("proxy_type", "none")).lower()
        res["is_proxy"] = 0 if proxy in ("none", "unknown") else 1
        res["is_tor"] = 1 if "tor" in proxy else 0

        conn_type = str(transaction.get("connection_type", "")).lower()
        res["is_datacenter"] = 1 if "datacenter" in conn_type else 0

        proxy_risk = 0
        if proxy not in ("none",):
            proxy_risk = 1
        if res["is_proxy"]:
            proxy_risk = 2
        if res["is_tor"]:
            proxy_risk = 3
        res["proxy_risk_score"] = proxy_risk

        return res
