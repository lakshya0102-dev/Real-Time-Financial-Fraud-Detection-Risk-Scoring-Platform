"""
Deterministic transaction replay service.

Reads ONLY data/CFR_data.csv, sorts by event_timestamp, and publishes
transactions chronologically to Kafka. Supports:
  - Accelerated simulation (configurable speed multiplier)
  - Real-time-like simulation (1x speed)
  - Deterministic ordering
  - Never sends post-event labels to online inference

This is the bridge between the historical dataset and the real-time architecture.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.data.loader import load_dataset
from src.validation.leakage import LeakageValidator

logger = logging.getLogger(__name__)

# Columns that MUST NOT be sent to the online inference service
FORBIDDEN_REPLAY_COLUMNS = [
    "true_fraud_label",
    "fraud_scenario",
    "observed_label",
    "label_timestamp",
]


class ReplayService:
    """Replays historical transactions from CFR_data.csv.

    Properties:
      - Reads ONLY from data/CFR_data.csv
      - Sorts by event_timestamp
      - Publishes chronologically
      - Configurable replay speed
      - NEVER sends post-event labels to inference
    """

    def __init__(
        self,
        speed_multiplier: float = 100.0,
        max_transactions: Optional[int] = None,
        use_kafka: bool = True,
    ) -> None:
        self.speed_multiplier = speed_multiplier
        self.max_transactions = max_transactions
        self.use_kafka = use_kafka
        self.leakage_validator = LeakageValidator()
        self._kafka_producer = None

    def _init_kafka(self) -> None:
        """Initialize Kafka producer."""
        if not self.use_kafka:
            return

        try:
            from src.streaming.kafka_client import KafkaProducer
            self._kafka_producer = KafkaProducer()
            self._kafka_producer.connect()
            logger.info("Kafka producer initialized for replay")
        except Exception as e:
            logger.warning("Kafka not available: %s. Running in log-only mode.", e)
            self.use_kafka = False

    def _strip_labels(self, record: dict) -> dict:
        """Remove post-event labels from transaction record."""
        return {k: v for k, v in record.items() if k not in FORBIDDEN_REPLAY_COLUMNS}

    def run(self) -> None:
        """Execute the replay."""
        logger.info("=" * 60)
        logger.info("REPLAY SERVICE STARTING")
        logger.info("  Speed: %.1fx", self.speed_multiplier)
        logger.info("  Max transactions: %s", self.max_transactions or "ALL")
        logger.info("  Kafka: %s", self.use_kafka)
        logger.info("=" * 60)

        # Load dataset
        df = load_dataset(parse_dates=True)
        df = df.sort_values("event_timestamp").reset_index(drop=True)

        if self.max_transactions:
            df = df.head(self.max_transactions)

        logger.info("Loaded %d transactions for replay", len(df))

        # Initialize Kafka
        self._init_kafka()

        # Replay loop
        settings = get_settings()
        prev_ts = None
        sent_count = 0
        start_time = time.time()

        for idx, row in df.iterrows():
            record = row.to_dict()

            # Strip post-event labels (CRITICAL: never leak to inference)
            online_record = self._strip_labels(record)

            # Convert timestamps to strings for JSON serialization
            for k, v in online_record.items():
                if isinstance(v, (pd.Timestamp, datetime)):
                    online_record[k] = v.isoformat()

            # Control replay speed
            if prev_ts is not None and self.speed_multiplier < 1000:
                current_ts = pd.Timestamp(row["event_timestamp"])
                real_delta = (current_ts - prev_ts).total_seconds()
                if real_delta > 0:
                    sleep_time = real_delta / self.speed_multiplier
                    if sleep_time > 0.001:  # Skip sub-millisecond sleeps
                        time.sleep(min(sleep_time, 1.0))  # Cap at 1 second

            prev_ts = pd.Timestamp(row["event_timestamp"])

            # Publish
            if self.use_kafka and self._kafka_producer:
                try:
                    self._kafka_producer.send(
                        settings.kafka.topic_raw,
                        key=online_record["transaction_id"],
                        value=online_record,
                    )
                except Exception as e:
                    logger.error("Failed to send transaction %s: %s",
                                online_record["transaction_id"], e)

            sent_count += 1

            if sent_count % 10000 == 0:
                elapsed = time.time() - start_time
                rate = sent_count / elapsed if elapsed > 0 else 0
                logger.info(
                    "  Replayed %d / %d transactions (%.0f txn/s)",
                    sent_count, len(df), rate,
                )

        # Flush
        if self._kafka_producer:
            self._kafka_producer.flush()
            self._kafka_producer.close()

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(
            "REPLAY COMPLETE: %d transactions in %.1f seconds (%.0f txn/s)",
            sent_count, elapsed, sent_count / elapsed if elapsed > 0 else 0,
        )
        logger.info("=" * 60)


def main() -> None:
    """CLI entry point for the replay service."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Transaction Replay Service")
    parser.add_argument("--speed", type=float, default=100.0,
                       help="Replay speed multiplier (default: 100x)")
    parser.add_argument("--max-txn", type=int, default=None,
                       help="Max transactions to replay")
    parser.add_argument("--no-kafka", action="store_true",
                       help="Run without Kafka (log-only)")

    args = parser.parse_args()

    service = ReplayService(
        speed_multiplier=args.speed,
        max_transactions=args.max_txn,
        use_kafka=not args.no_kafka,
    )
    service.run()


if __name__ == "__main__":
    main()
