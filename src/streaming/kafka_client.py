"""
Kafka streaming — producer, consumer, and topic management.

Topics:
  transactions.raw          — raw incoming transactions
  transactions.validated    — schema-validated transactions
  transactions.scored       — scored transactions with decisions
  fraud.alerts              — high-risk fraud alerts
  fraud.review              — transactions sent for manual review
  fraud.blocked             — blocked transactions
  model.events              — model lifecycle events
  dead_letter.transactions  — invalid/malformed transactions
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

from src.config.settings import get_settings, KafkaConfig

logger = logging.getLogger(__name__)


class KafkaProducer:
    """Kafka producer for transaction events."""

    def __init__(self, config: Optional[KafkaConfig] = None) -> None:
        self.config = config or get_settings().kafka
        self._producer = None

    def connect(self) -> None:
        """Initialize Kafka producer connection."""
        try:
            from kafka import KafkaProducer as _KafkaProducer

            self._producer = _KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1,
            )
            logger.info("Kafka producer connected to %s", self.config.bootstrap_servers)
        except Exception as e:
            logger.error("Failed to connect Kafka producer: %s", e)
            raise

    def send(self, topic: str, key: str, value: dict) -> None:
        """Send a message to a Kafka topic."""
        if self._producer is None:
            raise RuntimeError("Producer not connected. Call connect() first.")

        try:
            future = self._producer.send(topic, key=key, value=value)
            future.get(timeout=10)
        except Exception as e:
            logger.error("Failed to send message to %s: %s", topic, e)
            # Send to dead letter queue
            try:
                self._producer.send(
                    self.config.topic_dlq,
                    key=key,
                    value={"original_topic": topic, "error": str(e), "data": value},
                )
            except Exception:
                logger.error("Failed to send to dead letter queue")
            raise

    def flush(self) -> None:
        """Flush pending messages."""
        if self._producer:
            self._producer.flush()

    def close(self) -> None:
        """Close the producer."""
        if self._producer:
            self._producer.close()
            logger.info("Kafka producer closed")


class KafkaConsumer:
    """Kafka consumer for transaction processing pipeline."""

    def __init__(
        self,
        topics: list[str],
        group_id: Optional[str] = None,
        config: Optional[KafkaConfig] = None,
    ) -> None:
        self.config = config or get_settings().kafka
        self.topics = topics
        self.group_id = group_id or self.config.consumer_group
        self._consumer = None
        self._running = False

    def connect(self) -> None:
        """Initialize Kafka consumer."""
        try:
            from kafka import KafkaConsumer as _KafkaConsumer

            self._consumer = _KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.config.bootstrap_servers.split(","),
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                max_poll_interval_ms=300000,
            )
            logger.info("Kafka consumer connected, topics: %s", self.topics)
        except Exception as e:
            logger.error("Failed to connect Kafka consumer: %s", e)
            raise

    def consume(
        self,
        handler: Callable[[dict], None],
        max_messages: Optional[int] = None,
    ) -> None:
        """Consume messages and process with handler."""
        if self._consumer is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        self._running = True
        count = 0

        try:
            for message in self._consumer:
                if not self._running:
                    break

                try:
                    handler(message.value)
                    count += 1
                except Exception as e:
                    logger.error(
                        "Error processing message from %s: %s",
                        message.topic,
                        e,
                    )

                if max_messages and count >= max_messages:
                    break
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.close()

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self._running = False

    def close(self) -> None:
        """Close the consumer."""
        if self._consumer:
            self._consumer.close()
            logger.info("Kafka consumer closed")


class TopicManager:
    """Manages Kafka topic creation and configuration."""

    TOPICS = [
        "transactions.raw",
        "transactions.validated",
        "transactions.scored",
        "fraud.alerts",
        "fraud.review",
        "fraud.blocked",
        "model.events",
        "dead_letter.transactions",
    ]

    @staticmethod
    def create_topics(bootstrap_servers: str, num_partitions: int = 3) -> None:
        """Create all required topics."""
        try:
            from kafka.admin import KafkaAdminClient, NewTopic

            admin = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers.split(","),
            )

            existing = admin.list_topics()
            new_topics = [
                NewTopic(name=t, num_partitions=num_partitions, replication_factor=1)
                for t in TopicManager.TOPICS
                if t not in existing
            ]

            if new_topics:
                admin.create_topics(new_topics)
                logger.info("Created topics: %s", [t.name for t in new_topics])
            else:
                logger.info("All topics already exist")

            admin.close()
        except Exception as e:
            logger.error("Failed to create topics: %s", e)
