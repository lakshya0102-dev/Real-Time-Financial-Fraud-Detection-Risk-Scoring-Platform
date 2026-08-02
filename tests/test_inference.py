"""API endpoint and inference pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app


class TestAPIEndpoints:
    """Test FastAPI endpoint responses."""

    @pytest.fixture
    def client(self):
        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_version_endpoint(self, client):
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "fraud_api_up" in response.text
