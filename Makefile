.PHONY: install test test-unit test-integration test-leakage lint format train evaluate predict benchmark docker-build docker-up docker-down clean

# ──────────────────────────────────────────────────────
# Setup & Installation
# ──────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,infra,viz]"

# ──────────────────────────────────────────────────────
# Quality & Testing
# ──────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ training/ scripts/

format:
	ruff check --fix src/ tests/ training/ scripts/
	ruff format src/ tests/ training/ scripts/

test:
	pytest tests/ -v -m "not integration" --tb=short

test-all:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v -m "not integration and not slow"

test-integration:
	pytest tests/ -v -m integration

test-leakage:
	pytest tests/ -v tests/test_leakage.py

test-coverage:
	pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing --cov-report=html

# ──────────────────────────────────────────────────────
# Dataset & Pipeline Execution
# ──────────────────────────────────────────────────────

validate-dataset:
	python -c "from src.data.dataset_validator import validate_dataset; r = validate_dataset(); print(r.summary())"

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

predict:
	python scripts/predict.py

benchmark:
	python scripts/benchmark.py

# ──────────────────────────────────────────────────────
# Services & Docker
# ──────────────────────────────────────────────────────

api:
	uvicorn inference.server:app --host 0.0.0.0 --port 8000 --reload

replay:
	python replay/replay_service.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage
