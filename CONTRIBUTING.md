# Contributing to Real-Time Financial Fraud Detection Platform

Thank you for your interest in contributing! This document outlines guidelines and workflows for submitting improvements, bug fixes, and new features.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fraud-detection-platform.git
   cd fraud-detection-platform
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

---

## 🧪 Testing & Code Quality

Before submitting a Pull Request, ensure that all unit tests pass and code is formatted according to project standards:

1. **Run Unit Tests:**
   ```bash
   pytest tests/ -v -m "not integration"
   ```

2. **Lint & Code Formatting:**
   ```bash
   ruff check src/ tests/ training/ scripts/
   ruff format --check src/ tests/ training/ scripts/
   ```

---

## 🔒 Security & Data Policy

- **Never commit raw dataset files (`data/CFR_data.csv`) or model binaries (`*.pkl`).**
- **Never commit `.env` files, credentials, or private API keys.**
- **All dataset access MUST go through `src.data.loader.load_dataset()`.**
- **Online inference code MUST preserve post-event leakage boundaries.**

---

## 📥 Submitting Pull Requests

1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Commit your changes with clear, descriptive commit messages.
3. Verify that GitHub Actions CI passes on your branch.
4. Submit a Pull Request detailing the changes made and verification steps performed.
