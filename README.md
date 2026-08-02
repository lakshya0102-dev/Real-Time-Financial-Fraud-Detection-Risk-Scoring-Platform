# Real-Time Financial Fraud Detection & Risk Scoring Platform

An end-to-end machine learning platform for **real-time financial transaction fraud detection and risk scoring**, designed with a production-oriented ML pipeline covering data validation, temporal feature engineering, model experimentation, probability calibration, cost-sensitive decision making, explainability, statistical confidence intervals, and temporal drift monitoring.

## 🚀 Project Overview

Financial fraud detection is a highly imbalanced classification problem where missing a fraudulent transaction can be significantly more costly than incorrectly flagging a legitimate transaction.

This project addresses that problem through a complete ML pipeline that includes:

* Dataset validation and integrity verification
* Temporal train/validation/test splitting
* Leakage-aware feature engineering
* Transaction velocity features
* XGBoost and LightGBM experimentation
* Probability calibration
* Cost-sensitive threshold optimization
* SHAP explainability
* Bootstrap confidence intervals
* Population Stability Index (PSI) drift detection
* Production artifact bundling
* Model selection based on PR-AUC
* Fraud risk scoring and decision policies

---

## 🎯 Objectives

The primary objectives are to:

1. Detect fraudulent financial transactions.
2. Produce calibrated fraud probabilities.
3. Optimize decisions based on financial cost rather than accuracy alone.
4. Minimize false positives while maintaining meaningful fraud recall.
5. Detect temporal distribution changes between training and production-like data.
6. Provide interpretable predictions for fraud analysts.
7. Package the trained model and preprocessing components for production inference.

---

## 🏗️ End-to-End Architecture

```text
                         ┌──────────────────────┐
                         │   Transaction Data   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌────────────────────────────┐
                    │    Dataset Validation      │
                    │ SHA-256 / Schema / Nulls   │
                    │ Duplicates / Distribution  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │     Temporal Data Split     │
                    │ Train / Validation / Test   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    Feature Engineering      │
                    │ Static / Temporal / Geo     │
                    │ Velocity / Risk Features    │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
          ┌─────────────────┐          ┌─────────────────┐
          │   XGBoost       │          │    LightGBM     │
          │   Baseline      │          │   Comparison    │
          └────────┬────────┘          └────────┬────────┘
                   │                            │
                   └──────────────┬─────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │      Model Selection       │
                    │       PR-AUC Based         │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │  Probability Calibration   │
                    │ Platt / Isotonic Regression │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Cost-Sensitive Decision     │
                    │    Threshold Optimization   │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
          ┌─────────────────┐          ┌─────────────────┐
          │ SHAP Explain-   │          │ Drift Detection │
          │ ability         │          │ PSI / JS        │
          └─────────────────┘          └─────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   Production Artifact      │
                    │         Bundle              │
                    └────────────────────────────┘
```

---

# 📊 Dataset

The dataset used in the project contains:

| Property                  |                Value |
| ------------------------- | -------------------: |
| Rows                      |              999,984 |
| Columns                   |                   54 |
| Fraud Transactions        |                3,999 |
| Observed Labels           |               13,923 |
| Outliers                  |               20,000 |
| Null Values               |                    0 |
| Duplicate Transaction IDs |                    0 |
| Data Period               | Jan 1 – Jun 29, 2026 |

The dataset was validated before model training.

### Dataset Validation

The validation pipeline checks:

* File existence
* File readability
* SHA-256 checksum
* Column schema
* Data types
* Row count
* Transaction ID uniqueness
* Missing values
* Timestamp range
* Target distribution
* Outlier/fraud separation

**Validation status: PASSED**

---

# 🧠 Feature Engineering

The pipeline generates both static and historical transaction features.

The final engineered dataset contains **79 features**.

Feature categories include:

### Transaction Features

* Transaction amount
* Log-transformed amount
* Amount bucket
* Amount-to-credit ratio
* High-value transaction indicator
* Micro-transaction indicator

### Temporal Features

* Hour of day
* Day of week
* Day of month
* Weekend indicator
* Night-time indicator
* Minute bucket

### Geographic Features

* Country mismatch indicators
* Cross-border indicators
* Geographic mismatch counts

### Identity & Device Risk

* Proxy indicators
* TOR indicators
* Datacenter indicators
* Device trust level
* Device age
* Authentication method
* Connection type

### Velocity Features

Historical transaction behavior is calculated across:

* Customer
* Card
* Device
* IP
* Merchant
* Account

Examples include:

```text
customer_total_txn_count
card_total_txn_count
device_total_txn_count
ip_total_txn_count
merchant_total_txn_count
account_total_txn_count
```

and:

```text
time_since_prev_customer_txn_seconds
time_since_prev_card_txn_seconds
time_since_prev_device_txn_seconds
time_since_prev_ip_txn_seconds
time_since_prev_merchant_txn_seconds
time_since_prev_account_txn_seconds
```

---

# ⏱️ Temporal Validation Strategy

Random train/test splitting was intentionally avoided.

The dataset is split chronologically:

```text
TRAIN
Jan 1 → Apr 30
666,660 rows

        ↓

VALIDATION
May 1 → May 31
172,188 rows

    ├── Calibration
    │   83,416 rows
    │
    └── Threshold Optimization
        88,772 rows

        ↓

TEST
Jun 1 → Jun 29
161,136 rows
```

This approach better represents real-world fraud detection, where models must predict future transactions using historical information.

---

# 🧪 Model Experiments

Four primary model configurations were evaluated.

| Experiment | Model               | Features   |     PR-AUC |    ROC-AUC |
| ---------- | ------------------- | ---------- | ---------: | ---------: |
| EXP 1      | Logistic Regression | Basic      |     0.2599 |     0.7484 |
| EXP 2      | XGBoost             | Basic      |     0.4447 | **0.7668** |
| EXP 3      | XGBoost             | Engineered | **0.4880** |     0.7536 |
| EXP 4      | LightGBM            | Engineered |     0.2696 |     0.7584 |

### 🏆 Selected Model

**XGBoost + Engineered Features**

Selection criterion:

```text
PR-AUC = 0.4880
```

PR-AUC was prioritized because fraud detection involves a highly imbalanced target where precision-recall behavior is more informative than accuracy alone.

---

# 📈 Validation Performance

The selected XGBoost model achieved:

```text
PR-AUC:        0.4880
ROC-AUC:       0.7536
Precision:     0.7461
Recall:        0.4919
F1:            0.5929
Specificity:   0.9993
FPR:           0.0007
Brier Score:   0.091133
```

Validation confusion matrix:

```text
                 Predicted
              Legit       Fraud

Actual Legit  171,393       114
Actual Fraud      346       335
```

---

# 🎯 Probability Calibration

Raw model probabilities were calibrated using:

* Platt Scaling
* Isotonic Regression

The calibration process used a dedicated calibration validation split to avoid using the final test set for calibration.

### Results

| Method              |  Brier Score |          ECE |
| ------------------- | -----------: | -----------: |
| Before Calibration  |     0.088832 |     0.288699 |
| Platt Scaling       |     0.002301 |     0.000442 |
| Isotonic Regression | **0.002197** | **0.000183** |

**Selected calibration method: Isotonic Regression**

---

# 🧮 Cost-Sensitive Decision Engine

Fraud detection is not optimized purely for classification metrics.

The system considers:

* Fraud prevented
* Fraud missed
* False-positive cost
* Manual review cost
* Financial net benefit

The optimized decision policy was:

```text
APPROVE  : probability < 0.3479

REVIEW   : 0.3479 ≤ probability < 0.8034

BLOCK    : probability ≥ 0.8034
```

The thresholds were optimized on the validation optimization set and then evaluated on the untouched test set.

### Test Results

```text
Net Benefit:     $54,563.63
Fraud Caught:    300
Fraud Missed:    359
False Declines:  9
Reviews:         96
```

---

# 🧪 Final Test Evaluation

After calibration, the production pipeline achieved:

```text
PR-AUC:        0.4527
ROC-AUC:       0.7241
Precision:     0.8929
Recall:        0.4552
F1:            0.6030
Specificity:   0.9998
FPR:           0.0002
Brier Score:   0.002397
```

Confusion matrix:

```text
                 Predicted
              Legit       Fraud

Actual Legit  160,441        36
Actual Fraud      359       300
```

---

# 📊 Bootstrap Confidence Intervals

The final test metrics were evaluated using **500 bootstrap samples**.

| Metric      |      Score |                  95% CI |
| ----------- | ---------: | ----------------------: |
| PR-AUC      |     0.4527 |         0.4188 – 0.4907 |
| ROC-AUC     |     0.7241 |         0.7008 – 0.7504 |
| Precision   |     0.8929 |         0.8610 – 0.9206 |
| Recall      |     0.4552 |         0.4193 – 0.4916 |
| F1          |     0.6030 |         0.5692 – 0.6363 |
| P@100       |     0.9900 |         0.9800 – 1.0000 |
| P@500       |     0.6140 |         0.5490 – 0.6751 |
| Net Benefit | $54,368.63 | $30,347.98 – $79,233.12 |

---

# 🔍 Explainability with SHAP

SHAP TreeExplainer was used to understand the model's decision-making process.

Top features by SHAP importance:

```text
1. proxy_type_encoded
2. authentication_method_encoded
3. log_amount
4. merchant_total_txn_count
5. connection_type_encoded
6. payment_channel_encoded
7. is_proxy
8. proxy_risk_score
9. time_since_prev_device_txn_seconds
10. card_total_txn_count
```

This allows individual fraud predictions to be investigated instead of treating the model as a black box.

---

# 📉 Temporal Drift Monitoring

The pipeline calculates distribution drift between training and test data using:

* Population Stability Index (PSI)
* Jensen-Shannon divergence

Overall mean PSI:

```text
0.5019
```

Significant drift was detected in several historical/velocity features, including:

```text
merchant_total_txn_count
customer_total_txn_count
account_total_txn_count
device_total_txn_count
card_total_txn_count
ip_total_txn_count
customer_avg_amount
merchant_avg_amount
device_avg_amount
account_avg_amount
```

This demonstrates why production fraud systems require continuous monitoring rather than assuming model performance remains constant after deployment.

---

# ⚠️ Outlier Feature Analysis

An A/B analysis was performed on the `is_outlier` variable.

The 20,000 outlier rows contained **zero fraud examples**.

Including `is_outlier` directly would therefore create an artificial shortcut:

```text
is_outlier = 1 → model learns "never fraud"
```

Consequently:

> `is_outlier` is treated as data-quality metadata rather than a legitimate predictive feature and is excluded from the production model.

---

# 📦 Production Artifacts

The pipeline generates a unified production artifact bundle containing:

```text
production_model.pkl
production_calibrator.pkl
production_feature_engineer.pkl
production_scaler.pkl
production_bundle.pkl
```

The artifact bundle performs compatibility verification using:

* Bundle version
* Schema hash
* Component compatibility checks

Example:

```text
Artifact Bundle compatibility verified
Version: v1.0.0-20260801T091641
Schema Hash: a78c5c226348bce4
```

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* XGBoost
* LightGBM
* Scikit-learn
* SHAP

### Data Processing

* Pandas
* NumPy

### Evaluation

* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1
* Brier Score
* Expected Calibration Error
* Precision@K
* Recall@K
* Bootstrap Confidence Intervals
* PSI
* Jensen-Shannon Divergence

### Engineering

* Git
* GitHub
* Pickle-based artifact serialization
* Structured logging
* Dataset integrity validation

---

# 📁 Project Structure

```text
Real-Time-Financial-Fraud-Detection-Risk-Scoring-Platform/
│
├── data/
│   └── README.md
│
├── src/
│   ├── data/
│   │   └── dataset_validator.py
│   │
│   ├── features/
│   │   └── pipeline.py
│   │
│   ├── models/
│   │   ├── temporal_split.py
│   │   └── artifact_bundle.py
│   │
│   ├── calibration/
│   │   └── calibrator.py
│   │
│   ├── scoring/
│   │   └── decision_engine.py
│   │
│   └── explainability/
│       └── shap_explainer.py
│
├── training/
│   └── run_experiments.py
│
├── artifacts/
│   ├── models/
│   └── reports/
│
├── tests/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/lakshya0102-dev/Real-Time-Financial-Fraud-Detection-Risk-Scoring-Platform.git
cd Real-Time-Financial-Fraud-Detection-Risk-Scoring-Platform
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Training Pipeline

Run:

```bash
python training/run_experiments.py
```

The pipeline performs:

```text
1. Dataset validation
2. Dataset loading
3. Temporal splitting
4. Basic feature engineering
5. Velocity feature engineering
6. Model experiments
7. Model selection
8. Probability calibration
9. Threshold optimization
10. SHAP explainability
11. Bootstrap confidence intervals
12. Temporal drift analysis
13. Production artifact generation
```

---

# 🔐 Data & Security

The project should **not commit private or sensitive transaction data** to GitHub.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc
.env
.env.*
data/*.csv
data/*.parquet
*.pkl
artifacts/models/
artifacts/reports/
logs/
.DS_Store
```

For a public repository, use a synthetic dataset or provide instructions for reproducing the dataset locally.

---

# 🚧 Future Production Enhancements

The current pipeline provides a strong ML foundation. The next production-level improvements include:

* [ ] Bayesian hyperparameter optimization with Optuna
* [ ] MLflow experiment tracking
* [ ] Feast feature store
* [ ] FastAPI real-time inference service
* [ ] Kafka transaction streaming
* [ ] Docker / Docker Compose
* [ ] CI/CD pipeline
* [ ] Automated model validation
* [ ] Model registry
* [ ] Automated model rollback
* [ ] Prometheus metrics
* [ ] Grafana monitoring dashboard
* [ ] Automated drift alerts
* [ ] Online feature computation
* [ ] Real-time fraud-risk API

---

# 💼 Why This Project Matters

This project demonstrates more than simply training a classifier.

It addresses several challenges encountered in real-world financial ML systems:

```text
Class imbalance
       ↓
Temporal leakage prevention
       ↓
Historical velocity features
       ↓
Model comparison
       ↓
Probability calibration
       ↓
Cost-sensitive decisions
       ↓
Explainability
       ↓
Statistical uncertainty
       ↓
Data drift monitoring
       ↓
Production artifact management
```

The focus is therefore on **building an end-to-end fraud detection system rather than only achieving a high classification score**.

---

# 👨‍💻 Author

**Lakshya**

GitHub:

https://github.com/lakshya0102-dev

Repository:

https://github.com/lakshya0102-dev/Real-Time-Financial-Fraud-Detection-Risk-Scoring-Platform

---

# 📄 License

This project is intended for educational, portfolio, and research purposes.

Add an appropriate open-source license to the repository before distributing the project publicly.
