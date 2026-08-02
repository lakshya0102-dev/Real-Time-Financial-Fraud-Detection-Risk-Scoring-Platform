# Dataset Policy

## Absolute Dataset Rule

There is exactly **ONE** authoritative dataset for this project:

```
data/CFR_data.csv
```

**SHA-256:** `89b2ca3e6791d124e1e739d965b62c95491ad80a67f5d1f77ce7cddd2dd25ad8`

## What This Means

### The application MUST:

- Load the dataset ONLY from `data/CFR_data.csv`
- Validate the SHA-256 checksum at startup and training time
- Fail fast with a clear error if the dataset is missing or corrupted
- Keep the raw CSV file **immutable** — never modify it

### The application MUST NOT:

- Search the internet, Kaggle, or Hugging Face for a dataset
- Search local directories for alternative CSV/Parquet/JSON files
- Download a replacement dataset from any source
- Generate synthetic replacement records
- Silently substitute another data source
- Use `glob("**/*.csv")` or `find_dataset()` discovery patterns
- Create fallback dataset loading behavior
- Hard-code the dataset path in multiple locations

### Central Configuration

All code that needs the dataset path MUST import it from:

```python
from src.config.settings import get_settings

settings = get_settings()
dataset_path = settings.dataset.path
```

If the file is missing:

```python
raise DatasetNotFoundError(dataset_path)
```

### Synthetic Data in Tests

Synthetic data may **only** be generated for isolated unit tests (test fixtures).
Such fixtures MUST NEVER be used as the project's actual training dataset.

### Data Integrity Validation

At startup/training time, the system validates:

1. File existence
2. SHA-256 checksum
3. Column schema (54 columns)
4. Data types
5. Row count (≥ 999,984)
6. Transaction ID uniqueness
7. Timestamp range (Jan–Jun 2026)
8. Target distribution
9. No null values
10. No duplicate rows

## Target & Leakage Policy

**Target column:** `true_fraud_label`

**Forbidden online columns** (post-event, NEVER available at decision time):

| Column | Reason |
|---|---|
| `true_fraud_label` | Target variable |
| `fraud_scenario` | Reveals fraud type |
| `observed_label` | Delayed observation |
| `label_timestamp` | Post-event timestamp |

**`is_outlier` policy:** The 20,000 outlier rows contain ZERO fraud examples. Using `is_outlier` as a feature creates an artificial shortcut. It is excluded from the production model and treated as data-quality metadata only.
