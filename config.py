"""Central configuration: paths, feature definitions and constants."""

from __future__ import annotations

import os
from pathlib import Path

RANDOM_STATE = 42

# Project root is the repo root when running from a source checkout.
# Set CREDIT_RISK_HOME to override (e.g. when the package is installed
# non-editable or run inside a container).
PROJECT_ROOT = Path(
    os.environ.get("CREDIT_RISK_HOME", Path(__file__).resolve().parents[2])
)
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "credit_risk_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "default"

NUMERIC_FEATURES = [
    "income",
    "credit_score",
    "loan_amount",
    "debt_ratio",
    "employment_years",
    "age",
    "previous_defaults",
    "account_balance",
]

CATEGORICAL_FEATURES = [
    "employment_type",
    "loan_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business"]
LOAN_TYPES = ["Personal", "Home", "Auto", "Education"]

# Columns that receive artificial missing values in the synthetic dataset.
COLUMNS_WITH_MISSING = ["income", "credit_score", "loan_amount", "employment_years"]

# Probability at or above which a customer is labelled HIGH RISK.
RISK_THRESHOLD = 0.50
