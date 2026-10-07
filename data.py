"""Synthetic credit-risk dataset generation.

To use real data, replace :func:`generate_synthetic_data` with
``pd.read_csv(...)`` returning a frame that has the columns listed in
:data:`credit_risk.config.FEATURES` plus :data:`credit_risk.config.TARGET`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from . import config


def generate_synthetic_data(
    n_samples: int = 5000,
    missing_rate: float = 0.03,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Build a synthetic, imbalanced (75/25) credit-risk dataset.

    Numeric columns come from ``make_classification`` and are rescaled to
    plausible business ranges. Categorical columns are random noise and
    carry no signal. A fraction of some numeric columns is set to NaN to
    exercise the imputation step.
    """
    rng = np.random.default_rng(random_state)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.75, 0.25],
        class_sep=1.2,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=config.NUMERIC_FEATURES)

    df["income"] = df["income"].abs() * 20_000 + 25_000
    df["credit_score"] = np.clip(650 + df["credit_score"] * 50, 300, 850)
    df["loan_amount"] = df["loan_amount"].abs() * 15_000 + 5_000
    df["debt_ratio"] = np.clip(df["debt_ratio"].abs() / 5, 0, 1)
    df["employment_years"] = np.clip(df["employment_years"].abs() * 2, 0, 40)
    df["age"] = np.clip(df["age"].abs() * 5 + 30, 18, 75)
    df["previous_defaults"] = np.clip(np.round(df["previous_defaults"].abs()), 0, 5)
    df["account_balance"] = df["account_balance"].abs() * 10_000 + 1_000

    df["employment_type"] = rng.choice(config.EMPLOYMENT_TYPES, size=len(df))
    df["loan_type"] = rng.choice(config.LOAN_TYPES, size=len(df))
    df[config.TARGET] = y

    n_missing = int(len(df) * missing_rate)
    for column in config.COLUMNS_WITH_MISSING:
        idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, column] = np.nan

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` using the configured feature and target columns."""
    return df[config.FEATURES], df[config.TARGET]
