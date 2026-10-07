"""Model pipeline: preprocessing + Random Forest, and hyper-parameter search."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [8, 12, 16, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot encode categoricals."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numerical", numeric, config.NUMERIC_FEATURES),
            ("categorical", categorical, config.CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(random_state: int = config.RANDOM_STATE, **rf_params) -> Pipeline:
    """Full pipeline. ``class_weight='balanced'`` handles the class imbalance."""
    params = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(rf_params)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(**params)),
        ]
    )


def build_search(
    pipeline: Pipeline,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = config.RANDOM_STATE,
) -> RandomizedSearchCV:
    """Randomized hyper-parameter search optimising ROC-AUC."""
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
    )
