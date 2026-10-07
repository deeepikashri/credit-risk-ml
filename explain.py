"""Model explainability: impurity-based and permutation importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config


def feature_importance(model) -> pd.DataFrame:
    """Random Forest impurity importance per (encoded) feature, descending."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_df(
    model, X, y, n_repeats: int = 5, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Drop in ROC-AUC when each *raw* input column is shuffled."""
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_importance_plot(importance: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    """Horizontal bar chart of the ``top_n`` most important features."""
    top = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top Credit Risk Predictors")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
