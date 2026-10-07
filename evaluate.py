"""Model evaluation: metrics and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, X, y) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC-AUC on ``(X, y)``."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def save_confusion_matrix(model, X, y, path: Path) -> np.ndarray:
    """Save a confusion-matrix plot to ``path`` and return the matrix."""
    cm = confusion_matrix(y, model.predict(X))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(ax=ax)
    ax.set_title("Credit Risk Classification - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm
