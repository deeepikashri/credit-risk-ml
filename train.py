"""Train, evaluate, explain and save the credit-risk model.

Usage::

    python -m credit_risk.train                 # full run with tuning
    python -m credit_risk.train --no-search     # skip hyper-parameter search
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are saved, never shown

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

from . import config  # noqa: E402
from .data import generate_synthetic_data, split_features_target  # noqa: E402
from .evaluate import compute_metrics, save_confusion_matrix  # noqa: E402
from .explain import (  # noqa: E402
    feature_importance,
    permutation_importance_df,
    save_importance_plot,
)
from .pipeline import build_pipeline, build_search  # noqa: E402

logger = logging.getLogger("credit_risk")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-samples", type=int, default=5000, help="synthetic rows")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--n-iter", type=int, default=10, help="random-search iterations")
    p.add_argument("--no-search", action="store_true", help="skip hyper-parameter search")
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    p.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    p.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # --- data -------------------------------------------------------------
    df = generate_synthetic_data(n_samples=args.n_samples, random_state=args.seed)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    logger.info("Dataset: %s | train=%d test=%d", df.shape, len(X_train), len(X_test))
    logger.info("Default rate: %.1f%%", 100 * y.mean())

    # --- baseline + cross-validation -------------------------------------
    pipeline = build_pipeline(random_state=args.seed)
    pipeline.fit(X_train, y_train)
    baseline = compute_metrics(pipeline, X_test, y_test)
    logger.info("Baseline test metrics: %s", baseline)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    logger.info("5-fold CV ROC-AUC: %.4f +/- %.4f", cv_scores.mean(), cv_scores.std())

    # --- tuning ----------------------------------------------------------
    best_params, best_cv = None, None
    if args.no_search:
        best_model = pipeline
    else:
        search = build_search(pipeline, n_iter=args.n_iter, random_state=args.seed)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params, best_cv = search.best_params_, float(search.best_score_)
        logger.info("Best params: %s (CV ROC-AUC %.4f)", best_params, best_cv)

    # --- final evaluation ------------------------------------------------
    final = compute_metrics(best_model, X_test, y_test)
    logger.info("Final test metrics: %s", final)
    cm = save_confusion_matrix(
        best_model, X_test, y_test, args.reports_dir / "confusion_matrix.png"
    )

    # --- explainability --------------------------------------------------
    importance = feature_importance(best_model)
    save_importance_plot(importance, args.reports_dir / "feature_importance.png")
    importance.to_csv(args.reports_dir / "feature_importance.csv", index=False)

    perm = permutation_importance_df(best_model, X_test, y_test, random_state=args.seed)
    perm.to_csv(args.reports_dir / "permutation_importance.csv", index=False)
    logger.info("Permutation importance:\n%s", perm.to_string(index=False))

    # --- persist ---------------------------------------------------------
    joblib.dump(best_model, args.model_path)
    logger.info("Saved model -> %s", args.model_path)

    report = {
        "baseline_test": baseline,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "tuned": not args.no_search,
        "best_params": best_params,
        "best_cv_roc_auc": best_cv,
        "final_test": final,
        "confusion_matrix": cm.tolist(),
    }
    with open(args.reports_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
