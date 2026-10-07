"""Score customers with a trained model.

Usage::

    python -m credit_risk.predict                      # built-in sample customer
    python -m credit_risk.predict --input customer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from . import config

SAMPLE_CUSTOMER = {
    "income": 55000,
    "credit_score": 680,
    "loan_amount": 25000,
    "debt_ratio": 0.35,
    "employment_years": 6,
    "age": 35,
    "previous_defaults": 0,
    "account_balance": 15000,
    "employment_type": "Salaried",
    "loan_type": "Personal",
}


def load_model(path: Path = config.MODEL_PATH):
    """Load a trained pipeline. Only load model files you trust (joblib uses pickle)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one first: python -m credit_risk.train"
        )
    return joblib.load(path)


def predict(model, customers: dict | list[dict]) -> list[dict]:
    """Score one or more customers given as dicts keyed by feature name."""
    records = [customers] if isinstance(customers, dict) else list(customers)
    df = pd.DataFrame(records)

    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    df = df[config.FEATURES]

    proba = model.predict_proba(df)[:, 1]
    return [
        {
            "default_probability": float(p),
            "prediction": "Default" if p >= config.RISK_THRESHOLD else "No Default",
            "risk": "HIGH RISK" if p >= config.RISK_THRESHOLD else "LOW RISK",
        }
        for p in proba
    ]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, help="JSON file: one customer object or a list")
    p.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    args = p.parse_args(argv)

    customers = json.loads(args.input.read_text()) if args.input else SAMPLE_CUSTOMER
    results = predict(load_model(args.model_path), customers)
    print(json.dumps(results if isinstance(customers, list) else results[0], indent=2))


if __name__ == "__main__":
    main()
