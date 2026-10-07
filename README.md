# Credit Risk Prediction & Explainable ML

An end-to-end credit default classifier built with scikit-learn: preprocessing, class-imbalance handling, cross-validation, hyper-parameter tuning, explainability and model serialization, all runnable from the command line.

> **Note:** This project uses a **synthetic dataset** (generated with `make_classification` and relabelled as credit features) so it runs out of the box. The two categorical features are random noise. Metrics below demonstrate the pipeline, not real-world credit performance.

## Pipeline

```
Raw data
  ├─ numeric     → median imputation → standard scaling
  └─ categorical → most-frequent imputation → one-hot encoding
        ↓
Random Forest (class_weight="balanced")
        ↓
Stratified K-fold CV → RandomizedSearchCV (ROC-AUC)
        ↓
Evaluation + feature / permutation importance → saved pipeline (.joblib)
```

The whole preprocessing + model pipeline is serialized together, so inference needs no manual preprocessing.

## Results

Hold-out test set (1,000 samples, ~25% default rate, seed 42):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline Random Forest | 0.875 | 0.847 | 0.615 | 0.713 | 0.929 |
| After randomized search | 0.864 | 0.872 | 0.540 | 0.667 | 0.931 |

5-fold stratified CV ROC-AUC (baseline): **0.891 ± 0.012**.

The search optimizes ROC-AUC, which improved slightly, but recall at the default 0.50 threshold dropped. In real credit scoring you would typically tune the decision threshold to the business cost of missed defaults.

Top drivers by permutation importance: `credit_score`, `debt_ratio`, `previous_defaults`, `income`.

| Confusion matrix | Feature importance |
|---|---|
| ![Confusion matrix](reports/confusion_matrix.png) | ![Feature importance](reports/feature_importance.png) |

## Project structure

```
credit-risk-ml/
├── src/credit_risk/
│   ├── config.py      # paths, feature lists, constants
│   ├── data.py        # synthetic data generation
│   ├── pipeline.py    # preprocessing, model, hyper-parameter search
│   ├── evaluate.py    # metrics and confusion matrix
│   ├── explain.py     # feature + permutation importance
│   ├── train.py       # training CLI
│   └── predict.py     # inference CLI / API
├── models/            # trained model (git-ignored)
├── reports/           # metrics, plots, importance tables
├── requirements.txt
└── pyproject.toml
```

## Getting started

```bash
git clone https://github.com/<your-username>/credit-risk-ml.git
cd credit-risk-ml

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Train

```bash
python -m credit_risk.train                # full run with hyper-parameter search
python -m credit_risk.train --no-search    # faster, baseline only
python -m credit_risk.train --help         # all options
```

Outputs: `models/credit_risk_model.joblib` and, in `reports/`, `metrics.json`, `confusion_matrix.png`, `feature_importance.png`, `feature_importance.csv`, `permutation_importance.csv`.

### Predict

```bash
python -m credit_risk.predict                       # built-in sample customer
python -m credit_risk.predict --input customer.json
```

Example output:

```json
{
  "default_probability": 0.79,
  "prediction": "Default",
  "risk": "HIGH RISK"
}
```

Or from Python:

```python
from credit_risk.predict import load_model, predict

model = load_model()
predict(model, {
    "income": 55000, "credit_score": 680, "loan_amount": 25000,
    "debt_ratio": 0.35, "employment_years": 6, "age": 35,
    "previous_defaults": 0, "account_balance": 15000,
    "employment_type": "Salaried", "loan_type": "Personal",
})
```

## Using real data

Replace `generate_synthetic_data()` in `src/credit_risk/data.py` with a `pd.read_csv(...)` call that returns the columns listed in `config.FEATURES` plus the `default` target.

## Security note

Models are saved with `joblib` (pickle). Only load model files from sources you trust.

## Roadmap

- [ ] FastAPI inference service
- [ ] Dockerfile
- [ ] Unit tests and CI
- [ ] Decision-threshold tuning and probability calibration

## License

MIT, see [LICENSE](LICENSE).
