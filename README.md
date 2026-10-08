# Heart Disease Prediction: Advanced Machine Learning Project

An end-to-end, leakage-safe ML pipeline that predicts heart disease from 13 routine clinical measurements,
with Bayesian hyper-parameter tuning, ensembles, probability calibration, a recall-weighted decision threshold,
SHAP explainability, a Streamlit app, tests, and Docker packaging.

> Educational prototype. Not a medical device; it does not replace a clinician.

## What makes it "advanced"

| Area | Technique |
|---|---|
| Data integrity | Duplicate removal, invalid-code handling (`thal=0`, `ca=4` -> missing), **label-polarity verification** against clinical expectations |
| Features | Clinically motivated engineered features (heart-rate reserve, % of age-predicted max HR, ST-depression x angina, hypertension / cholesterol bands) |
| Leakage control | Feature engineering, imputation, scaling and encoding all live **inside** one sklearn `Pipeline`, so they are re-fit in every CV fold |
| Model zoo | Logistic Regression, SVM, Random Forest, XGBoost, LightGBM |
| Tuning | **Optuna (TPE)** Bayesian search, objective = cross-validated ROC-AUC on the training set only |
| Ensembles | Soft-voting and **stacking** (out-of-fold probabilities -> logistic-regression meta learner) |
| Validation | Repeated stratified 5-fold CV for model selection; test set touched **once** at the end |
| Calibration | Sigmoid calibration (`CalibratedClassifierCV`) so outputs behave like probabilities |
| Decision rule | Threshold chosen on out-of-fold train predictions to maximise **F2** (recall-weighted screening objective) |
| Uncertainty | Bootstrap 95% confidence intervals on test metrics |
| Explainability | Global SHAP (beeswarm + importance) and fast per-patient Shapley contributions in the app |
| Engineering | Package layout, CLI, 8 unit tests, Streamlit app, Dockerfile, Makefile |

## Project layout

```
heart-disease-ml/
  src/  config.py data.py features.py models.py evaluate.py explain.py train.py predict.py
  app/streamlit_app.py
  tests/test_pipeline.py
  data/heart.csv            (downloaded automatically if missing)
  models/heart_model.joblib (created by training)
  reports/                  metrics.json, cv_results.csv, shap_importance.csv, figures/
```

## Quick start

```bash
# 1. (recommended) isolated environment, so your other projects are not affected
python -m venv venv
venv\Scripts\activate            # Windows   |   macOS/Linux: source venv/bin/activate

# 2. install
pip install -r requirements.txt
pip uninstall -y pyarrow           # optional, see Troubleshooting (needed only if Windows blocks it)

# 3. TRAIN FIRST (creates models/heart_model.joblib, ~5-10 min)
python -m src.train --trials 25 --repeats 3

# 4. use it
python -m pytest -q
python -m streamlit run app/streamlit_app.py
python -m src.predict --json "{\"age\":63,\"sex\":1,\"cp\":0,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":0,\"thalach\":120,\"exang\":1,\"oldpeak\":2.3,\"slope\":0,\"ca\":2,\"thal\":3}"
```
The trained model is **not** shipped in the zip: pickled models only load reliably with the same library
versions that created them, so each machine trains its own (the `reports/` folder shows the author's run).
Docker: `docker build -t heart-ml . && docker run -p 8501:8501 heart-ml`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `DLL load failed ... Application Control policy has blocked this file` (pyarrow) | Windows Smart App Control / WDAC blocks pyarrow's DLL. This project does not need pyarrow: run `pip uninstall -y pyarrow`. pip may warn that streamlit wants it; ignore the warning, the app avoids Arrow-based widgets. |
| `'pytest' is not recognized` / `'streamlit' is not recognized` | The user Scripts folder is not on PATH. Use `python -m pytest` and `python -m streamlit run ...` instead. |
| Installing broke other packages (numpy, protobuf, tensorflow) | Use the virtual environment from step 1 rather than installing globally. |
| App says "No trained model found" | Run `python -m src.train` first. |
| `ModuleNotFoundError` for xgboost / lightgbm / shap | Re-run `pip install -r requirements.txt` inside the activated environment. |

## Dataset

UCI Heart Disease (Cleveland): 303 rows, 13 features + target (302 after removing 1 duplicate).

Two things the code handles deliberately, because most tutorials get them wrong:

1. **The `target` column in the popular CSV is inverted** (1 = *no* disease). This is verified in
   `data.check_label_polarity`: after flipping, exercise angina, ST depression and vessel count correlate positively
   with disease and max heart rate negatively, as clinical knowledge says they should. A unit test guards this.
2. **Placeholder codes**: `thal=0` (2 rows) and `ca=4` (4 rows) are not valid categories and are treated as missing.
   Category labels for `cp`, `thal`, `slope`, `restecg` follow the UCI documentation, matched by category counts.

## Results (training run included in `reports/`)

Cross-validated ROC-AUC on the training set (241 patients, 5-fold x 2 repeats). Tuning improved every model
over its default, but the top models are **statistically indistinguishable** (differences are far smaller than the
fold-to-fold standard deviation of about 0.03-0.04):

| Model | CV ROC-AUC |
|---|---|
| **SVM (tuned), selected** | 0.912 +/- 0.033 |
| Voting ensemble | 0.911 +/- 0.037 |
| Stacking ensemble | 0.910 +/- 0.038 |
| Logistic regression (tuned) | 0.908 +/- 0.041 |

Held-out test set (61 patients), calibrated SVM:

| Operating point | Accuracy | Recall | Specificity | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Threshold 0.50 | 0.84 | 0.75 | 0.91 | 0.88 | 0.92 |
| Tuned threshold 0.18 (F2) | 0.77 | **1.00** | 0.58 | 0.67 | 0.92 |

The tuned threshold catches all 28 diseased test patients but raises 14 false alarms among 33 healthy ones. That
trade-off is the point of a screening tool; choose the operating point with clinicians.

Most influential features (SHAP): `thal`, `ca`, `cp`, `slope`, `thalach`, `oldpeak`, `exang`.

## Honest limitations

- **Small test set.** 61 patients means wide intervals (test ROC-AUC 95% CI about 0.86-0.97). Recall = 1.00 on 28
  positives has a bootstrap interval of [1, 1], which is a bootstrap artefact; an exact interval has a lower bound
  near 0.88. Do not read 100% as a guarantee.
- **Single-site, 1980s data**, mostly men, which will not generalise to modern or other populations without external validation.
- Selecting the final model from near-tied candidates is within noise; the SVM won by a hair.
- The deployed artifact is re-fit on all 302 rows; reported test numbers come from the train-only model.
- Correlation, not causation: SHAP explains the model, not the physiology.

## Extending it

External validation on the Statlog / Hungarian / Switzerland / VA sets; nested CV for fully unbiased
tuning estimates; fairness audit by sex; a FastAPI service; MLflow experiment tracking; conformal prediction intervals.
