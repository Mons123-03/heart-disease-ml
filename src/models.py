"""Model zoo, Optuna hyper-parameter search spaces, and ensembles."""
from __future__ import annotations

import warnings

import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config as C
from .features import make_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
# sklearn 1.8 deprecates LogisticRegression(penalty=...); kept for compatibility with older versions.
warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)

SEED = C.RANDOM_STATE


def default_estimator(name: str):
    """Baseline (untuned) estimators."""
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == "svm":
        return SVC(probability=True, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1)
    if name == "xgb":
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric="logloss",
                             random_state=SEED, n_jobs=1, verbosity=0)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=10,
                              random_state=SEED, n_jobs=1, verbose=-1)
    raise ValueError(name)


MODEL_NAMES = ["logreg", "svm", "rf", "xgb", "lgbm"]


def _suggest(trial: optuna.Trial, name: str) -> dict:
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"])}
    if name == "svm":
        return {"C": trial.suggest_float("C", 1e-2, 100, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"])}
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])}
    if name == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 80, 400, step=40),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20, log=True)}
    if name == "lgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 80, 400, step=40),
                "num_leaves": trial.suggest_int("num_leaves", 4, 24),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20, log=True)}
    raise ValueError(name)


def build_estimator(name: str, params: dict | None = None):
    """Default estimator with (optionally) tuned parameters applied."""
    est = default_estimator(name)
    if params:
        p = dict(params)
        if name == "logreg" and p.get("penalty") == "l1":
            est.set_params(solver="liblinear")
        est.set_params(**p)
    return est


def tune(name: str, X, y, n_trials: int = C.N_TRIALS_DEFAULT) -> tuple[dict, float]:
    """Bayesian (TPE) search maximising cross-validated ROC-AUC on the training set only."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)

    def objective(trial):
        est = build_estimator(name, _suggest(trial, name))
        return cross_val_score(make_pipeline(est), X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


def build_stack(tuned: dict[str, dict]):
    """Stacking ensemble: tuned base learners + logistic-regression meta learner on out-of-fold probabilities."""
    base = [(n, build_estimator(n, tuned.get(n))) for n in ("svm", "rf", "xgb", "lgbm", "logreg")]
    stack = StackingClassifier(estimators=base,
                               final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
                               stack_method="predict_proba", cv=5, n_jobs=1)
    return make_pipeline(stack)


def build_voting(tuned: dict[str, dict]):
    """Soft-voting ensemble over the tuned base learners."""
    base = [(n, build_estimator(n, tuned.get(n))) for n in ("svm", "rf", "xgb", "lgbm", "logreg")]
    return make_pipeline(VotingClassifier(estimators=base, voting="soft", n_jobs=1))
