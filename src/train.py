"""End-to-end training: EDA -> baselines -> Optuna tuning -> ensembles -> calibration -> evaluation -> SHAP -> export.

Run:  python -m src.train --trials 25
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from . import __version__
from . import config as C
from .data import load_data, split_data
from .evaluate import (bootstrap_ci, choose_threshold, cv_compare, metrics_at, plot_calibration,
                       plot_cv_comparison, plot_eda, plot_test_curves)
from .explain import global_explanation
from .features import make_pipeline
from .models import MODEL_NAMES, build_estimator, build_stack, build_voting, default_estimator, tune


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=C.N_TRIALS_DEFAULT, help="Optuna trials per model")
    ap.add_argument("--repeats", type=int, default=3, help="CV repeats for model comparison")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    for d in (C.MODELS_DIR, C.REPORTS_DIR, C.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    df, data_report = load_data()
    log(f"Loaded {data_report['rows']} rows after cleaning: {data_report}")
    plot_eda(df)
    X_train, X_test, y_train, y_test = split_data(df)
    log(f"Train {X_train.shape}, test {X_test.shape} (test set stays untouched until the final step)")

    # 2. baselines ----------------------------------------------------------
    baselines = {f"{n} (default)": make_pipeline(default_estimator(n)) for n in MODEL_NAMES}
    log("Cross-validating baselines ...")
    base_cv = cv_compare(baselines, X_train, y_train, args.repeats)

    # 3. Bayesian hyper-parameter tuning -------------------------------------
    tuned, tune_scores = {}, {}
    for n in MODEL_NAMES:
        t0 = time.time()
        tuned[n], tune_scores[n] = tune(n, X_train, y_train, args.trials)
        log(f"Tuned {n:7s} CV-AUC={tune_scores[n]:.4f} ({time.time() - t0:.0f}s)")

    # 4. tuned models + ensembles, compared on the training folds ---------------
    candidates = {f"{n} (tuned)": make_pipeline(build_estimator(n, tuned[n])) for n in MODEL_NAMES}
    candidates["voting ensemble"] = build_voting(tuned)
    candidates["stacking ensemble"] = build_stack(tuned)
    log("Cross-validating tuned models and ensembles ...")
    tuned_cv = cv_compare(candidates, X_train, y_train, args.repeats)
    cv_all = pd.concat([base_cv, tuned_cv]).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    cv_all.to_csv(C.REPORTS_DIR / "cv_results.csv", index=False)
    plot_cv_comparison(cv_all)
    print(cv_all[["model", "roc_auc_mean", "roc_auc_std", "accuracy_mean", "recall_mean", "f1_mean"]]
          .round(4).to_string(index=False))

    best_name = tuned_cv.iloc[0]["model"]
    best_pipe = candidates[best_name]
    log(f"Selected model by CV ROC-AUC: {best_name}")

    # 5. calibration + recall-weighted threshold (training data only) ---------
    calibrated = CalibratedClassifierCV(clone(best_pipe), method="sigmoid", cv=5)
    skf = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    oof = cross_val_predict(calibrated, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof)
    log(f"Decision threshold (F{C.F_BETA:g}-optimal on out-of-fold train predictions): {threshold}")

    # 6. final, one-shot evaluation on the held-out test set -------------------
    raw_model = clone(best_pipe).fit(X_train, y_train)
    cal_model = clone(calibrated).fit(X_train, y_train)
    p_raw = raw_model.predict_proba(X_test)[:, 1]
    p_cal = cal_model.predict_proba(X_test)[:, 1]
    results = {
        "at_0.50": metrics_at(y_test, p_cal, 0.5),
        "at_tuned_threshold": metrics_at(y_test, p_cal, threshold),
        "bootstrap_95ci_at_tuned_threshold": bootstrap_ci(y_test, p_cal, threshold),
        "brier_uncalibrated": float(metrics_at(y_test, p_raw)["brier"]),
        "brier_calibrated": float(metrics_at(y_test, p_cal)["brier"]),
    }
    plot_test_curves(y_test, p_cal, threshold, best_name)
    plot_calibration(y_test, {"uncalibrated": p_raw, "sigmoid-calibrated": p_cal})
    log("Test metrics @ tuned threshold: " + json.dumps(_jsonable(
        {k: round(v, 3) for k, v in results["at_tuned_threshold"].items() if isinstance(v, float)})))

    # 7. explainability --------------------------------------------------------
    shap_rank = None
    background = X_train.sample(min(60, len(X_train)), random_state=C.RANDOM_STATE)
    if not args.no_shap:
        log("Computing SHAP values ...")
        shap_rank = global_explanation(cal_model, background, X_test)
        shap_rank.to_csv(C.REPORTS_DIR / "shap_importance.csv", index=False)
        print(shap_rank.round(4).to_string(index=False))

    # 8. export: refit the chosen recipe on ALL data for deployment -------------
    X_all = pd.concat([X_train, X_test]); y_all = pd.concat([y_train, y_test])
    final_model = clone(calibrated).fit(X_all, y_all)
    metadata = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": best_name, "n_train_rows": int(len(X_all)), "threshold": threshold,
        "test_metrics_train_only_model": results["at_tuned_threshold"],
        "tuned_params": tuned,
    }
    joblib.dump({"model": final_model, "threshold": threshold, "features": C.RAW_FEATURES,
                 "background": background.to_numpy(dtype=float), "metadata": metadata}, C.MODEL_PATH)
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(_jsonable({
        "data": data_report, "selected_model": best_name, "tuning_cv_auc": tune_scores,
        "best_params": tuned, "test": results,
        "shap_top": None if shap_rank is None else shap_rank.head(8).to_dict("records")}), indent=2))
    log(f"Saved model -> {C.MODEL_PATH}  | reports -> {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
