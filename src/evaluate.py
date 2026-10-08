"""Metrics, threshold selection, bootstrap confidence intervals and plots."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from . import config as C


def cv_compare(models: dict, X, y, n_repeats: int = 3) -> pd.DataFrame:
    """Repeated stratified 5-fold CV on the training set for every model."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=C.RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "recall": "recall",
               "precision": "precision", "f1": "f1", "brier": "neg_brier_score"}
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for m in scoring:
            vals = res[f"test_{m}"] * (-1 if m == "brier" else 1)
            row[f"{m}_mean"], row[f"{m}_std"] = vals.mean(), vals.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, oof_proba, beta: float = C.F_BETA) -> float:
    """Pick the threshold maximising F-beta on out-of-fold training predictions (recall-weighted)."""
    y_true = np.asarray(y_true)
    best_t, best = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        p = (oof_proba >= t).astype(int)
        tp = ((p == 1) & (y_true == 1)).sum()
        fp = ((p == 1) & (y_true == 0)).sum()
        fn = ((p == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if prec + rec else 0
        if fb > best:
            best, best_t = fb, float(t)
    return round(best_t, 2)


def bootstrap_ci(y_true, proba, threshold: float, n: int = 1000, seed: int = C.RANDOM_STATE) -> dict:
    """Percentile 95% bootstrap confidence intervals on the held-out test set."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    keys = ["accuracy", "recall_sensitivity", "specificity", "precision", "f1", "roc_auc"]
    draws = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = metrics_at(y_true[idx], proba[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


# ---------------------------------------------------------------- plots
def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=140)
    plt.close(fig)


def plot_eda(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    df[C.TARGET].map({0: "No disease", 1: "Disease"}).value_counts().plot.bar(ax=ax, color=["#4c72b0", "#c44e52"])
    ax.set_title("Class balance"); ax.set_xlabel(""); plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, "01_class_balance.png")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[C.RAW_FEATURES + [C.TARGET]].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, "02_correlation.png")


def plot_cv_comparison(cv_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    d = cv_df.sort_values("roc_auc_mean")
    ax.barh(d["model"], d["roc_auc_mean"], xerr=d["roc_auc_std"], color="#4c72b0", capsize=3)
    ax.set_xlim(max(0.5, d["roc_auc_mean"].min() - 0.05), 1.0)
    ax.set_xlabel("Cross-validated ROC-AUC (train set)"); ax.set_title("Model comparison")
    _save(fig, "03_model_comparison.png")


def plot_test_curves(y_true, proba, threshold: float, label: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, proba):.3f})")
    axs[0].plot([0, 1], [0, 1], "k--", lw=1); axs[0].set_xlabel("False positive rate")
    axs[0].set_ylabel("True positive rate"); axs[0].set_title("ROC curve"); axs[0].legend()
    axs[1].plot(rec, prec); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[1].set_title(f"Precision-recall (AP={average_precision_score(y_true, proba):.3f})")
    _save(fig, "04_roc_pr.png")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, t, title in zip(axs, [0.5, threshold], ["Threshold 0.50", f"Tuned threshold {threshold:.2f}"]):
        cm = confusion_matrix(y_true, (proba >= t).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
        ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    _save(fig, "05_confusion_matrices.png")


def plot_calibration(y_true, probas: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probas.items():
        frac, mean = calibration_curve(y_true, p, n_bins=5, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive"); ax.set_title("Calibration (test set)"); ax.legend()
    _save(fig, "06_calibration.png")
