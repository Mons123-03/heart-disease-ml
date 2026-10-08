"""Model-agnostic SHAP explanations computed on the raw clinical features."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config as C


def make_explainer(model, X_background: pd.DataFrame, max_samples: int = 25) -> shap.Explainer:
    """Permutation explainer over the full pipeline, so attributions are per *raw* input feature."""
    cols = list(X_background.columns)

    def f(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=cols))[:, 1]

    masker = shap.maskers.Independent(X_background.to_numpy(dtype=float), max_samples=min(max_samples, len(X_background)))
    return shap.Explainer(f, masker, algorithm="permutation", feature_names=cols)


def global_explanation(model, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP values, save beeswarm + bar plots, return a mean-|SHAP| ranking."""
    explainer = make_explainer(model, X_background)
    sv = explainer(X_explain.to_numpy(dtype=float))
    sv.feature_names = [C.FRIENDLY.get(c, c) for c in X_explain.columns]
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(); shap.plots.beeswarm(sv, show=False, max_display=13)
    plt.title("SHAP summary: effect of each feature on predicted risk")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "07_shap_beeswarm.png", dpi=140); plt.close()

    plt.figure(); shap.plots.bar(sv, show=False, max_display=13)
    plt.title("Mean |SHAP value|")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "08_shap_importance.png", dpi=140); plt.close()

    rank = pd.DataFrame({"feature": X_explain.columns, "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
    return rank.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(model, X_background: pd.DataFrame, row: pd.DataFrame,
                      n_permutations: int = 200, seed: int = C.RANDOM_STATE) -> pd.Series:
    """Monte-Carlo (interventional) Shapley values for one patient.

    Estimates the same quantity as SHAP's permutation explainer, but evaluates every coalition in a single
    batched ``predict_proba`` call, so it takes ~0.2 s instead of ~10 s. Contributions are in probability units
    and sum (approximately) to  P(patient) - mean P(background).
    """
    rng = np.random.default_rng(seed)
    cols = list(row.columns)
    d = len(cols)
    x = row.to_numpy(dtype=float)[0]
    bg = X_background[cols].to_numpy(dtype=float)
    blocks, orders = [], []
    for _ in range(n_permutations):
        order = rng.permutation(d)
        z = bg[rng.integers(len(bg))].copy()          # start from a random background patient
        steps = [z.copy()]
        for j in order:                               # switch features to the patient's values one by one
            z[j] = x[j]
            steps.append(z.copy())
        blocks.append(np.vstack(steps)); orders.append(order)
    preds = model.predict_proba(pd.DataFrame(np.vstack(blocks), columns=cols))[:, 1].reshape(n_permutations, d + 1)
    contrib = np.zeros(d)
    for p, order in zip(preds, orders):
        contrib[order] += np.diff(p)
    return pd.Series(contrib / n_permutations, index=cols)
