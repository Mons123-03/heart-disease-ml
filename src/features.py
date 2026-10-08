"""Feature engineering and preprocessing, all inside the sklearn pipeline (no leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds clinically motivated features. Stateless, so it is leakage-free by construction."""

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        age_pred_max_hr = 220 - X["age"]                      # age-predicted maximum heart rate
        X["hr_reserve"] = age_pred_max_hr - X["thalach"]     # unused chronotropic reserve
        X["hr_pct_max"] = X["thalach"] / age_pred_max_hr     # fraction of predicted max reached
        X["oldpeak_x_exang"] = X["oldpeak"] * X["exang"]     # ST depression under exertional angina
        X["high_chol"] = (X["chol"] >= 240).astype(int)      # "high" cholesterol band
        X["stage2_htn"] = (X["trestbps"] >= 140).astype(int) # stage 2 hypertension threshold
        X["age_x_oldpeak"] = X["age"] * X["oldpeak"] / 100
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.feature_names_in_) + C.ENGINEERED)


def build_preprocessor() -> ColumnTransformer:
    numeric = C.NUMERIC + ["hr_reserve", "hr_pct_max", "oldpeak_x_exang", "age_x_oldpeak"]
    passthrough = C.BINARY + ["high_chol", "stage2_htn"]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             C.CATEGORICAL),
            ("bin", "passthrough", passthrough),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(clf) -> Pipeline:
    """Full model: feature engineering -> preprocessing -> classifier."""
    return Pipeline([("fe", FeatureEngineer()), ("pre", build_preprocessor()), ("clf", clf)])
