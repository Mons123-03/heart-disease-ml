import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config as C
from src.data import load_data, split_data
from src.evaluate import choose_threshold, metrics_at
from src.features import FeatureEngineer, make_pipeline
from src.models import build_estimator, build_stack


@pytest.fixture(scope="module")
def data():
    df, report = load_data()
    return df, report


def test_cleaning(data):
    df, report = data
    assert not df.duplicated().any()
    assert set(C.RAW_FEATURES + [C.TARGET]) <= set(df.columns)
    assert df["thal"].isna().sum() == 2 and (df["thal"] == 0).sum() == 0
    assert (df["ca"] == 4).sum() == 0


def test_label_polarity_is_clinically_plausible(data):
    df, report = data
    assert all(report["polarity_checks"].values())
    corr = df[C.RAW_FEATURES + [C.TARGET]].corr()[C.TARGET]
    assert corr["exang"] > 0 and corr["thalach"] < 0


def test_split_is_stratified_and_disjoint(data):
    df, _ = data
    Xtr, Xte, ytr, yte = split_data(df)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert abs(ytr.mean() - yte.mean()) < 0.03


def test_feature_engineer_is_stateless_and_adds_columns(data):
    df, _ = data
    X = df[C.RAW_FEATURES]
    fe = FeatureEngineer().fit(X)
    out = fe.transform(X)
    assert list(out.columns) == list(fe.get_feature_names_out())
    single = fe.transform(X.iloc[[0]])          # row-wise result must not depend on other rows
    pd.testing.assert_frame_equal(single, out.iloc[[0]])
    assert (out["hr_pct_max"] > 0).all()


def test_pipeline_learns_signal_and_handles_missing(data):
    df, _ = data
    Xtr, Xte, ytr, yte = split_data(df)
    pipe = make_pipeline(build_estimator("logreg")).fit(Xtr, ytr)
    assert roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]) > 0.80
    row = Xte.iloc[[0]].copy(); row["thal"] = np.nan; row["ca"] = np.nan
    assert pipe.predict_proba(row).shape == (1, 2)


def test_stack_builds_and_fits(data):
    df, _ = data
    Xtr, Xte, ytr, yte = split_data(df)
    stack = build_stack({}).fit(Xtr, ytr)
    assert roc_auc_score(yte, stack.predict_proba(Xte)[:, 1]) > 0.80


def test_threshold_prefers_recall():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.05, 0.6, 50), np.linspace(0.3, 0.95, 50)])
    t = choose_threshold(y, p, beta=2.0)
    assert t < 0.5
    m = metrics_at(y, p, t)
    assert m["recall_sensitivity"] >= metrics_at(y, p, 0.5)["recall_sensitivity"]


@pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="model not trained yet")
def test_saved_model_inference():
    from src.predict import load_artifact, predict_dataframe
    art = load_artifact()
    df, _ = load_data(clean=False)
    out = predict_dataframe(df.head(10), art)
    assert out["risk_probability"].between(0, 1).all()
    assert set(out["prediction"]) <= {0, 1}
