"""Streamlit decision-support demo.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config as C
from src.explain import local_explanation
from src.predict import load_artifact, predict_one

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")


@st.cache_resource
def get_artifact():
    return load_artifact()


@st.cache_data(show_spinner=False)
def cached_explain(items: tuple):
    row = pd.DataFrame([dict(items)])[get_artifact()["features"]]
    return local_explanation(get_artifact()["model"], get_artifact()["background"], row)


if not C.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

art = get_artifact()
meta = art["metadata"]

st.title("❤️ Heart Disease Risk Estimator")
st.caption("Educational decision-support prototype. It is not a medical device and does not replace a clinician.")

with st.sidebar:
    st.header("Patient data")
    sel = lambda col, default: st.selectbox(
        C.FRIENDLY[col], list(C.CATEGORY_LABELS[col]), index=list(C.CATEGORY_LABELS[col]).index(default),
        format_func=lambda v, c=col: C.CATEGORY_LABELS[c][v])
    rec = {
        "age": st.slider(C.FRIENDLY["age"], 20, 90, 55),
        "sex": sel("sex", 1),
        "cp": sel("cp", 0),
        "trestbps": st.slider(C.FRIENDLY["trestbps"], 80, 220, 130),
        "chol": st.slider(C.FRIENDLY["chol"], 100, 600, 240),
        "fbs": sel("fbs", 0),
        "restecg": sel("restecg", 0),
        "thalach": st.slider(C.FRIENDLY["thalach"], 60, 220, 150),
        "exang": sel("exang", 0),
        "oldpeak": st.slider(C.FRIENDLY["oldpeak"], 0.0, 7.0, 1.0, 0.1),
        "slope": sel("slope", 2),
        "ca": st.selectbox(C.FRIENDLY["ca"], [0, 1, 2, 3]),
        "thal": sel("thal", 2),
    }

tab_pred, tab_model = st.tabs(["Prediction", "About the model"])

with tab_pred:
    result = predict_one(rec, art)
    p = result["risk_probability"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated risk", f"{p:.0%}")
    c2.metric("Risk band", result["risk_band"])
    c3.metric("Screening decision", "Refer for further tests" if result["prediction"] else "Low concern",
              help=f"Positive when probability >= {art['threshold']:.2f} (recall-weighted threshold).")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"The risk band is a coarse guide (<33% / 33-66% / >66%). The screening decision uses the recall-weighted threshold of {art['threshold']:.2f}, chosen so that sick patients are rarely missed, at the cost of more false alarms.")

    st.subheader("Why this estimate?")
    with st.spinner("Computing SHAP explanation ..."):
        contrib = cached_explain(tuple(rec.items()))
    contrib = contrib.rename(index=C.FRIENDLY).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))   # matplotlib instead of st.bar_chart: no pyarrow needed
    ax.barh(contrib.index, contrib.values, color=["#c44e52" if v > 0 else "#4c72b0" for v in contrib.values])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Change in estimated probability")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Bars to the right raise the estimated risk, bars to the left lower it (change in probability).")

with tab_model:
    st.write(f"**Model:** {meta['model']}  |  **Trained:** {meta['trained_at']}  |  **Rows:** {meta['n_train_rows']}")
    m = meta["test_metrics_train_only_model"]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("ROC-AUC", "roc_auc"), ("Recall", "recall_sensitivity"),
                                        ("Specificity", "specificity"), ("Precision", "precision"),
                                        ("Accuracy", "accuracy")]):
        col.metric(label, f"{m[key]:.2f}")
    st.caption("Held-out test metrics (61 patients, so intervals are wide; see reports/metrics.json).")
    for fig in ["03_model_comparison.png", "04_roc_pr.png", "07_shap_beeswarm.png"]:
        if (C.FIG_DIR / fig).exists():
            st.image(str(C.FIG_DIR / fig))
