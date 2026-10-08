"""Inference helpers used by the CLI and the Streamlit app."""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from . import config as C


def load_artifact(path=C.MODEL_PATH) -> dict:
    art = joblib.load(path)
    bg = art.get("background")
    if not isinstance(bg, pd.DataFrame):   # stored as a plain array so the file has no pandas/pyarrow dependency
        art["background"] = pd.DataFrame(bg, columns=art["features"])
    return art


def risk_band(p: float) -> str:
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def predict_dataframe(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Return probability, thresholded label and risk band for each row of raw features."""
    art = artifact or load_artifact()
    X = df[art["features"]].copy()
    for col, bad in C.INVALID_VALUES.items():       # same cleaning as training
        if col in X:
            X.loc[X[col].isin(bad), col] = float("nan")
    proba = art["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "risk_probability": proba.round(4),
        "prediction": (proba >= art["threshold"]).astype(int),
        "risk_band": [risk_band(p) for p in proba],
    }, index=df.index)


def predict_one(record: dict, artifact: dict | None = None) -> dict:
    out = predict_dataframe(pd.DataFrame([record]), artifact).iloc[0]
    return {"risk_probability": float(out.risk_probability), "prediction": int(out.prediction),
            "risk_band": out.risk_band}


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict heart disease risk from a CSV or JSON record.")
    ap.add_argument("--csv", help="CSV with the 13 raw feature columns")
    ap.add_argument("--json", help="JSON string with one patient record")
    a = ap.parse_args()
    if a.csv:
        print(predict_dataframe(pd.read_csv(a.csv)).to_string())
    elif a.json:
        print(json.dumps(predict_one(json.loads(a.json)), indent=2))
    else:
        ap.error("provide --csv or --json")


if __name__ == "__main__":
    main()
