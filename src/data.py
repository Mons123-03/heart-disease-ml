"""Data acquisition, validation, cleaning and splitting."""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def download_data(force: bool = False) -> None:
    """Download the dataset from the first reachable mirror."""
    if C.DATA_PATH.exists() and not force:
        return
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in C.DATA_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                C.DATA_PATH.write_bytes(r.read())
            df = pd.read_csv(C.DATA_PATH, encoding="utf-8-sig")
            _validate_schema(df)
            return
        except Exception as e:  # try next mirror
            last_err = e
    raise RuntimeError(f"Could not download dataset from any mirror: {last_err}")


def _validate_schema(df: pd.DataFrame) -> None:
    expected = set(C.RAW_FEATURES + [C.TARGET_RAW])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")


def check_label_polarity(df: pd.DataFrame) -> dict:
    """Verify the target polarity using well-established clinical associations.

    With 'disease' = 1, exercise angina, ST depression and vessel count should
    correlate positively with disease, and maximum heart rate negatively.
    """
    corr = df[C.RAW_FEATURES + [C.TARGET]].corr()[C.TARGET]
    checks = {"exang": corr["exang"] > 0, "oldpeak": corr["oldpeak"] > 0,
              "ca": corr["ca"] > 0, "thalach": corr["thalach"] < 0}
    if not all(checks.values()):
        raise ValueError(f"Label polarity looks wrong: {checks}")
    return {k: bool(v) for k, v in checks.items()}


def load_data(clean: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load the CSV and return (dataframe, cleaning_report)."""
    download_data()
    df = pd.read_csv(C.DATA_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    _validate_schema(df)
    report = {"rows_raw": int(len(df))}

    df[C.TARGET] = (1 - df[C.TARGET_RAW]) if C.LABEL_INVERTED else df[C.TARGET_RAW]
    df = df.drop(columns=[C.TARGET_RAW])

    if clean:
        report["duplicates_removed"] = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        n_invalid = 0
        for col, bad in C.INVALID_VALUES.items():
            mask = df[col].isin(bad)
            n_invalid += int(mask.sum())
            df.loc[mask, col] = np.nan
        report["invalid_codes_to_nan"] = n_invalid

    report["rows"] = int(len(df))
    report["class_balance"] = df[C.TARGET].value_counts(normalize=True).round(3).to_dict()
    report["polarity_checks"] = check_label_polarity(df)
    return df, report


def split_data(df: pd.DataFrame):
    """Stratified train/test split. The test set is touched exactly once, at the end."""
    X = df[C.RAW_FEATURES]
    y = df[C.TARGET]
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
