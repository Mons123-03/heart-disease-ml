"""Central configuration: paths, schema, label metadata and global settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DATA_PATH = DATA_DIR / "heart.csv"
MODEL_PATH = MODELS_DIR / "heart_model.joblib"

# Public mirrors of the UCI Cleveland data (first reachable one is used).
DATA_URLS = [
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv",
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_RAW = "target"      # column as shipped in the CSV
TARGET = "disease"         # 1 = heart disease present, 0 = absent
# The widely distributed CSV encodes target=1 as *healthy*. See data.check_label_polarity,
# which verifies this against clinical expectations instead of trusting the file.
LABEL_INVERTED = True

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["cp", "restecg", "slope", "thal", "ca"]
BINARY = ["sex", "fbs", "exang"]
RAW_FEATURES = NUMERIC + CATEGORICAL + BINARY
ENGINEERED = ["hr_reserve", "hr_pct_max", "oldpeak_x_exang", "high_chol", "stage2_htn", "age_x_oldpeak"]

# Values that are placeholders for "missing" in this CSV.
INVALID_VALUES = {"thal": [0], "ca": [4]}

FRIENDLY = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST segment slope",
    "ca": "Major vessels coloured (0-3)",
    "thal": "Thalassemia (thal)",
}

# Category labels, derived by matching category counts and disease rates to the UCI documentation.
CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Asymptomatic", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "Left ventricular hypertrophy", 2: "ST-T wave abnormality"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Downsloping", 1: "Flat", 2: "Upsloping"},
    "thal": {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"},
}

# Recall-weighted screening objective: missing a sick patient costs more than a false alarm.
F_BETA = 2.0
N_TRIALS_DEFAULT = 25
