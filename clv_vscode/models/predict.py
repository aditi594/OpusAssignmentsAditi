"""
models/predict.py
-----------------
Inference helpers — single prediction and batch prediction.

FINAL SAFE VERSION:
- CLV prediction using reg_best.pkl
- Segment prediction using CLV thresholds
- CLV-only KMeans clustering (kmeans_k3.pkl)
- Prevents PCA / feature mismatch errors
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────
# Paths and cache
# ─────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
_cache = {}

def _load(name):
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {name}")
    if name not in _cache:
        _cache[name] = joblib.load(path)
    return _cache[name]

# ─────────────────────────────────────
# Single CLV prediction
# ─────────────────────────────────────

def predict_clv(input_dict: dict) -> float:
    """
    Predict CLV for a single customer.
    """

    model    = _load("reg_best.pkl")
    scaler   = _load("scaler.pkl")
    features = _load("feature_names.pkl")
    le_inc   = _load("le_income.pkl")
    le_ch    = _load("le_channel.pkl")

    row = {
        "Age":          input_dict["Age"],
        "IncomeValue":  input_dict["IncomeValue"],
        "Income_enc":   le_inc.transform([input_dict["Income"]])[0],
        "Channel_enc":  le_ch.transform([input_dict["Channel"]])[0],
        "Tenure":       input_dict["Tenure"],
        "Frequency":    input_dict["Frequency"],
        "AvgSpend":     input_dict["AvgSpend"],
        "MonthlySpend": input_dict["MonthlySpend"],
        "Recency":      input_dict["Recency"],
        "RFM_Score":    input_dict["RFM_Score"],
    }

    X = pd.DataFrame([row])[features]
    X_scaled = scaler.transform(X)

    return float(model.predict(X_scaled)[0])

# ─────────────────────────────────────
# Segment prediction (rule-based)
# ─────────────────────────────────────

def predict_segment(clv_value: float) -> str:
    """
    Predict segment using CLV thresholds.
    """

    thresh = _load("clv_thresholds.pkl")

    if clv_value >= thresh["p66"]:
        return "High"
    if clv_value >= thresh["p33"]:
        return "Medium"
    return "Low"

# ─────────────────────────────────────
# Batch prediction (CSV upload)
# ─────────────────────────────────────

def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch prediction for uploaded CSV:
    - CLV prediction
    - Segment prediction
    - CLV-only clustering (safe)
    """

    # Load artifacts
    model    = _load("reg_best.pkl")
    scaler   = _load("scaler.pkl")
    features = _load("feature_names.pkl")
    le_inc   = _load("le_income.pkl")
    le_ch    = _load("le_channel.pkl")
    km       = _load("kmeans_k3.pkl")        # CLV-only KMeans
    thresh   = _load("clv_thresholds.pkl")

    # Safety guard: ensure correct model
    if km.n_features_in_ != 1:
        raise RuntimeError(
            "Wrong KMeans model loaded. Expected CLV-only model with 1 feature."
        )

    df = df.copy()

    # ── Encode categorical columns ─────────────────
    df["Income_enc"]  = le_inc.transform(df["Income"])
    df["Channel_enc"] = le_ch.transform(df["Channel"])

    # ── CLV prediction ─────────────────────────────
    X = df[features]
    X_scaled = scaler.transform(X)

    df["CLV_Predicted"] = model.predict(X_scaled)

    # ──  CLV-only clustering (FINAL, SAFE) ───────
    clv_scaled = StandardScaler().fit_transform(
    df[["CLV_Predicted"]]
    ).astype("float32")

    df["Cluster"] = km.predict(clv_scaled)

    # ── Segment prediction ─────────────────────────
    def assign_segment(clv):
        if clv >= thresh["p66"]:
            return "High"
        if clv >= thresh["p33"]:
            return "Medium"
        return "Low"

    df["Segment_Predicted"] = df["CLV_Predicted"].apply(assign_segment)

    return df