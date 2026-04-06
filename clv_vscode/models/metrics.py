# models/metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd


def compute_mae(
    df: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    feature_list_path: str,
    target_col: str = "CLV"
):
    """
    Computes Mean Absolute Error (MAE) for CLV regression model.

    Parameters:
    - df: DataFrame with true CLV values
    - model_path: Path to trained regression model
    - scaler_path: Path to scaler
    - feature_list_path: Path to feature names list
    - target_col: Target column name (default: CLV)

    Returns:
    - float MAE value
    """

    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_list = joblib.load(feature_list_path)

    # Prepare features & target
    X = df[feature_list]
    y_true = df[target_col]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict CLV
    y_pred = model.predict(X_scaled)

    # Compute MAE
    mae = mean_absolute_error(y_true, y_pred)

    return round(mae, 2)
def get_regression_metrics(df):
    return {
        "MAE": compute_mae(
            df,
            model_path="models/artifacts/reg_best.pkl",
            scaler_path="models/artifacts/scaler.pkl",
            feature_list_path="models/artifacts/feature_names.pkl"
        )
    }
