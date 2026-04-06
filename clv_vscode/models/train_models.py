"""
models/train_models.py
----------------------
Trains all ML models:
  - CLV Regression (Linear, Ridge, RF, XGBoost)
  - Clustering:
      * Behavioral clustering (K-Means + GMM + PCA)   FINAL
      * CLV-only clustering (K-Means)                 EXPERIMENTAL
  - Decision Tree classifier (CLV → Segment)

Run:
  python models/train_models.py
"""

import os, json, joblib, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────
# Optional XGBoost
# ─────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    XGB_OK = False
    print("⚠ xgboost not installed — using GradientBoosting fallback")

# ─────────────────────────────────────
# Paths
# ─────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
DATA_PATH = os.path.join(HERE, "..", "data", "customers.csv")
os.makedirs(ARTIFACTS, exist_ok=True)

# ─────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────

# Regression features
REG_FEATURES = [
    "Age","IncomeValue","Income_enc","Channel_enc",
    "Tenure","Frequency","AvgSpend","MonthlySpend","Recency","RFM_Score"
]

#  FINAL: Behavioral clustering features
CLUSTER_FEATURES_BEHAVIORAL = [
    "Frequency","AvgSpend","Tenure","Recency","RFM_Score"
]

#EXPERIMENTAL: CLV-only clustering
CLUSTER_FEATURES_CLV_ONLY = ["CLV"]

# ─────────────────────────────────────
# Utilities
# ─────────────────────────────────────

def adjusted_r2_score(r2, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# ─────────────────────────────────────
# Load & preprocess
# ─────────────────────────────────────

def load_and_prepare():
    print("Loading data")
    df = pd.read_csv(DATA_PATH)

    le_inc = LabelEncoder().fit(df["Income"])
    le_ch  = LabelEncoder().fit(df["Channel"])
    le_seg = LabelEncoder().fit(df["Segment"])

    df["Income_enc"]  = le_inc.transform(df["Income"])
    df["Channel_enc"] = le_ch.transform(df["Channel"])
    df["Segment_enc"] = le_seg.transform(df["Segment"])

    joblib.dump(le_inc, os.path.join(ARTIFACTS, "le_income.pkl"))
    joblib.dump(le_ch,  os.path.join(ARTIFACTS, "le_channel.pkl"))
    joblib.dump(le_seg, os.path.join(ARTIFACTS, "le_segment.pkl"))
    joblib.dump(REG_FEATURES, os.path.join(ARTIFACTS, "feature_names.pkl"))

    return df

# ─────────────────────────────────────
# Regression
# ─────────────────────────────────────

def train_regression(X_tr, X_te, y_tr, y_te):
    print("\nCLV Regression Models")

    boost = (
        xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ) if XGB_OK else
        _GBR(n_estimators=200, max_depth=6, learning_rate=0.08)
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=5,
            random_state=42
        ),
        "XGBoost": boost
    }

    results = {}
    best_rmse = float("inf")
    best_model = None

    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2   = r2_score(y_te, preds)
        adj_r2 = adjusted_r2_score(r2, len(y_te), X_tr.shape[1])

        results[name] = {
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "Adj_R2": round(float(adj_r2), 4)
        }

        joblib.dump(m, os.path.join(ARTIFACTS, f"reg_{name}.pkl"))
        print(f"  {name:<15} RMSE={rmse:.2f}  R²={r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = name

    joblib.dump(
        joblib.load(os.path.join(ARTIFACTS, f"reg_{best_model}.pkl")),
        os.path.join(ARTIFACTS, "reg_best.pkl")
    )

    results["best"] = best_model
    print(f"Best Regression Model: {best_model}")
    return results

# ─────────────────────────────────────
#  FINAL: Behavioral clustering (PCA)
# ─────────────────────────────────────

def prepare_behavioral_cluster_data(df):
    X = df[CLUSTER_FEATURES_BEHAVIORAL]
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=3, random_state=42).fit_transform(X_scaled)
    return X_pca

def train_kmeans_behavioral(X_pca):
    print("\nK-Means (Behavioral Features)")
    scores = {}

    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_pca)
        scores[k] = round(silhouette_score(X_pca, labels), 4)
        print(f"  k={k} → silhouette={scores[k]}")

    joblib.dump(
        KMeans(n_clusters=3, n_init=20, random_state=42).fit(X_pca),
        os.path.join(ARTIFACTS, "kmeans_behavioral_k3.pkl")
    )

    return scores

def train_gmm_behavioral(X_pca):
    print("\nGMM (Behavioral Features)")
    scores = {}

    for k in range(2, 7):
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(X_pca)
        scores[k] = round(silhouette_score(X_pca, labels), 4)
        print(f"  k={k} → silhouette={scores[k]}")

    joblib.dump(
        GaussianMixture(n_components=3, random_state=42).fit(X_pca),
        os.path.join(ARTIFACTS, "gmm_behavioral_k3.pkl")
    )

    return scores

# ─────────────────────────────────────
#  EXPERIMENTAL: CLV-only clustering
# ─────────────────────────────────────

def train_kmeans_clv_only(df):
    print("\n K-Means (CLV ONLY – Illustrative)")
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES_CLV_ONLY])

    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        scores[k] = round(silhouette_score(X, labels), 4)
        print(f"  k={k} → silhouette={scores[k]}")

    return scores

# ─────────────────────────────────────
# Classification
# ─────────────────────────────────────

def train_classifier(df):
    print("\n Segment Classifier (CLV-based)")

    X = df[["CLV"]]
    y = df["Segment"]

    y_enc = LabelEncoder().fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2,
        random_state=42, stratify=y_enc
    )

    dt = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )

    dt.fit(X_tr, y_tr)
    preds = dt.predict(X_te)

    acc = classification_report(y_te, preds, output_dict=True)["accuracy"]
    print(f"Accuracy: {acc:.4f}")

    joblib.dump(dt, os.path.join(ARTIFACTS, "dt_classifier.pkl"))
    return {"accuracy": round(acc, 4)}

# ─────────────────────────────────────
# MAIN
# ─────────────────────────────────────

def main():
    df = load_and_prepare()

    # Regression
    X = StandardScaler().fit_transform(df[REG_FEATURES])
    y = df["CLV"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_results = train_regression(X_tr, X_te, y_tr, y_te)

    #  FINAL clustering
    X_pca = prepare_behavioral_cluster_data(df)
    sil_km_behavioral  = train_kmeans_behavioral(X_pca)
    sil_gmm_behavioral = train_gmm_behavioral(X_pca)

    #  Experimental clustering
    sil_km_clv_only = train_kmeans_clv_only(df)

    # Classification
    cls_results = train_classifier(df)

    metrics = {
        "regression": reg_results,
        "clustering_silhouette": {
            "behavioral_kmeans": {str(k): v for k, v in sil_km_behavioral.items()},
            "behavioral_gmm": {str(k): v for k, v in sil_gmm_behavioral.items()},
            "clv_only_kmeans": {str(k): v for k, v in sil_km_clv_only.items()}
        },
        "classification": cls_results
    }

    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll artifacts saved successfully")

if __name__ == "__main__":
    main()
# """
# models/train_models.py
# ----------------------
# Trains all ML models:
#   - Linear Regression, Ridge, Random Forest, XGBoost (CLV regression)
#   - K-Means clustering (behavioral features + PCA)
#   - Decision Tree classifier (segment prediction from CLV)

# Run:  python models/train_models.py
# """

# import os, json, joblib, warnings
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, classification_report

# warnings.filterwarnings("ignore")

# # ─────────────────────────────────────────────────────────────
# # Optional XGBoost
# # ─────────────────────────────────────────────────────────────
# try:
#     import xgboost as xgb
#     XGB_OK = True
# except ImportError:
#     from sklearn.ensemble import GradientBoostingRegressor as _GBR
#     XGB_OK = False
#     print("⚠ xgboost not installed — using sklearn GradientBoosting fallback")

# # ─────────────────────────────────────────────────────────────
# # Paths
# # ─────────────────────────────────────────────────────────────
# HERE      = os.path.dirname(os.path.abspath(__file__))
# ARTIFACTS = os.path.join(HERE, "artifacts")
# DATA_PATH = os.path.join(HERE, "..", "data", "customers.csv")
# os.makedirs(ARTIFACTS, exist_ok=True)

# # ─────────────────────────────────────────────────────────────
# # Feature definitions
# # ─────────────────────────────────────────────────────────────

# # Used for CLV regression
# REG_FEATURES = [
#     "Age","IncomeValue","Income_enc","Channel_enc",
#     "Tenure","Frequency","AvgSpend","MonthlySpend","Recency","RFM_Score"
# ]

# # Used only for clustering
# CLUSTER_FEATURES = [
#     "Frequency","AvgSpend","Tenure","Recency","RFM_Score"
# ]

# # ─────────────────────────────────────────────────────────────
# # Load & prepare data
# # ─────────────────────────────────────────────────────────────

# def load_and_prepare():
#     print("📂 Loading data")
#     df = pd.read_csv(DATA_PATH)

#     le_inc = LabelEncoder().fit(df["Income"])
#     le_ch  = LabelEncoder().fit(df["Channel"])
#     le_seg = LabelEncoder().fit(df["Segment"])

#     df["Income_enc"]  = le_inc.transform(df["Income"])
#     df["Channel_enc"] = le_ch.transform(df["Channel"])
#     df["Segment_enc"] = le_seg.transform(df["Segment"])

#     joblib.dump(le_inc, os.path.join(ARTIFACTS, "le_income.pkl"))
#     joblib.dump(le_ch,  os.path.join(ARTIFACTS, "le_channel.pkl"))
#     joblib.dump(le_seg, os.path.join(ARTIFACTS, "le_segment.pkl"))
#     joblib.dump(REG_FEATURES, os.path.join(ARTIFACTS, "feature_names.pkl"))

#     return df, le_seg

# # ─────────────────────────────────────────────────────────────
# # Regression models (UNCHANGED)
# # ─────────────────────────────────────────────────────────────

# def train_regression(X_tr, X_te, y_tr, y_te):
#     print("\n🔢 CLV regression models")

#     boost_model = (
#         xgb.XGBRegressor(
#             n_estimators=300, max_depth=6, learning_rate=0.08,
#             subsample=0.8, colsample_bytree=0.8,
#             n_jobs=-1, random_state=42, verbosity=0
#         )
#         if XGB_OK else
#         _GBR(
#             n_estimators=200, max_depth=6,
#             learning_rate=0.08, subsample=0.8, random_state=42
#         )
#     )
#     boost_label = "XGBoost" if XGB_OK else "GradientBoosting"

#     models = {
#         "LinearRegression": LinearRegression(),
#         "Ridge": Ridge(alpha=10.0),
#         "RandomForest": RandomForestRegressor(
#             n_estimators=200, max_depth=14,
#             min_samples_leaf=5, n_jobs=-1, random_state=42
#         ),
#         boost_label: boost_model,
#     }

#     results   = {}
#     best_rmse = float("inf")
#     best_name = None

#     for name, model in models.items():
#         model.fit(X_tr, y_tr)
#         preds = model.predict(X_te)
#         rmse  = np.sqrt(mean_squared_error(y_te, preds))
#         r2    = r2_score(y_te, preds)

#         results[name] = {"RMSE": round(float(rmse), 2), "R2": round(float(r2), 4)}
#         joblib.dump(model, os.path.join(ARTIFACTS, f"reg_{name}.pkl"))

#         print(f"  {name:<20} RMSE={rmse:.2f}  R²={r2:.4f}")

#         if rmse < best_rmse:
#             best_rmse, best_name = rmse, name

#     joblib.dump(
#         joblib.load(os.path.join(ARTIFACTS, f"reg_{best_name}.pkl")),
#         os.path.join(ARTIFACTS, "reg_best.pkl")
#     )
#     results["best"] = best_name
#     print(f"\n🏆 Best regression model: {best_name}")

#     return results

# # ─────────────────────────────────────────────────────────────
# # Improved clustering (behavior + PCA)
# # ─────────────────────────────────────────────────────────────

# def train_clustering(df):
#     print("\n🔵 K-Means clustering (behavioral features + PCA)")

#     X = df[CLUSTER_FEATURES]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     pca = PCA(n_components=3, random_state=42)
#     X_pca = pca.fit_transform(X_scaled)

#     joblib.dump(pca, os.path.join(ARTIFACTS, "pca_cluster.pkl"))

#     sil_scores = {}

#     for k in range(2, 7):
#         km = KMeans(n_clusters=k, n_init=20, random_state=42)
#         labels = km.fit_predict(X_pca)
#         sil = silhouette_score(X_pca, labels)
#         sil_scores[k] = round(float(sil), 4)
#         print(f"  k={k} → silhouette={sil:.4f}")

#     km3 = KMeans(n_clusters=3, n_init=20, random_state=42)
#     km3.fit(X_pca)

#     joblib.dump(km3, os.path.join(ARTIFACTS, "kmeans_k3.pkl"))
#     print(f"✅ Saved KMeans k=3 | silhouette={sil_scores[3]}")

#     return sil_scores

# # ─────────────────────────────────────────────────────────────
# # ✅ FIXED CLASSIFIER (CLV → Segment)
# # ─────────────────────────────────────────────────────────────
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# def train_gmm_clustering(df):
#     print("\n🟣 Gaussian Mixture Model clustering (behavioral features + PCA)")

#     # ✅ Behavioral features only (no CLV)
#     CLUSTER_FEATURES = [
#         "Frequency",
#         "AvgSpend",
#         "Tenure",
#         "Recency",
#         "RFM_Score"
#     ]

#     X = df[CLUSTER_FEATURES]

#     # ✅ Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # ✅ Apply PCA (same as K-Means)
#     pca = PCA(n_components=3, random_state=42)
#     X_pca = pca.fit_transform(X_scaled)

#     sil_scores = {}

#     for k in range(2, 7):
#         gmm = GaussianMixture(
#             n_components=k,
#             covariance_type="full",
#             random_state=42
#         )

#         labels = gmm.fit_predict(X_pca)

#         sil = silhouette_score(X_pca, labels)
#         sil_scores[k] = round(float(sil), 4)

#         print(f"  GMM k={k} → silhouette={sil:.4f}")

#     # ✅ Save final GMM with k=3
#     gmm3 = GaussianMixture(
#         n_components=3,
#         covariance_type="full",
#         random_state=42
#     )
#     gmm3.fit(X_pca)

#     joblib.dump(gmm3, os.path.join(ARTIFACTS, "gmm_k3.pkl"))

#     print(f"✅ Saved GMM k=3 | silhouette={sil_scores[3]}")
#     return sil_scores

# def train_classifier(df):
#     print("\n🌳 Segment classifier (CLV-based)")

#     X = df[["CLV"]]
#     y = df["Segment"]

#     le = LabelEncoder()
#     y_enc = le.fit_transform(y)

#     X_tr, X_te, y_tr, y_te = train_test_split(
#         X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
#     )

#     dt = DecisionTreeClassifier(
#         max_depth=None,
#         min_samples_leaf=20,
#         class_weight="balanced",
#         random_state=42
#     )
#     dt.fit(X_tr, y_tr)

#     preds = dt.predict(X_te)
#     report = classification_report(
#         y_te, preds, target_names=le.classes_, output_dict=True
#     )
#     acc = report["accuracy"]

#     print(f"✅ Classification accuracy: {acc:.4f}")

#     joblib.dump(dt, os.path.join(ARTIFACTS, "dt_classifier.pkl"))

#     return {"accuracy": round(float(acc), 4), "report": report}

# # ─────────────────────────────────────────────────────────────
# # Save CLV thresholds
# # ─────────────────────────────────────────────────────────────

# def save_thresholds(y_clv):
#     thresh = {
#         "p33": float(np.percentile(y_clv, 33)),
#         "p66": float(np.percentile(y_clv, 66)),
#     }
#     joblib.dump(thresh, os.path.join(ARTIFACTS, "clv_thresholds.pkl"))
#     return thresh

# # ─────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────

# def main():
#     df, le_seg = load_and_prepare()

#     X = df[REG_FEATURES]
#     y_clv = df["CLV"]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     joblib.dump(scaler, os.path.join(ARTIFACTS, "scaler.pkl"))

#     X_tr, X_te, yc_tr, yc_te = train_test_split(
#         X_scaled, y_clv, test_size=0.2, random_state=42
#     )

#     reg_results = train_regression(X_tr, X_te, yc_tr, yc_te)
#     sil_results = train_clustering(df)
#     cls_results = train_classifier(df)
#     save_thresholds(y_clv)

#     metrics = {
#     "regression": reg_results,
#     "clustering_silhouette": {
#         "kmeans": {str(k): v for k, v in sil_results.items()},
#     },
#     "classification": cls_results,
# }

#     with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=2)

#     print(f"\n✅ All artifacts saved to {ARTIFACTS}/")

# if __name__ == "__main__":
#     main()