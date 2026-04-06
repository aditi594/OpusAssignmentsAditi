import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("data/customers.csv")

#  CLV-only training
X = df[["CLV"]]
X_scaled = StandardScaler().fit_transform(X).astype("float32")

kmeans_clv = KMeans(n_clusters=3, random_state=42)
kmeans_clv.fit(X_scaled)

#  OVERWRITE with true CLV-only model
joblib.dump(kmeans_clv, "models/artifacts/kmeans_k3.pkl")

print(" kmeans_k3.pkl retrained as CLV-only")
