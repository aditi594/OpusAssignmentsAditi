"""
models/elbow_method.py
----------------------
Standalone script to find optimal K using the Elbow Method
for CLV-only clustering and visualize the inertia (WCSS) curve.

This is for exploratory analysis only.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "customers.csv")

# ─────────────────────────────────────────────
# CLV ONLY FEATURE
# ─────────────────────────────────────────────

CLUSTER_FEATURES = ["CLV"]

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("📂 Loading data for CLV-only Elbow Method analysis")
    df = pd.read_csv(DATA_PATH)

    #  Use CLV only
    X = df[CLUSTER_FEATURES]

    #  Scale CLV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(" CLV scaled successfully")

    #  Elbow Method
    K_RANGE = range(2, 8)
    inertia = []

    for k in K_RANGE:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=42
        )
        km.fit(X_scaled)
        inertia.append(km.inertia_)
        print(f"k={k} → inertia={km.inertia_:.2f}")

    #  Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_RANGE, inertia, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for CLV-Only K-Means")
    plt.grid(True)
    plt.show()

    print("\nLook for the elbow point where inertia reduction slows down.")

if __name__ == "__main__":
    main()