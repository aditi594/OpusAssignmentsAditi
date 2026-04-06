"""
data/generate_data.py
---------------------
Generates 100K synthetic customers with realistic CLV correlations.
Run:  python data/generate_data.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 100_000

def generate():
    age      = np.random.randint(18, 70, N)
    income   = np.random.choice(["Low","Medium","High","Very High"], p=[0.30,0.35,0.25,0.10], size=N)
    income_map = {"Low":25_000,"Medium":55_000,"High":95_000,"Very High":160_000}
    income_val = np.array([income_map[i] for i in income]) * (1 + np.random.normal(0,0.15,N))
    income_val = np.clip(income_val, 10_000, 300_000)
    channel  = np.random.choice(["App","Web","Branch"], p=[0.45,0.35,0.20], size=N)
    income_norm = (income_val - income_val.min()) / (income_val.max() - income_val.min())
    tenure   = np.clip(np.random.exponential(scale=30, size=N).astype(int)+1, 1, 120)
    frequency= np.clip(np.random.poisson(lam=5+income_norm*20, size=N).astype(int), 1, 100)
    avg_spend= np.clip(income_norm*800 + np.random.normal(100,50,N), 10, 2000)
    monthly_spend = frequency * avg_spend / 12
    recency  = np.clip(np.random.exponential(scale=30+(1-income_norm)*90, size=N).astype(int)+1, 1, 365)
    r_score  = pd.qcut(recency, 5, labels=[5,4,3,2,1]).astype(int)
    f_score  = pd.qcut(frequency.clip(1,99), 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    m_score  = pd.qcut(monthly_spend, 5, labels=[1,2,3,4,5]).astype(int)
    rfm_score= r_score + f_score + m_score
    margin   = 0.05 + income_norm * 0.10
    dr       = 0.01
    clv      = monthly_spend * margin * (1-(1+dr)**-tenure) / dr
    clv     += np.random.normal(0, clv*0.05)
    clv      = np.clip(clv, 10, 500_000)
    clv_33   = np.percentile(clv, 33)
    clv_66   = np.percentile(clv, 66)
    segment  = np.where(clv>=clv_66, "High", np.where(clv>=clv_33, "Medium", "Low"))

    df = pd.DataFrame({
        "CustomerID":   [f"CUST{str(i).zfill(6)}" for i in range(1, N+1)],
        "Age":          age,
        "Income":       income,
        "IncomeValue":  income_val.round(2),
        "Channel":      channel,
        "Tenure":       tenure,
        "Frequency":    frequency,
        "AvgSpend":     avg_spend.round(2),
        "MonthlySpend": monthly_spend.round(2),
        "Recency":      recency,
        "RFM_Score":    rfm_score,
        "CLV":          clv.round(2),
        "Segment":      segment,
    })

    out = os.path.join(os.path.dirname(__file__), "customers.csv")
    df.to_csv(out, index=False)
    print(f"✅  Generated {N:,} records → {out}")
    print(f"    CLV percentiles  p33=${clv_33:,.0f}  p66=${clv_66:,.0f}")
    print(f"\n    Segment distribution:")
    for seg, cnt in df["Segment"].value_counts().items():
        print(f"      {seg:<8}: {cnt:,}  ({cnt/N*100:.1f}%)")
    return df

if __name__ == "__main__":
    generate()
