"""
rag/build_index.py
------------------
Builds FAISS cosine index using Phi-2 semantic embeddings.

FINAL STABLE VERSION:
✔ Phi‑2 loaded once
✔ pad_token set correctly
✔ Batch logging enabled
✔ Customer chunks limited
✔ FAISS IndexFlatIP
"""

import os
import json
import numpy as np
import torch
import faiss
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────
# Paths & config
# ─────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INDEX_DIR = os.path.join(HERE, "faiss_index")
DATA_PATH = os.path.join(ROOT, "data", "customers.csv")

PHI2_PATH = r"D:\phi2_local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_CUSTOMERS = 200

os.makedirs(INDEX_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load Phi‑2 ONCE (CRITICAL)
# ─────────────────────────────────────────────

print("Loading Phi‑2 embedding model (one‑time)...")

tokenizer = AutoTokenizer.from_pretrained(
    PHI2_PATH, local_files_only=True, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # ✅ FIX

model = AutoModel.from_pretrained(
    PHI2_PATH, local_files_only=True, trust_remote_code=True
).to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
# Chunk builders
# ─────────────────────────────────────────────

def build_business_chunks():
    return [
        {
            "id": "biz_high",
            "type": "analysis",
            "title": "High-Value Customers",
            "text": (
                "High-value customers are in the top segment of predicted CLV and "
                "contribute a disproportionately large share of total revenue."
            ),
        },
        {
            "id": "biz_medium",
            "type": "analysis",
            "title": "Medium-Value Customers",
            "text": (
                "Medium-value customers represent the largest growth opportunity "
                "and can be upgraded with engagement strategies."
            ),
        },
        {
            "id": "biz_low",
            "type": "analysis",
            "title": "Low-Value Customers",
            "text": (
                "Low-value customers are new or inactive and are targeted with "
                "activation campaigns."
            ),
        },
        {
            "id": "biz_rfm",
            "type": "analysis",
            "title": "RFM Model",
            "text": (
                "The RFM model evaluates recency, frequency, and monetary value "
                "to summarize customer behaviour."
            ),
        },
    ]


def build_dataset_chunks(df):
    chunks = []
    total = len(df)
    seg_pct = df["Segment"].value_counts(normalize=True) * 100

    chunks.append({
        "id": "ds_overview",
        "type": "analysis",
        "title": "Customer Base Overview",
        "text": (
            f"The dataset contains {total} customers. "
            f"High-value customers represent {seg_pct.get('High', 0):.1f}%, "
            f"Medium-value customers {seg_pct.get('Medium', 0):.1f}%, "
            f"Low-value customers {seg_pct.get('Low', 0):.1f}%."
        ),
    })

    return chunks


def build_customer_entity_chunks(df, max_customers=MAX_CUSTOMERS):
    chunks = []

    top_customers = df.sort_values("CLV", ascending=False).head(max_customers)

    for _, r in top_customers.iterrows():
        chunks.append({
            "id": f"cust_{r['CustomerID']}",
            "type": "customer",
            "title": f"Customer {r['CustomerID']}",
            "text": (
                f"Customer {r['CustomerID']} is {r['Age']} years old with {r['Income']} income. "
                f"Tenure is {r['Tenure']} months. Frequency is {r['Frequency']} per month. "
                f"Average spend is {r['AvgSpend']:.2f}. Monthly spend is {r['MonthlySpend']:.2f}. "
                f"Recency is {r['Recency']} days. RFM score is {r['RFM_Score']}. "
                f"Predicted CLV is {r['CLV']:.2f}. Segment is {r['Segment']}."
            ),
        })

    return chunks

# ─────────────────────────────────────────────
# Embedding with progress
# ─────────────────────────────────────────────

def embed(texts):
    vectors = []
    total = len(texts)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Embedding {total} chunks in {total_batches} batches")

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, total, BATCH_SIZE), start=1):
            batch = texts[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1)

            vecs = pooled.cpu().numpy()
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

            vectors.append(vecs)

            print(f"Batch {batch_idx}/{total_batches} embedded")

    return np.vstack(vectors).astype("float32")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Building chunks...")

    df = pd.read_csv(DATA_PATH)

    chunks = []
    chunks.extend(build_business_chunks())
    chunks.extend(build_dataset_chunks(df))
    chunks.extend(build_customer_entity_chunks(df))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed([c["text"] for c in chunks])

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    print("✅ RAG index built successfully")

if __name__ == "__main__":
    main()
