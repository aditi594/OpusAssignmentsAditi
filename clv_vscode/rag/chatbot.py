"""
rag/chatbot.py
--------------
Hybrid RAG chatbot (FINAL, FIXED)

- Uses FAISS for knowledge retrieval
- Uses Phi-2 for embeddings and answer generation
- Does NOT compute CLV
- Provides explanations & insights only
"""

import os
import sys
import json
import re
import numpy as np
import torch
import faiss
import pandas as pd

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ─────────────────────────────────────────────
# ✅ Ensure project root is importable (Streamlit-safe)
# ─────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from rag.qa_memory import get_cached_answer, store_answer

# ─────────────────────────────────────────────
# Paths & Globals
# ─────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "faiss_index")
DATA_PATH = os.path.join(ROOT_DIR, "data", "customers.csv")

PHI2_PATH = r"D:\phi2_local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_chunks = None
_index = None
_df = None

_embed_tokenizer = None
_embed_model = None
_gen_tokenizer = None
_gen_model = None

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def is_customer_query(query: str) -> bool:
    return (
        bool(re.search(r"CUST\d+", query)) or
        any(k in query.lower() for k in [
            "customer", "income", "segment",
            "clv", "recency", "frequency", "tenure"
        ])
    )

def load_df():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df

# ─────────────────────────────────────────────
# Load FAISS + Phi‑2 ONCE
# ─────────────────────────────────────────────

def _load():
    global _chunks, _index
    global _embed_tokenizer, _embed_model
    global _gen_tokenizer, _gen_model

    if _index is not None:
        return

    # Load FAISS index and chunks
    with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    _index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))

    # Embedding tokenizer + model
    _embed_tokenizer = AutoTokenizer.from_pretrained(
        PHI2_PATH, local_files_only=True, trust_remote_code=True
    )
    _embed_tokenizer.pad_token = _embed_tokenizer.eos_token

    _embed_model = AutoModel.from_pretrained(
        PHI2_PATH, local_files_only=True, trust_remote_code=True
    ).to(DEVICE)
    _embed_model.eval()

    # Generation tokenizer + model
    _gen_tokenizer = AutoTokenizer.from_pretrained(
        PHI2_PATH, local_files_only=True, trust_remote_code=True
    )
    _gen_tokenizer.pad_token = _gen_tokenizer.eos_token

    _gen_model = AutoModelForCausalLM.from_pretrained(
        PHI2_PATH, local_files_only=True, trust_remote_code=True
    ).to(DEVICE)
    _gen_model.eval()

# ─────────────────────────────────────────────
# Query Embedding (matching index logic)
# ─────────────────────────────────────────────

def _embed_query(text: str) -> np.ndarray:
    with torch.no_grad():
        inputs = _embed_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        hidden = _embed_model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        vec = pooled[0].cpu().numpy()
        vec = vec / np.linalg.norm(vec)

        return vec.reshape(1, -1).astype("float32")

# ─────────────────────────────────────────────
# Chat Function
# ─────────────────────────────────────────────

def chat(query: str, history=None, top_k: int = 8):
    _load()
    history = history or []

    # ✅ QA memory
    cached = get_cached_answer(query)
    if cached:
        return cached["answer"], history

    # ── Structured customer context (optional) ──
    context = ""
    if is_customer_query(query):
        df = load_df()

        if "high income" in query.lower():
            rows = df[df["Income"] == "High"].head(5)
        elif "medium income" in query.lower():
            rows = df[df["Income"] == "Medium"].head(5)
        elif "low income" in query.lower():
            rows = df[df["Income"] == "Low"].head(5)
        else:
            m = re.search(r"(CUST\d+)", query)
            rows = df[df["CustomerID"] == m.group(1)] if m else df.head(5)

        context = "\n".join(
            f"Customer {r.CustomerID} | Age {r.Age} | Income {r.Income} | "
            f"CLV {r.CLV:.2f} | Segment {r.Segment}"
            for r in rows.itertuples()
        )

    # ── FAISS retrieval ──
    q_vec = _embed_query(query)
    _, idxs = _index.search(q_vec, top_k)

    blocks = []
    char_count = 0
    MAX_CHARS = 3000

    for idx in idxs[0]:
        chunk = _chunks[idx]
        block = f"{chunk['title']}: {chunk['text']}\n"

        if char_count + len(block) > MAX_CHARS:
            break

        blocks.append(block)
        char_count += len(block)

    context += "\n" + "\n".join(blocks)

    # ── Prompt ──
    prompt = f"""
You are a banking analytics assistant.
Use ONLY the context below.
If information is missing, say "Not available in data".

Context:
{context}

Question:
{query}

Answer:
""".strip()

    inputs = _gen_tokenizer(
        prompt, padding=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = _gen_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True,
            pad_token_id=_gen_tokenizer.eos_token_id
        )

    # ✅ ✅ ✅ CORRECT OUTPUT DECODING (FIXED)
    prompt_len = inputs["input_ids"].shape[1]
    answer = _gen_tokenizer.decode(
        output[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    store_answer(query, answer)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer, history