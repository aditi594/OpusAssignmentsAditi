import json
import faiss
import joblib
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch

# ─────────────────────────────────────
# Paths
# ─────────────────────────────────────

INDEX_DIR = "rag/faiss_index"
CHUNKS_FILE = f"{INDEX_DIR}/chunks.json"
FAISS_INDEX = f"{INDEX_DIR}/index.faiss"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────
# Load chunks & index
# ─────────────────────────────────────

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

index = faiss.read_index(FAISS_INDEX)

# ─────────────────────────────────────
# Embedding model (use SAME as build_index.py)
# ─────────────────────────────────────

PHI2_PATH = r"D:\phi2_local"   # same path you used before

tokenizer = AutoTokenizer.from_pretrained(
    PHI2_PATH, local_files_only=True, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(
    PHI2_PATH, local_files_only=True, trust_remote_code=True
).to(DEVICE)


model.eval()

def embed(texts):
    with torch.no_grad():
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
        emb = emb.cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype("float32")

# ─────────────────────────────────────
# ✅ SIMPLE RETRIEVER (NO retriever.py needed)
# ─────────────────────────────────────

def retrieve_context(question, top_k=3):
    q_emb = embed([question])
    _, idxs = index.search(q_emb, top_k)

    results = []
    for i in idxs[0]:
        results.append(CHUNKS[i])
    return results

# ─────────────────────────────────────
# ✅ YOUR EXISTING ANSWER GENERATION
# ─────────────────────────────────────

def generate_answer(question, contexts):
    """
    Replace this with your existing generator logic
    (LLM, Phi‑2, OpenAI, etc.)
    """
    context_text = "\n".join(contexts)

    # 🔴 PLACEHOLDER — replace with your actual LLM call
    return f"Answer based on context: {context_text[:300]}..."

# ─────────────────────────────────────
# Prepare RAGAS dataset
# ─────────────────────────────────────

with open("rag_evaluation/eval_questions.json") as f:
    eval_data = json.load(f)

questions, answers, contexts, ground_truths = [], [], [], []

for item in eval_data:
    question = item["question"]

    retrieved = retrieve_context(question, top_k=3)
    context_texts = [c["text"] for c in retrieved]

    answer = generate_answer(question, context_texts)

    questions.append(question)
    answers.append(answer)
    contexts.append(context_texts)
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

dataset.save_to_disk("rag_evaluation/ragas_dataset")
print("✅ RAGAS dataset prepared")