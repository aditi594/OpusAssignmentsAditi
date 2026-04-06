"""
setup.py
--------
First-time setup script. Run this once before launching the app.

What it does:
  1. Installs all Python dependencies
  2. Generates 100K synthetic customer dataset
  3. Trains all ML models (Linear, Ridge, RandomForest, XGBoost, KMeans, DecisionTree)
  4. Builds the RAG vector index for the AI Advisor chatbot

Usage:
    python setup.py
"""

import subprocess, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  ERROR in: {label}")
        print(f"  Fix the error above and run setup.py again.")
        sys.exit(1)

print("""
╔══════════════════════════════════════════════════════════════╗
║              - CLV Intelligence Platform      —              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# 1. Install dependencies
#run(f'"{sys.executable}" -m pip install -r requirements.txt -q',
#     "Step 1 of 4 — Installing dependencies")

# 2. Generate dataset
data_path = os.path.join(ROOT, "data", "customers.csv")
if not os.path.exists(data_path):
    run(f'"{sys.executable}" data/generate_data.py',
        "Step 2 of 4 — Generating synthetic dataset (100K customers)")
else:
    print("\n✅  Dataset already exists — skipping generation.")

# 3. Train models
model_path = os.path.join(ROOT, "models", "artifacts", "reg_best.pkl")
if not os.path.exists(model_path):
    run(f'"{sys.executable}" models/train_models.py',
        "Step 3 of 4 — Training ML models (3-5 minutes)")
else:
    print("✅  Models already trained — skipping.")

# 4. Build RAG index
rag_path = os.path.join(ROOT, "rag", "faiss_index", "index_numpy.npy")
if not os.path.exists(rag_path):
    run(f'"{sys.executable}" rag/build_index.py',
        "Step 4 of 4 — Building RAG vector index")
else:
    print("✅  RAG index already built — skipping.")

print("""
╔══════════════════════════════════════════════════════════════╗
║  ✅  Setup complete!                                          ║
║                                                              ║
║  Next steps:                                                 ║
║                                                              ║
║  1. Copy .env.example to .env                                ║
║       Windows:   copy .env.example .env                      ║
║       Mac/Linux: cp .env.example .env                        ║
║                                                              ║
║  2. Open .env and add your keys:                             ║
║       ANTHROPIC_API_KEY = your Claude API key                ║
║       MONGO_URI         = your MongoDB connection string     ║
║                                                              ║
║  3. Seed users into MongoDB (first time only):               ║
║       python seed_users.py                                   ║
║                                                              ║
║  4. Launch the dashboard:                                    ║
║       streamlit run ui/app.py                                ║
║                                                              ║
║  5. Open in browser:                                         ║
║       http://localhost:8501                                  ║
║                                                              ║
║  Default login:  admin / admin123                            ║
╚══════════════════════════════════════════════════════════════╝
""")
