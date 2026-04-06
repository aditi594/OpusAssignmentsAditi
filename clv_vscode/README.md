# CLV Intelligence Platform

An end-to-end AI platform for predicting Customer Lifetime Value, segmenting customers, recommending personalised offers, and answering business questions via a GenAI RAG chatbot — built with Python, Streamlit, scikit-learn, XGBoost, FAISS, Anthropic Claude, and MongoDB.

---

## Project Structure

```
clv_vscode/
│
├── data/
│   └── generate_data.py          # Generates 100K synthetic customers
│
├── models/
│   ├── train_models.py            # Trains all ML models
│   ├── predict.py                 # Single + batch inference
│   └── personalization.py        # Offer recommendation engine
│
├── rag/
│   ├── build_index.py             # Builds FAISS/NumPy vector index
│   └── chatbot.py                 # RAG chatbot using Anthropic Claude
│
├── ui/
│   ├── app.py                     # Main Streamlit app — run this
│   ├── auth.py                    # Login system (MongoDB + credentials.json)
│   ├── credentials.json           # Default user accounts (fallback)
│   ├── page_search.py             # Customer Search page
│   ├── page_simulator.py          # What-If Simulator page
│   ├── page_add_customer.py       # Add Customer page
│   ├── page_users.py              # User Management page
│   └── style.css                  # Dark theme CSS
│
├── .env.example                   # Copy to .env and fill in your keys
├── .gitignore                     # Files excluded from Git
├── requirements.txt               # Python dependencies
├── setup.py                       # First-time setup script
├── seed_users.py                  # Seeds default users into MongoDB
├── add_manager.py                 # Adds a manager account
└── README.md                      # This file
```

---

## Prerequisites

| Requirement | Version | Download |
|-------------|---------|----------|
| Python | 3.10, 3.11, or 3.12 | [python.org](https://python.org) |
| VS Code | Any recent version | [code.visualstudio.com](https://code.visualstudio.com) |
| Git | Any recent version | [git-scm.com](https://git-scm.com) |
| MongoDB Atlas | Free tier | [mongodb.com/atlas](https://mongodb.com/atlas) |
| Anthropic API Key | Free credits available | [console.anthropic.com](https://console.anthropic.com) |

---

## Setup — Run Once

### Step 1 — Open in VS Code

```
Extract zip → File → Open Folder → select clv_vscode
```

### Step 2 — Open the terminal

```
Ctrl + ` (backtick)
```

### Step 3 — Create virtual environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 4 — Run first-time setup

```powershell
python setup.py
```

This automatically:
- Installs all dependencies (~3 minutes)
- Generates 100K synthetic customers
- Trains all ML models (~3-5 minutes)
- Builds the RAG vector index

### Step 5 — Create your .env file

**Windows:**
```powershell
copy .env.example .env
```
Open `.env` and fill in your keys:

```
(OPTIONAL)
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
MONGO_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/CLV
```

- Get Anthropic key: [console.anthropic.com](https://console.anthropic.com) → API Keys → Create Key
- Get MongoDB URI: [mongodb.com/atlas](https://mongodb.com/atlas) → Free cluster → Connect → Drivers → copy connection string

### Step 6 — Seed users into MongoDB

```powershell
python seed_users.py
```

### Step 7 — Launch the dashboard

```powershell
streamlit run ui/app.py
```

Open **http://localhost:8501** in your browser.

---

## Every Time You Want to Run the App

```powershell
# Activate virtual environment
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate          # Mac/Linux

# Launch
streamlit run ui/app.py
```

---

## All Commands Reference

| Command | What it does | When to run |
|---------|-------------|-------------|
| `python -m venv venv` | Creates virtual environment | Once |
| `venv\Scripts\Activate.ps1` | Activates venv (Windows) | Every session |
| `source venv/bin/activate` | Activates venv (Mac/Linux) | Every session |
| `python setup.py` | Full first-time setup | Once |
| `pip install -r requirements.txt` | Install/update dependencies | When requirements change |
| `python data/generate_data.py` | Generate 100K customer CSV | Once (or to regenerate) |
| `python models/train_models.py` | Train all ML models | Once (or to retrain) |
| `python rag/build_index.py` | Build RAG vector index | Once (or to rebuild) |
| `python seed_users.py` | Add default users to MongoDB | Once |
| `python add_user.py` | Add manager account | Once |
| `streamlit run ui/app.py` | Launch the dashboard | Every session |

---

## Login Credentials

| Username | Password | Role | Access |
|----------|----------|------|--------|
| admin | admin123 | Admin | All 8 pages |
| marketingmanager1 | marketingmanager123 | MarketingManager | All pages except User Management restrictions |
| productmanager1 | productmanager123 | ProductManager | All pages except User Management restrictions |
| Analyst | analyst123 | Analyst | All pages except Offer Management restrictions |

**Change passwords** from the User Management page after first login.

---

## Role Permissions

| Permission | Admin | Manager | Viewer |
|-----------|-------|---------|--------|
| Dashboard | ✅ | ✅ | ✅ |
| CLV Predictor | ✅ | ✅ | ❌ |
| Customer Search | ✅ | ✅ | ❌ |
| Add Customer | ✅ | ✅ | ❌ |
| What-If Simulator | ✅ | ✅ | ❌ |
| AI Advisor | ✅ | ✅ | ✅ |
| Model Metrics | ✅ | ✅ | ❌ |
| Offer Management | ✅ | ✅ | ❌ |



---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Dashboard | KPIs, 6 Plotly charts, segment summary table |
| CLV Predictor | Single prediction + personalised offers + batch CSV upload |
| Customer Search | Search/filter 100K customers, radar chart, percentile ranking |
| Add Customer | Add new customers with live CLV preview, saves to MongoDB + CSV |
| What-If Simulator | Drag sliders to simulate interventions, sensitivity analysis, ROI estimator |
| AI Advisor | Claude RAG chatbot grounded in customer data |
| Model Metrics | RMSE comparison, silhouette sweep, classifier report |
| Offer Management | Add, edit, delete Offers |

---

## ML Models

| Model | Purpose | Metric |
|-------|---------|--------|
| Linear Regression | CLV baseline | RMSE |
| Random Forest | Ensemble CLV | RMSE |
| XGBoost | Best CLV performer | RMSE |
| K-Means (k=3) | Customer segmentation | Silhouette Score |
| Decision Tree | Segment classifier | Accuracy / F1 |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| UI | Streamlit |
| Charts | Plotly |
| ML | scikit-learn, XGBoost |
| Clustering | K-Means |
| Vector DB | FAISS (NumPy fallback) |
| Embeddings | sentence-transformers |
| GenAI | Anthropic Claude |
| Database | MongoDB Atlas |
| Auth | SHA-256 hashed passwords |

---
