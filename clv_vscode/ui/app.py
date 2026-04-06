"""
ui/app.py
---------
Main Streamlit application — CLV Intelligence Platform.
Run:  streamlit run ui/app.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from auth import current_role

# Load .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
st.set_page_config(
    page_title="CLV Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
_css = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
st.markdown(f"<style>{open(_css).read()}</style>", unsafe_allow_html=True)

# Auth
from ui.auth import require_auth, logout, is_admin, current_user
require_auth()

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    p = os.path.join(ROOT, "data", "customers.csv")
    if not os.path.exists(p): st.error("Dataset not found. Run: python setup.py"); st.stop()
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def load_metrics():
    p = os.path.join(ROOT, "models", "artifacts", "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None

def models_ready():
    return os.path.exists(os.path.join(ROOT, "models", "artifacts", "reg_best.pkl"))

def rag_ready():
    idx = os.path.join(ROOT, "rag", "faiss_index")
    return (os.path.exists(os.path.join(idx, "index.faiss")) or
            os.path.exists(os.path.join(idx, "index_numpy.npy")))

PLOT = dict(
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(color="#94a3b8", family="Inter", size=11),
    margin=dict(l=16, r=16, t=40, b=16),
    title_font=dict(size=13, color="#e2e8f0"),
)
SEG_COLORS = {"High":"#10b981","Medium":"#0ea5e9","Low":"#64748b"}

def topnav():
    role = current_role()

    # Role-based pages
    if role == "admin":
        pages = [
            "Dashboard",
            "CLV Predictor",
            "Customer Search",
            "What-If Simulator",
            "AI Advisor",
            "Model Metrics",
            "Offer Management",
        ]

    elif role == "marketing":
        pages = [
            "Dashboard",
            "Customer Search",
            "AI Advisor",
        ]

    elif role == "product":
        pages = [
            "Dashboard",
            "What-If Simulator",
            "AI Advisor",
            "Offer Management",
        ]

    elif role == "analyst":
        pages = [
            "Dashboard",
            "Customer Search",
            "Model Metrics",
            "AI Advisor",
        ]

    else:  # viewer
        pages = [
            "Dashboard",
            "AI Advisor",
        ]

    data_ok = os.path.exists(os.path.join(ROOT, "data", "customers.csv"))

    role_badge_map = {
        "admin": "badge-blue",
        "marketing": "badge-purple",
        "product": "badge-orange",
        "analyst": "badge-green",
        "viewer": "badge-gray",
    }

    role_badge = (
        f'<span class="badge {role_badge_map.get(role, "badge-gray")}">'
        f'{role.capitalize()}</span>'
    )

    n = len(pages)
    cols = st.columns([1.8] + [1]*n + [2])

    # ── Logo ─────────────────────────────
    with cols[0]:
        st.markdown(
            '<div style="padding:0.6rem 0;font-size:0.85rem;font-weight:700;'
            'color:#f1f5f9;letter-spacing:0.08em">'
            'CLV <span style="color:#0ea5e9">INTELLIGENCE</span></div>',
            unsafe_allow_html=True
        )

    # ── Navigation buttons ───────────────
    for i, label in enumerate(pages):
        with cols[i+1]:
            if st.button(label, key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()

    # ── Status / user info ───────────────
    with cols[-1]:
        ds_b = (
            '<span class="badge badge-green">Data</span>'
            if data_ok else
            '<span class="badge badge-red">No Data</span>'
        )

        name = current_user()

        st.markdown(
            f'<div style="padding:0.4rem 0;text-align:right;display:flex;'
            f'align-items:center;justify-content:flex-end;gap:6px">'
            f'{ds_b}{role_badge}'
            f'<span style="color:#475569;font-size:0.72rem">{name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        if st.button("Sign out", key="signout"):
            logout()

    st.markdown(
        '<div style="border-bottom:1px solid #1e2d45;margin-bottom:1.5rem"></div>',
        unsafe_allow_html=True
    )
topnav()
page = st.session_state.page
role = current_role()

PAGE_ACCESS = {
    "CLV Predictor": ["admin", "marketing"],
    "Customer Search": ["admin", "marketing", "analyst"],
    "What-If Simulator": ["admin", "product"],
    "Model Metrics": ["admin", "analyst"],
    "Offer Management": ["admin", "product"],
}

if page in PAGE_ACCESS and role not in PAGE_ACCESS[page]:
    st.warning("You do not have permission to access this page.")
    st.stop()
# ── Dashboard ────────────────────────────────────────────────────────────────

if page == "Dashboard":
    st.markdown(
        '<div class="page-header"><h1>Customer Intelligence Dashboard</h1>'
        '<p>Overview of customer lifetime value, segmentation, and revenue metrics</p></div>',
        unsafe_allow_html=True
    )

    df = load_data()
    rev_tot = df["CLV"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)    # ── KPI Cards ────────────────────────────────────────────────────────────
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Avg Predicted CLV", f"${df['CLV'].mean():,.0f}")
    c3.metric("High-Value Customers", f"{(df['Segment']=='High').mean()*100:.1f}%")
    c4.metric("Revenue — Top Tier", f"{df[df['Segment']=='High']['CLV'].sum()/rev_tot*100:.1f}%")
    c5.metric("Total Revenue Pool", f"${rev_tot/1e6:.1f}M")

    st.divider()

    # ── TOP ROW (2 charts) ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    # Chart 1: Segment Distribution
    with col1:
        seg = df["Segment"].value_counts()
        fig1 = px.pie(
            values=seg.values,
            names=seg.index,
            title="Segment Distribution",
            color=seg.index,
            color_discrete_map=SEG_COLORS,
            hole=0.5
        )
        fig1.update_layout(**PLOT)
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Revenue Contribution by Segment
    with col2:
        rev = df.groupby("Segment")["CLV"].sum().reset_index()
        rev["pct"] = (rev["CLV"] / rev["CLV"].sum() * 100).round(1)

        fig2 = px.bar(
            rev,
            x="Segment",
            y="CLV",
            color="Segment",
            title="Revenue Contribution by Segment",
            text=rev["pct"].map(lambda x: f"{x}%"),
            color_discrete_map=SEG_COLORS
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(**PLOT, showlegend=False, yaxis_title="Total CLV ($)")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── BOTTOM ROW (2 charts) ────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    # Chart 3: Segment Distribution by Channel
    with col3:
        chan = df.groupby(["Channel", "Segment"]).size().reset_index(name="Count")
        fig3 = px.bar(chan, x="Channel",  y="Count",color="Segment", barmode="group",  title="Segment Distribution by Channel",
            color_discrete_map=SEG_COLORS
        )
        fig3.update_layout(
            **PLOT,
            yaxis_title="Customers",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Frequency vs Avg Spend
    with col4:
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig4 = px.scatter(
            sample,
            x="Frequency",
            y="AvgSpend",
            color="Segment",
            size="CLV",
            title="Frequency vs Avg Spend",
            opacity=0.55,
            size_max=14,
            color_discrete_map=SEG_COLORS
        )
        fig4.update_layout(
            **PLOT,
            xaxis_title="Transactions / Month",
            yaxis_title="Avg Spend ($)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Segment Summary Table ────────────────────────────────────────────────
    st.markdown('<p class="section-label">Segment Summary</p>', unsafe_allow_html=True)

    summary = df.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_CLV=("CLV", "mean"),
        Avg_Monthly_Spend=("MonthlySpend", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Tenure_Months=("Tenure", "mean"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_RFM=("RFM_Score", "mean"),
    ).round(1).reset_index()

    summary["Avg_CLV"] = summary["Avg_CLV"].map("${:,.0f}".format)
    summary["Avg_Monthly_Spend"] = summary["Avg_Monthly_Spend"].map("${:,.0f}".format)

    st.dataframe(summary, use_container_width=True, hide_index=True)


# ── CLV Predictor ──────────────────────────────────────────────────────────────

elif page == "CLV Predictor":
    st.markdown(
        '<div class="page-header"><h1>CLV Predictor</h1>'
        '<p>Predict customer lifetime value and receive personalised offer recommendations</p></div>',
        unsafe_allow_html=True
    )

    if not models_ready():
        st.error("Models not trained. Run: python setup.py")
        st.stop()

    from models.predict import predict_clv, predict_segment
    from models.personalization import get_offers, get_offer_rationale

    # ─────────────────────────────
    # Input form
    # ─────────────────────────────
    with st.form("clv_form"):
        st.markdown('<p class="section-label">Customer Profile</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Age", 18, 70, 35)
            income = st.selectbox("Income Bracket", ["Low","Medium","High","Very High"])
            channel = st.selectbox("Primary Channel", ["App","Web","Branch"])

        with c2:
            tenure = st.slider("Tenure (months)", 1, 120, 24)
            frequency = st.slider("Monthly Transactions", 1, 60, 10)
            recency = st.slider("Days Since Last Transaction", 1, 365, 15)

        with c3:
            avg_spend = st.number_input(
                "Avg Spend per Transaction ($)",
                10.0, 2000.0, 150.0, 10.0
            )

            income_map = {
                "Low": 25000,
                "Medium": 55000,
                "High": 95000,
                "Very High": 160000
            }

            monthly_spend = frequency * avg_spend / 12
            rfm_score = min(
                15,
                max(
                    3,
                    int(recency < 30) * 3 +
                    int(frequency > 10) * 6 +
                    int(avg_spend > 200) * 6
                )
            )

            st.metric("Estimated Monthly Spend", f"${monthly_spend:,.0f}")
            st.metric("Estimated RFM Score", rfm_score)

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    # ─────────────────────────────
    # Prediction & results
    # ─────────────────────────────
    if submitted:
        cust = {
            "Age": age,
            "Income": income,
            "IncomeValue": income_map[income],
            "Channel": channel,
            "Tenure": tenure,
            "Frequency": frequency,
            "AvgSpend": avg_spend,
            "MonthlySpend": monthly_spend,
            "Recency": recency,
            "RFM_Score": rfm_score
        }

        try:
            clv = predict_clv(cust)
            segment = predict_segment(clv)
        except Exception as e:
            dr = 0.01
            clv = float(monthly_spend * 0.07 * (1 - (1 + dr) ** -tenure) / dr)
            segment = "High" if clv >= 2000 else ("Medium" if clv >= 400 else "Low")
            st.warning(f"Formula fallback used: {e}")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted CLV", f"${clv:,.0f}")
        r2.metric("Segment", segment)
        r3.metric("Monthly Revenue Potential", f"${monthly_spend*0.07:,.0f}")
        r4.metric("Tenure", f"{tenure} months")

        # ─────────────────────────────
        # Recommendation rationale
        # ─────────────────────────────
        rationale = get_offer_rationale(segment, clv, frequency, avg_spend)
        st.markdown(
            f'''
            <div class="card card-accent" style="margin-top:1rem">
                <p class="section-label">Recommendation Rationale</p>
                <p style="color:#cbd5e1;font-size:0.875rem;margin:0">{rationale}</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # ─────────────────────────────
        # Personalised offers ( FIXED)
        # ─────────────────────────────
        st.markdown(
            '<p class="section-label" style="margin-top:1.5rem">Personalised Offers</p>',
            unsafe_allow_html=True
        )

        offers = get_offers(segment, top_n=4)

        if not offers:
            st.info("No active offers available for this segment.")
        else:
            for col, offer in zip(st.columns(4), offers):
                col.markdown(
                    f'''
                    <div class="offer-card">
                        <span class="offer-tag">{offer["category"].replace("_"," ")}</span>
                        <div class="offer-title">{offer["title"]}</div>
                        <div class="offer-desc">{offer["description"]}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
    REQUIRED_COLUMNS = [
        "CustomerID",
        "Age",
        "Income",
        "IncomeValue",
        "Channel",
        "Tenure",
        "Frequency",
        "AvgSpend",
        "MonthlySpend",
        "Recency",
        "RFM_Score"
    ]
    def clean_uploaded_data(df: pd.DataFrame):
        report = []

        original_rows = len(df)

        # ── 1. Remove duplicate CustomerIDs ──────────────────
        if df["CustomerID"].duplicated().any():
            dup_count = df["CustomerID"].duplicated().sum()
            df = df.drop_duplicates(subset="CustomerID")
            report.append(f"Removed {dup_count} duplicate CustomerID(s).")

        # ── 2. Drop rows with missing critical values ────────
        critical_cols = [
            "Age", "Income", "IncomeValue", "Channel",
            "Tenure", "Frequency", "AvgSpend",
            "MonthlySpend", "Recency", "RFM_Score"
        ]

        missing_before = df[critical_cols].isnull().any(axis=1).sum()
        if missing_before > 0:
            df = df.dropna(subset=critical_cols)
            report.append(
                f"Dropped {missing_before} row(s) with missing critical values."
            )

        # ── 3. Remove invalid numeric values ─────────────────
        numeric_cols = [
            "Age", "IncomeValue", "Tenure", "Frequency",
            "AvgSpend", "MonthlySpend", "Recency", "RFM_Score"
        ]

        invalid_mask = (df[numeric_cols] < 0).any(axis=1)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            df = df.loc[~invalid_mask]
            report.append(
                f"Removed {invalid_count} row(s) with invalid negative values."
            )

        final_rows = len(df)
        report.append(
            f"Rows before cleaning: {original_rows}, after cleaning: {final_rows}."
        )

        return df, report

    def validate_uploaded_data(df):
        errors = []

        # 1️⃣ Required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}")
            return errors  # schema error → stop immediately

        # 2️⃣ Duplicate Customer IDs
        if df["CustomerID"].duplicated().any():
            count = df["CustomerID"].duplicated().sum()
            errors.append(f"{count} duplicate CustomerID(s) found.")

        # 3️⃣ Missing values
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            errors.append(f"Missing values in columns: {', '.join(null_cols)}")

        # 4️⃣ Invalid numeric values
        numeric_cols = [
            "Age", "IncomeValue", "Tenure", "Frequency",
            "AvgSpend", "MonthlySpend", "Recency", "RFM_Score"
        ]

        for col in numeric_cols:
            if (df[col] < 0).any():
                errors.append(f"Negative values found in column '{col}'.")

        # 5️⃣ Logical sanity checks
        if (df["Age"] > 120).any():
            errors.append("Age values exceed realistic limits (> 120).")

        if (df["Frequency"] > 100).any():
            errors.append("Frequency values seem unrealistically high (> 100).")

        return errors
        st.markdown(
        '<p class="section-label">Data Preprocessing & Cleaning</p>',
        unsafe_allow_html=True
    )

    auto_clean = st.checkbox(
        "Automatically clean data (remove duplicates & handle missing values)",
        help=(
            "If enabled, the system will:\n"
            "• Remove duplicate CustomerIDs\n"
            "• Drop rows with missing critical values\n"
            "• Ensure numeric columns are valid\n"
            "\nYou will see a summary before prediction."
        )
    )

    # ─────────────────────────────
    # Batch prediction
    # ─────────────────────────────
    st.divider()
    st.markdown('<p class="section-label">Batch Prediction</p>', unsafe_allow_html=True)
    st.caption("Upload a CSV with the same columns as customers.csv")

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
    # ── File size validation (500 MB) ────────────────────
        if uploaded.size > MAX_FILE_SIZE_BYTES:
            st.error(
                f" File too large. Uploaded size: "
                f"{uploaded.size / (1024 * 1024):.2f} MB. "
                f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
            )
            st.stop()

        try:
            df = pd.read_csv(uploaded)

            # ── Validation ─────────────────────────────────────
            errors = validate_uploaded_data(df)

            if errors and not auto_clean:
                st.error(" Dataset validation failed:")
                for err in errors:
                    st.warning(f"• {err}")
                st.info("Enable data cleaning to automatically fix these issues.")
                st.stop()

            # ── Optional Cleaning ──────────────────────────────
            if auto_clean:
                df, clean_report = clean_uploaded_data(df)

                st.success(" Dataset cleaned successfully!")
                for item in clean_report:
                    st.info(item)

            # ── Predict ────────────────────────────────────────
            from models.predict import batch_predict
            result = batch_predict(df)

            st.success(f" Predicted CLV for {len(result):,} customers.")

            st.dataframe(
                result[["CustomerID", "CLV_Predicted", "Segment_Predicted"]].head(200),
                use_container_width=True
            )

            st.download_button(
                "Download Results",
                result.to_csv(index=False).encode(),
                "clv_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Unexpected error while processing dataset: {e}")
# ── Customer Search ────────────────────────────────────────────────────────────

elif page == "Customer Search":
    from ui.page_search import render as _rs; _rs()


# ── What-If Simulator ──────────────────────────────────────────────────────────

elif page == "What-If Simulator":
    from ui.page_simulator import render as _rw; _rw()


# ── AI Advisor ─────────────────────────────────────────────────────────────────

elif page == "AI Advisor":
    st.markdown(
        '<div class="page-header"><h1>AI Advisor</h1>'
        '<p>Retrieval-augmented chatbot grounded in your customer data and model outputs</p></div>',
        unsafe_allow_html=True)

    if not rag_ready():
        st.error("RAG index not built. Run: python setup.py")
        st.stop()

    from rag.chatbot import chat

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    # Show once
    # 

    st.markdown('<p class="section-label">Suggested Questions</p>', unsafe_allow_html=True)
    suggested = [
        "Who are my top customers?",
        "How can I improve customer retention?",
        "Which segment generates the most revenue?",
        "What offers should high-value customers receive?",
        "How should I allocate the marketing budget?",
        "Explain the RFM model.",
    ]

    cols = st.columns(3)
    for i, q in enumerate(suggested):
        if cols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state._pending = q

    st.divider()

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                '<div class="chat-label" style="text-align:right">You</div>'
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="chat-label">AI Advisor</div>'
                f'<div class="chat-bot">{msg["content"]}</div>',
                unsafe_allow_html=True)

    user_input = st.chat_input("Ask about your customers, segments, or strategies ...")

    if not user_input and hasattr(st.session_state, "_pending"):
        user_input = st.session_state._pending
        del st.session_state._pending

    if user_input:
        with st.spinner("Generating response ..."):
            try:
                answer, updated = chat(
                    user_input,
                    history=st.session_state.rag_history
                )
                st.session_state.rag_history = updated
            except Exception as e:
                answer = f"⚠ Error: {e}"

        # ✅ CORRECT indentation — always append
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.rag_history = []
        st.rerun()

# ── Model Metrics ──────────────────────────────────────────────────────────────

# ── Model Metrics ─────────────────────────────────────────────────────────────

elif page == "Model Metrics":
    st.markdown(
        '<div class="page-header"><h1>Model Performance</h1>'
        '<p>Evaluation metrics for regression, clustering, and classification models</p></div>',
        unsafe_allow_html=True
    )

    metrics = load_metrics()
    if not metrics:
        st.error("No metrics found. Run: python setup.py")
        st.stop()

    # ── CLV REGRESSION MODELS (RIDGE REMOVED) ─────────────────────────────────
    st.markdown(
        '<p class="section-label">CLV Regression — Model Comparison</p>',
        unsafe_allow_html=True
    )

    reg = {
        k: v
        for k, v in metrics["regression"].items()
        if k not in ["best", "Ridge", "Ridge Regression"]
    }

    best = metrics["regression"]["best"]

    # Table
    st.dataframe(
        pd.DataFrame([
            {
                "Model": name,
                "RMSE": f"${vals['RMSE']:,.0f}",
                "R²": f"{vals['R2']:.4f}",
                "Adj R²": f"{vals['Adj_R2']*100:.2f}%",
                "Status": "Best" if name == best else "—"
            }
            for name, vals in reg.items()
        ]),
        use_container_width=True,
        hide_index=True
    )

    # RMSE Bar Chart
    names = list(reg.keys())
    rmses = [reg[n]["RMSE"] for n in names]

    fig = go.Figure(go.Bar(
        x=names,
        y=rmses,
        marker_color=[
            "#10b981" if n == best else "#1e2d45"
            for n in names
        ],
        text=[f"${r:,.0f}" for r in rmses],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11)
    ))

    fig.update_layout(
        **PLOT,
        title="RMSE by Model — lower is better",
        yaxis_title="RMSE ($)",
        showlegend=False,
        yaxis=dict(gridcolor="#1e2d45")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── K-MEANS CLUSTERING (BEHAVIORAL FEATURES) ─────────────────────────────

    st.caption(
        "Clustering is performed using behavioral customer features such as "
        "frequency, recency, tenure, and spending patterns. "
        "This view illustrates how customers naturally group based on behavior."
    )

    st.markdown(
        '<p class="section-label">K-Means Clustering — Behavioral Features</p>',
        unsafe_allow_html=True
    )

    # ✅ Load behavioral silhouette scores (OLD one)
    sil_behavioral = metrics["clustering_silhouette"]["behavioral_kmeans"]

    # ✅ Sort K values safely
    k_vals = sorted([int(k) for k in sil_behavioral.keys()])
    sil_vals = [sil_behavioral[str(k)] for k in k_vals]

    fig = px.line(
        pd.DataFrame({
            "K": k_vals,
            "Silhouette Score": sil_vals
        }),
        x="K",
        y="Silhouette Score",
        markers=True,
        title="Silhouette Score vs Number of Clusters (Behavioral Clustering)",
        color_discrete_sequence=["#22c55e"]
    )

    # ✅ Selected K (same as before)
    fig.add_vline(
        x=3,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="  Selected (k=3)",
        annotation_font_color="#f59e0b",
        annotation_font_size=11
    )

    fig.update_layout(
        **PLOT,
        yaxis=dict(gridcolor="#1e2d45"),
        xaxis=dict(gridcolor="#1e2d45", dtick=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── DECISION TREE CLASSIFIER METRICS ──────────────────────────────────────
    st.markdown(
        '<p class="section-label">Decision Tree Classifier</p>',
        unsafe_allow_html=True
    )

    cls = metrics["classification"]

    st.metric(
        "Overall Classification Accuracy",
        f"{cls['accuracy'] * 100:.2f}%"
    )

    st.caption(
        "The classifier predicts customer segments based on CLV thresholds. "
        "Detailed precision/recall metrics are omitted for clarity."
    )
elif page == "Offer Management":
    from ui.page_offers import render
    render()