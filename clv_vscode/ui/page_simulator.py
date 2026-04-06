"""
ui/page_simulator.py
---------------------
What-If CLV Simulator page.
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SEG = {"High":"#10b981","Medium":"#0ea5e9","Low":"#64748b"}


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(os.path.join(ROOT, "data", "customers.csv"))


@st.cache_resource(show_spinner=False)
def load_model():
    import joblib
    art = os.path.join(ROOT, "models", "artifacts")
    return (joblib.load(os.path.join(art, "reg_best.pkl")),
            joblib.load(os.path.join(art, "scaler.pkl")),
            joblib.load(os.path.join(art, "feature_names.pkl")),
            joblib.load(os.path.join(art, "le_income.pkl")),
            joblib.load(os.path.join(art, "le_channel.pkl")),
            joblib.load(os.path.join(art, "clv_thresholds.pkl")))


def predict_one(row_dict, model, scaler, features, le_inc, le_ch):
    row = {
        "Age":          row_dict["Age"],
        "IncomeValue":  row_dict["IncomeValue"],
        "Income_enc":   le_inc.transform([row_dict["Income"]])[0],
        "Channel_enc":  le_ch.transform([row_dict["Channel"]])[0],
        "Tenure":       row_dict["Tenure"],
        "Frequency":    row_dict["Frequency"],
        "AvgSpend":     row_dict["AvgSpend"],
        "MonthlySpend": row_dict["MonthlySpend"],
        "Recency":      row_dict["Recency"],
        "RFM_Score":    row_dict["RFM_Score"],
    }
    X = pd.DataFrame([row])[features]
    return float(model.predict(scaler.transform(X))[0])


def get_segment(clv, thresh):
    if clv >= thresh["p66"]: return "High"
    if clv >= thresh["p33"]: return "Medium"
    return "Low"


def render():
    st.markdown(
        '<div class="page-header"><h1>What-If CLV Simulator</h1>'
        '<p>Adjust customer parameters and see how CLV and segment change in real time</p></div>',
        unsafe_allow_html=True)

    try:
        model, scaler, features, le_inc, le_ch, thresh = load_model()
        model_ok = True
    except Exception as e:
        st.warning(f"Model not loaded — using formula fallback. ({e})")
        model_ok = False
        thresh = {"p33": 400, "p66": 2000}

    income_map = {"Low":25000,"Medium":55000,"High":95000,"Very High":160000}

    left, right = st.columns([1, 1.6])

    with left:
        st.markdown('<p class="section-label">Baseline Customer</p>', unsafe_allow_html=True)
        age       = st.slider("Age",                         18,   70,  35, key="age")
        income    = st.selectbox("Income Bracket", ["Low","Medium","High","Very High"], index=2, key="inc")
        channel   = st.selectbox("Primary Channel", ["App","Web","Branch"], key="ch")
        tenure    = st.slider("Tenure (months)",              1,  120,  24, key="ten")
        frequency = st.slider("Transactions / Month",         1,   60,  10, key="freq")
        avg_spend = st.slider("Avg Spend per Transaction ($)",10, 2000,150, key="spend")
        recency   = st.slider("Days Since Last Transaction",  1,  365,  15, key="rec")

        st.divider()
        st.markdown('<p class="section-label">Intervention Levers</p>', unsafe_allow_html=True)
        st.caption("Simulate the effect of business interventions.")
        delta_freq    = st.slider("Increase transaction frequency by",  0, 30,  0, 1,  key="df",  format="+%d tx/mo")
        delta_spend   = st.slider("Increase avg spend by",              0, 500, 0, 10, key="ds",  format="+$%d")
        delta_tenure  = st.slider("Extend tenure by",                   0, 36,  0, 1,  key="dt",  format="+%d mo")
        delta_recency = st.slider("Reduce recency by",                  0, 180, 0, 5,  key="dr",  format="-%d days")

    def build_row(f, s, t, r):
        ms  = f * s / 12
        rfm = min(15, max(3, int(r<30)*3 + int(f>10)*6 + int(s>200)*6))
        return {"Age":age,"Income":income,"IncomeValue":income_map[income],
                "Channel":channel,"Tenure":t,"Frequency":f,"AvgSpend":s,
                "MonthlySpend":ms,"Recency":r,"RFM_Score":rfm}

    base_row = build_row(frequency, avg_spend, tenure, recency)
    new_row  = build_row(min(100, frequency+delta_freq), min(2000, avg_spend+delta_spend),
                         min(120, tenure+delta_tenure),  max(1, recency-delta_recency))

    if model_ok:
        base_clv = predict_one(base_row, model, scaler, features, le_inc, le_ch)
        new_clv  = predict_one(new_row,  model, scaler, features, le_inc, le_ch)
    else:
        dr = 0.01
        base_clv = base_row["MonthlySpend"]*0.07*(1-(1+dr)**-base_row["Tenure"])/dr
        new_clv  = new_row["MonthlySpend"] *0.07*(1-(1+dr)**-new_row["Tenure"]) /dr

    base_clv    = max(0, base_clv)
    new_clv     = max(0, new_clv)
    delta_clv   = new_clv - base_clv
    delta_pct   = (delta_clv / base_clv * 100) if base_clv > 0 else 0
    base_seg    = get_segment(base_clv, thresh)
    new_seg     = get_segment(new_clv, thresh)
    seg_changed = base_seg != new_seg

    with right:
        st.markdown('<p class="section-label">Simulation Results</p>', unsafe_allow_html=True)
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Baseline CLV",  f"${base_clv:,.0f}")
        k2.metric("Projected CLV", f"${new_clv:,.0f}",
                  f"+${delta_clv:,.0f}" if delta_clv>=0 else f"-${abs(delta_clv):,.0f}")
        k3.metric("CLV Change",    f"{delta_pct:+.1f}%")
        k4.metric("Segment",       new_seg, f"{base_seg} → {new_seg}" if seg_changed else "No change")

        if seg_changed:
            sc = SEG[new_seg]
            st.markdown(
                '<div class="card card-accent" style="margin:0.5rem 0">'
                f'<p style="color:{sc};font-weight:600;font-size:0.9rem;margin:0">'
                f'Segment upgrade: {base_seg} → {new_seg} — this customer crosses a value tier.</p></div>',
                unsafe_allow_html=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Baseline", x=["CLV"], y=[base_clv],
            marker_color=SEG[base_seg], opacity=0.6,
            text=[f"${base_clv:,.0f}"], textposition="outside",
            textfont=dict(color="#94a3b8",size=12)))
        fig_bar.add_trace(go.Bar(name="Projected", x=["CLV"], y=[new_clv],
            marker_color=SEG[new_seg],
            text=[f"${new_clv:,.0f}"], textposition="outside",
            textfont=dict(color="#e2e8f0",size=13)))
        fig_bar.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#94a3b8",family="Inter",size=11),
            margin=dict(l=16,r=16,t=36,b=16),
            title=dict(text="Baseline vs Projected CLV",font=dict(color="#e2e8f0",size=13)),
            barmode="group", showlegend=True,
            legend=dict(font=dict(color="#94a3b8",size=10),bgcolor="#111827"),
            yaxis=dict(gridcolor="#1e2d45",title="CLV ($)"),
            xaxis=dict(showticklabels=False))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<p class="section-label">Parameter Comparison</p>', unsafe_allow_html=True)
        params = [
            ("Transactions / Month", frequency,             min(100,frequency+delta_freq)),
            ("Avg Spend ($)",        avg_spend,             min(2000,avg_spend+delta_spend)),
            ("Monthly Spend ($)",    round(base_row["MonthlySpend"],0), round(new_row["MonthlySpend"],0)),
            ("Tenure (months)",      tenure,                min(120,tenure+delta_tenure)),
            ("Recency (days)",       recency,               max(1,recency-delta_recency)),
            ("RFM Score",            base_row["RFM_Score"], new_row["RFM_Score"]),
        ]
        tbl = pd.DataFrame(params, columns=["Parameter","Baseline","Projected"])
        tbl["Change"] = tbl.apply(
            lambda r: f'{r["Projected"]-r["Baseline"]:+.0f}' if r["Projected"]!=r["Baseline"] else "—", axis=1)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<p class="section-label">Sensitivity Analysis — Impact of Each Lever</p>', unsafe_allow_html=True)
    st.caption("How much CLV changes if only that one lever is applied, all else equal.")

    levers = []
    for label, lo, hi, step, field in [
        ("Frequency +tx/mo",   1, 15,  1, "Frequency"),
        ("Avg Spend +$",      50, 500, 50, "AvgSpend"),
        ("Tenure +months",     6, 36,  6, "Tenure"),
        ("Recency -days",     15, 90, 15, "Recency"),
    ]:
        clvs = []
        steps = list(range(lo, hi+step, step))
        for s in steps:
            r = build_row(
                min(100, frequency + (s if field=="Frequency" else 0)),
                min(2000, avg_spend  + (s if field=="AvgSpend"  else 0)),
                min(120,  tenure     + (s if field=="Tenure"    else 0)),
                max(1,    recency    - (s if field=="Recency"   else 0)),
            )
            if model_ok:
                clvs.append(predict_one(r, model, scaler, features, le_inc, le_ch))
            else:
                dr = 0.01
                clvs.append(r["MonthlySpend"]*0.07*(1-(1+dr)**-r["Tenure"])/dr)
        levers.append({"label":label,"steps":steps,"clvs":clvs})

    fig_sens = go.Figure()
    colors   = ["#10b981","#0ea5e9","#f59e0b","#e879f9"]
    for lev, col in zip(levers, colors):
        fig_sens.add_trace(go.Scatter(x=lev["steps"], y=lev["clvs"], name=lev["label"],
            mode="lines+markers", line=dict(color=col,width=2), marker=dict(size=5)))
    fig_sens.add_hline(y=base_clv, line_dash="dash", line_color="#475569",
                       annotation_text="Baseline", annotation_font_color="#475569")
    fig_sens.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#94a3b8",family="Inter",size=11),
        margin=dict(l=16,r=16,t=36,b=16),
        title=dict(text="CLV Response to Individual Levers",font=dict(color="#e2e8f0",size=13)),
        legend=dict(font=dict(color="#94a3b8",size=10),bgcolor="#111827",orientation="h",y=1.1),
        xaxis=dict(title="Intervention Magnitude",gridcolor="#1e2d45"),
        yaxis=dict(title="Projected CLV ($)",gridcolor="#1e2d45"))
    st.plotly_chart(fig_sens, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-label">ROI Estimator</p>', unsafe_allow_html=True)
    ri1, ri2, ri3 = st.columns(3)
    with ri1: cost_per_cust = st.number_input("Intervention cost per customer ($)", 0, 10000, 50, 5)
    with ri2: n_customers   = st.number_input("Number of customers targeted", 1, 100000, 1000, 100)
    with ri3: uptake_pct    = st.slider("Expected uptake rate (%)", 1, 100, 30)
    uptake       = int(n_customers * uptake_pct / 100)
    total_cost   = n_customers * cost_per_cust
    total_uplift = uptake * delta_clv
    roi          = ((total_uplift - total_cost) / total_cost * 100) if total_cost > 0 else 0
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Campaign Cost",   f"${total_cost:,.0f}")
    m2.metric("Customers Who Respond", f"{uptake:,}")
    m3.metric("Total CLV Uplift",      f"${total_uplift:,.0f}")
    m4.metric("Estimated ROI",         f"{roi:+.0f}%")
    if roi > 0:
        st.markdown('<div class="card card-accent"><p style="color:#10b981;font-weight:600;font-size:0.875rem;margin:0">Positive ROI — this intervention is projected to be profitable.</p></div>', unsafe_allow_html=True)
    elif roi < 0:
        st.markdown('<div class="card" style="border-left:3px solid #ef4444"><p style="color:#ef4444;font-weight:600;font-size:0.875rem;margin:0">Negative ROI — reduce cost per customer or target a higher-CLV segment.</p></div>', unsafe_allow_html=True)
