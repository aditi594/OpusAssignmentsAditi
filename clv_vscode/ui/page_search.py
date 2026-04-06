"""
ui/page_search.py
-----------------
Customer Search & Profile page.
"""

import os, sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SEG = {"High":"#10b981","Medium":"#0ea5e9","Low":"#64748b"}


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(os.path.join(ROOT, "data", "customers.csv"))


def render():
    st.markdown(
        '<div class="page-header"><h1>Customer Search</h1>'
        '<p>Look up any customer by ID or filter by segment and channel</p></div>',
        unsafe_allow_html=True)
    df = load_data()

    ca, cb, cc = st.columns([3, 1.5, 1.5])
    with ca:
        q  = st.text_input("", placeholder="Search by Customer ID — e.g. CUST000042",
                           label_visibility="collapsed")
    with cb:
        sf = st.selectbox("Segment", ["All","High","Medium","Low"], label_visibility="collapsed")
    with cc:
        cf = st.selectbox("Channel", ["All","App","Web","Branch"], label_visibility="collapsed")

    flt = df.copy()
    if q.strip():  flt = flt[flt["CustomerID"].str.contains(q.strip(), case=False, na=False)]
    if sf != "All": flt = flt[flt["Segment"] == sf]
    if cf != "All": flt = flt[flt["Channel"] == cf]

    st.markdown(f'<p class="section-label">{len(flt):,} customers found</p>', unsafe_allow_html=True)
    if len(flt) == 0: st.info("No customers match your search."); return

    d = flt[["CustomerID","Age","Income","Channel","Tenure","Frequency",
             "AvgSpend","MonthlySpend","Recency","RFM_Score","CLV","Segment"]].copy()
    d["CLV"]          = d["CLV"].map("${:,.0f}".format)
    d["AvgSpend"]     = d["AvgSpend"].map("${:,.0f}".format)
    d["MonthlySpend"] = d["MonthlySpend"].map("${:,.0f}".format)
    st.dataframe(d.head(200), use_container_width=True, hide_index=True,
        column_config={
            "CustomerID": st.column_config.TextColumn("Customer ID"),
            "CLV":        st.column_config.TextColumn("Predicted CLV"),
            "RFM_Score":  st.column_config.NumberColumn("RFM Score"),
            "Frequency":  st.column_config.NumberColumn("Tx / Month"),
            "Recency":    st.column_config.NumberColumn("Days Since Last Tx"),
        })
    st.download_button("Download Results", flt.to_csv(index=False).encode(), "search_results.csv", "text/csv")

    st.divider()
    st.markdown('<p class="section-label">Customer Profile</p>', unsafe_allow_html=True)
    exact = flt[flt["CustomerID"].str.upper() == q.strip().upper()]
    if exact.empty and q.strip():
        sel   = st.selectbox("Select a customer to profile", flt["CustomerID"].tolist()[:50],
                              label_visibility="collapsed")
        exact = flt[flt["CustomerID"] == sel]
    elif exact.empty:
        st.caption("Enter a Customer ID above to view a full profile.")
        return
    if exact.empty: return

    c  = exact.iloc[0]
    sc = SEG.get(c["Segment"], "#64748b")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Customer ID",       c["CustomerID"])
    k2.metric("Predicted CLV",     f"${c['CLV']:,.0f}")
    k3.metric("Segment",            c["Segment"])
    k4.metric("RFM Score",          int(c["RFM_Score"]))
    k5.metric("Tenure",             f"{int(c['Tenure'])} mo")
    k6.metric("Days Since Last Tx", int(c["Recency"]))

    html  = '<div class="card card-accent" style="margin-top:1rem">'
    html += '<p class="section-label">Profile Summary</p>'
    html += '<div class="profile-grid">'
    def pcol(title, rows):
        s  = "<div>"
        s += f'<p class="profile-col-title">{title}</p>'
        for r in rows: s += f'<p class="profile-row">{r}</p>'
        s += "</div>"
        return s
    html += pcol("Demographics",    [f"Age: {int(c['Age'])}", f"Income: {c['Income']}", f"Channel: {c['Channel']}"])
    html += pcol("Spend Behaviour", [f"Avg Spend: ${c['AvgSpend']:,.0f}", f"Monthly: ${c['MonthlySpend']:,.0f}", f"Frequency: {int(c['Frequency'])}x / mo"])
    html += pcol("Loyalty Signals", [f"Tenure: {int(c['Tenure'])} months", f"Recency: {int(c['Recency'])} days", f"RFM: {int(c['RFM_Score'])} / 15"])
    html += ('<div>'
             f'<p class="profile-col-title">CLV Intelligence</p>'
             f'<p style="color:{sc};font-size:1.1rem;font-weight:700;margin:2px 0">${c["CLV"]:,.0f}</p>'
             f'<p class="profile-row">Segment: <span style="color:{sc};font-weight:600">{c["Segment"]}</span></p>'
             f'<p class="profile-row">Monthly revenue: ${c["MonthlySpend"]*0.07:,.0f}</p>'
             '</div>')
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        rfm   = int(c["RFM_Score"])
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=rfm, domain={"x":[0,1],"y":[0,1]},
            title={"text":"RFM Score","font":{"color":"#e2e8f0","size":13}},
            gauge={"axis":{"range":[3,15],"tickcolor":"#475569","tickfont":{"color":"#475569","size":10}},
                   "bar":{"color":sc},"bgcolor":"#1e2d45",
                   "steps":[{"range":[3,7],"color":"#1e293b"},{"range":[7,11],"color":"#1e3a5f"},{"range":[11,15],"color":"#0f3460"}],
                   "threshold":{"line":{"color":"#f59e0b","width":2},"thickness":0.75,"value":rfm}},
            number={"font":{"color":sc,"size":32}}))
        fig_g.update_layout(paper_bgcolor="#111827", font_color="#94a3b8",
                            margin=dict(l=24,r=24,t=40,b=16), height=220)
        st.plotly_chart(fig_g, use_container_width=True)

    with col2:
        savg  = df[df["Segment"]==c["Segment"]].mean(numeric_only=True)
        omax  = df.max(numeric_only=True)
        dims  = ["Frequency","AvgSpend","Tenure","MonthlySpend","RFM_Score"]
        lbls  = ["Frequency","Avg Spend","Tenure","Monthly Spend","RFM Score"]
        cv    = [round(c[d]/omax[d]*100,1) for d in dims]+[round(c[dims[0]]/omax[dims[0]]*100,1)]
        av    = [round(savg[d]/omax[d]*100,1) for d in dims]+[round(savg[dims[0]]/omax[dims[0]]*100,1)]
        th    = lbls+[lbls[0]]
        r,g,b = int(sc[1:3],16),int(sc[3:5],16),int(sc[5:7],16)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=av,theta=th,fill="toself",name="Segment avg",
            fillcolor="rgba(14,165,233,0.1)",line=dict(color="#0ea5e9",width=1.5,dash="dot")))
        fig_r.add_trace(go.Scatterpolar(r=cv,theta=th,fill="toself",name=c["CustomerID"],
            fillcolor=f"rgba({r},{g},{b},0.2)",line=dict(color=sc,width=2)))
        fig_r.update_layout(
            polar=dict(bgcolor="#111827",
                radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color="#475569",size=9),gridcolor="#1e2d45"),
                angularaxis=dict(tickfont=dict(color="#94a3b8",size=10),gridcolor="#1e2d45")),
            showlegend=True, legend=dict(font=dict(color="#94a3b8",size=10),bgcolor="#111827"),
            paper_bgcolor="#111827", margin=dict(l=40,r=40,t=40,b=16), height=220,
            title=dict(text="vs Segment Average",font=dict(color="#e2e8f0",size=12)))
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('<p class="section-label" style="margin-top:0.5rem">Recommended Offers</p>', unsafe_allow_html=True)
    try:
        from models.personalization import get_offers, get_offer_rationale
        rat = get_offer_rationale(c["Segment"], c["CLV"], int(c["Frequency"]), c["AvgSpend"])
        st.markdown('<div class="card card-accent"><p style="color:#cbd5e1;font-size:0.875rem;margin:0">' + rat + '</p></div>', unsafe_allow_html=True)
        for col, offer in zip(st.columns(4), get_offers(c["Segment"], top_n=4)):
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
    except Exception as e:
        st.warning(f"Could not load offers: {e}")

    st.divider()
    st.markdown('<p class="section-label">Percentile Ranking</p>', unsafe_allow_html=True)
    pc = round((df["CLV"]<=c["CLV"]).mean()*100,1)
    pf = round((df["Frequency"]<=c["Frequency"]).mean()*100,1)
    ps = round((df["MonthlySpend"]<=c["MonthlySpend"]).mean()*100,1)
    pr = round((df["RFM_Score"]<=c["RFM_Score"]).mean()*100,1)
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("CLV Percentile",           f"Top {100-pc:.0f}%")
    p2.metric("Frequency Percentile",     f"Top {100-pf:.0f}%")
    p3.metric("Monthly Spend Percentile", f"Top {100-ps:.0f}%")
    p4.metric("RFM Percentile",           f"Top {100-pr:.0f}%")
