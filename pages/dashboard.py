"""pages/dashboard.py — Live KPI Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import generate_claims_dataset, get_fraud_type_breakdown


@st.cache_data(show_spinner=False)
def load_data():
    return generate_claims_dataset(2000)


def render():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; margin-bottom:4px;">
      🛡️ Fraud Intelligence Dashboard
    </h1>
    <p style="color:#8899aa; font-family:'DM Mono',monospace; font-size:0.85rem; margin-bottom:24px;">
      Real-time monitoring · SADC Funeral Insurance · Live stream simulation
    </p>
    """, unsafe_allow_html=True)

    df = load_data()

    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0]
    fraud_rate = df["is_fraud"].mean()
    total_exposure = df["claim_amount_usd"].sum()
    fraud_exposure = fraud_df["claim_amount_usd"].sum()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Claims", f"{len(df):,}", "+12 today")
    with c2:
        st.metric("Fraud Detected", f"{len(fraud_df):,}", f"{fraud_rate:.1%} rate")
    with c3:
        st.metric("Fraud Exposure", f"${fraud_exposure:,.0f}", "USD at risk")
    with c4:
        avg_fraud = fraud_df["claim_amount_usd"].mean()
        avg_legit = legit_df["claim_amount_usd"].mean()
        st.metric("Avg Fraud Claim", f"${avg_fraud:,.0f}", f"+{avg_fraud/avg_legit:.1f}x vs legit")
    with c5:
        loss_ratio_dist = fraud_exposure / total_exposure
        st.metric("Fraud Loss Ratio", f"{loss_ratio_dist:.1%}", "of total exposure")

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### 📅 Claims Volume & Fraud Rate — 12-Month Trend")
        df_copy = df.copy()
        df_copy["month"] = pd.to_datetime(df_copy["claim_date"]).dt.to_period("M").astype(str)
        monthly = df_copy.groupby("month").agg(
            total=("is_fraud", "count"),
            fraud=("is_fraud", "sum"),
        ).reset_index()
        monthly["rate"] = monthly["fraud"] / monthly["total"]
        monthly = monthly.tail(12)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["total"],
            name="Total Claims", marker_color="#1a2235",
            marker_line_color="#2a3a55", marker_line_width=1,
        ))
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["fraud"],
            name="Fraud Claims", marker_color="#ff6b2b",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["rate"],
            name="Fraud Rate", yaxis="y2",
            line=dict(color="#00e5ff", width=2, dash="dot"),
            mode="lines+markers", marker=dict(size=5),
        ))
        fig.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=300,
            yaxis=dict(gridcolor="#1a2235", title="Claims"),
            yaxis2=dict(overlaying="y", side="right", tickformat=".1%", title="Rate"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🍩 Fraud Typology")
        typology = get_fraud_type_breakdown(df)
        labels = list(typology.keys())
        values = list(typology.values())
        short_labels = [
            "Forged Docs", "Cross-Insurer Dup.",
            "Policy Backdating", "Parlour Collusion", "Internal Fraud"
        ]
        fig2 = go.Figure(go.Pie(
            labels=short_labels[:len(labels)],
            values=values,
            hole=0.55,
            marker=dict(colors=["#00e5ff", "#ff6b2b", "#00e676", "#ffd600", "#ab47bc"]),
            textinfo="percent+label",
            textfont=dict(size=10, family="DM Mono"),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa"),
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### 🗺️ Fraud by Region")
        reg_fraud = df.groupby("region").agg(
            fraud=("is_fraud", "sum"),
            total=("is_fraud", "count"),
        ).reset_index()
        reg_fraud["rate"] = reg_fraud["fraud"] / reg_fraud["total"]
        reg_fraud = reg_fraud.sort_values("rate", ascending=True)

        fig3 = go.Figure(go.Bar(
            y=reg_fraud["region"],
            x=reg_fraud["rate"],
            orientation="h",
            marker=dict(
                color=reg_fraud["rate"],
                colorscale=[[0, "#1a2235"], [0.5, "#ff6b2b"], [1, "#ff1744"]],
                showscale=False,
            ),
            text=[f"{v:.1%}" for v in reg_fraud["rate"]],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#8899aa"),
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            height=350,
            xaxis=dict(tickformat=".0%", gridcolor="#1a2235"),
            yaxis=dict(gridcolor="#1a2235"),
            margin=dict(l=0, r=60, t=10, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("#### 📦 Claim Amount Distribution")
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=legit_df["claim_amount_usd"],
            name="Legitimate", nbinsx=40,
            marker_color="rgba(0,230,118,0.4)",
            marker_line_color="#00e676", marker_line_width=1,
        ))
        fig4.add_trace(go.Histogram(
            x=fraud_df["claim_amount_usd"],
            name="Fraud", nbinsx=40,
            marker_color="rgba(255,107,43,0.5)",
            marker_line_color="#ff6b2b", marker_line_width=1,
        ))
        fig4.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            height=350,
            xaxis=dict(gridcolor="#1a2235", title="USD"),
            yaxis=dict(gridcolor="#1a2235"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Recent High-Risk Claims Table ─────────────────────────────────────────
    st.markdown("#### 🚨 Recent High-Risk Claims — Requires Investigation")
    high_risk = df[df["raw_fraud_score"] > 0.6].sort_values(
        "raw_fraud_score", ascending=False
    ).head(10)

    display_cols = [
        "claim_id", "claim_date", "region", "claim_amount_usd",
        "policy_age_months", "raw_fraud_score", "document_status", "is_fraud",
    ]
    hr_display = high_risk[display_cols].copy()
    hr_display["claim_date"] = pd.to_datetime(hr_display["claim_date"]).dt.strftime("%Y-%m-%d")
    hr_display["claim_amount_usd"] = hr_display["claim_amount_usd"].apply(lambda x: f"${x:,.0f}")
    hr_display["raw_fraud_score"] = hr_display["raw_fraud_score"].apply(lambda x: f"{x:.3f}")
    hr_display.columns = [
        "Claim ID", "Date", "Region", "Amount", "Policy Age (mo)",
        "Fraud Score", "Doc Status", "Confirmed Fraud",
    ]

    def highlight_fraud(row):
        if row["Confirmed Fraud"] == 1:
            return ["background-color: rgba(255,107,43,0.12)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        hr_display.style.apply(highlight_fraud, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    <div class="info-alert">
      ℹ️ <b>Note:</b> Fraud scores are generated by the ensemble model pipeline.
      Claims flagged above 0.60 are routed to the investigation queue.
      <br>Source: Synthetic SADC funeral insurance claims dataset — UNZA Research 2026.
    </div>
    """, unsafe_allow_html=True)
