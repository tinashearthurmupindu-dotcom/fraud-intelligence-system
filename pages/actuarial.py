"""pages/actuarial.py — Actuarial & Financial Impact Analysis"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import generate_claims_dataset


@st.cache_data(show_spinner=False)
def load_data():
    return generate_claims_dataset(2000)


def render():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; margin-bottom:4px;">
      💰 Actuarial & Financial Impact Analysis
    </h1>
    <p style="color:#8899aa; font-family:'DM Mono',monospace; font-size:0.85rem; margin-bottom:24px;">
      Objective 3: Quantifying the financial benefit of ML-driven fraud detection
    </p>
    """, unsafe_allow_html=True)

    df = load_data()
    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0]

    # ── Sidebar parameters ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**⚙️ Actuarial Parameters**")
        portfolio_size   = st.number_input("Portfolio Size (policies)", 1000, 500000, 50000, 1000)
        avg_premium      = st.number_input("Avg Annual Premium (USD)", 50, 2000, 180, 10)
        detection_rate   = st.slider("Model Detection Rate (Recall)", 0.50, 0.99, 0.87, 0.01)
        fp_rate          = st.slider("False Positive Rate", 0.01, 0.20, 0.05, 0.01)
        investigation_cost = st.number_input("Investigation Cost per Flag (USD)", 50, 2000, 200, 50)
        fraud_prevalence = st.slider("Fraud Prevalence (%)", 1.0, 10.0, 4.5, 0.1) / 100

    total_premium = portfolio_size * avg_premium
    expected_fraud_claims = int(portfolio_size * fraud_prevalence)
    avg_fraud_claim = fraud_df["claim_amount_usd"].mean()
    total_fraud_loss = expected_fraud_claims * avg_fraud_claim

    detected = int(expected_fraud_claims * detection_rate)
    missed   = expected_fraud_claims - detected
    fp_count = int(portfolio_size * (1 - fraud_prevalence) * fp_rate)

    savings_from_detection  = detected * avg_fraud_claim
    investigation_costs     = (detected + fp_count) * investigation_cost
    net_savings             = savings_from_detection - investigation_costs
    roi                     = (net_savings / max(investigation_costs, 1)) * 100

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Portfolio-Level Impact Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Premium Income", f"${total_premium:,.0f}", "Annual")
    k2.metric("Expected Fraud Loss",  f"${total_fraud_loss:,.0f}", f"{expected_fraud_claims} claims")
    k3.metric("Net Savings (ML)",     f"${net_savings:,.0f}", f"ROI: {roi:.0f}%")
    k4.metric("Loss Ratio w/o ML",    f"{(total_fraud_loss/total_premium):.1%}", "Fraud only")

    st.markdown("---")

    # ── Waterfall chart ───────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 🌊 Financial Waterfall — Annual P&L Impact")
        waterfall_vals = [
            total_premium,
            -total_fraud_loss,
            savings_from_detection,
            -investigation_costs,
            net_savings,
        ]
        waterfall_labels = [
            "Premium Income",
            "Gross Fraud Loss",
            "Fraud Recovered",
            "Investigation Costs",
            "Net Benefit",
        ]
        colors = ["#00e5ff", "#ff1744", "#00e676", "#ff6b2b", "#00e676" if net_savings > 0 else "#ff1744"]
        measures = ["absolute", "relative", "relative", "relative", "total"]

        fig_wf = go.Figure(go.Waterfall(
            name="P&L",
            orientation="v",
            measure=measures,
            x=waterfall_labels,
            y=waterfall_vals,
            text=[f"${abs(v):,.0f}" for v in waterfall_vals],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#8899aa"),
            connector=dict(line=dict(color="#2a3a55", width=1)),
            increasing=dict(marker=dict(color="#00e676")),
            decreasing=dict(marker=dict(color="#ff1744")),
            totals=dict(marker=dict(color="#00e5ff")),
        ))
        fig_wf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            yaxis=dict(gridcolor="#1a2235", title="USD"),
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Detection Outcome Breakdown")
        outcome_labels = ["True Positives\n(Fraud Caught)", "False Negatives\n(Missed Fraud)",
                          "False Positives\n(False Alarms)"]
        outcome_vals   = [detected, missed, fp_count]
        outcome_colors = ["#00e676", "#ff1744", "#ff6b2b"]

        fig_out = go.Figure(go.Bar(
            x=outcome_labels, y=outcome_vals,
            marker_color=outcome_colors,
            text=[str(v) for v in outcome_vals],
            textposition="outside",
            textfont=dict(family="Syne", size=14),
        ))
        fig_out.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            yaxis=dict(gridcolor="#1a2235"),
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_out, use_container_width=True)

        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace; font-size:0.82rem; color:#8899aa; line-height:1.8;">
          <b style="color:#00e5ff;">Fraud Caught:</b> {detected:,} claims (${detected*avg_fraud_claim:,.0f} saved)<br>
          <b style="color:#ff1744;">Fraud Missed:</b> {missed:,} claims (${missed*avg_fraud_claim:,.0f} lost)<br>
          <b style="color:#ff6b2b;">False Alarms:</b> {fp_count:,} (${fp_count*investigation_cost:,.0f} cost)<br>
          <b style="color:#00e676;">Net ROI:</b> {roi:.0f}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Break-even & sensitivity ──────────────────────────────────────────────
    st.markdown("#### 📐 Sensitivity Analysis — Net Savings vs Detection Rate")

    recall_range = np.linspace(0.40, 0.99, 60)
    net_savings_range = []
    for r in recall_range:
        det = expected_fraud_claims * r
        fp  = portfolio_size * (1 - fraud_prevalence) * fp_rate
        ns  = det * avg_fraud_claim - (det + fp) * investigation_cost
        net_savings_range.append(ns)

    breakeven_idx = next((i for i, v in enumerate(net_savings_range) if v > 0), None)
    breakeven_recall = recall_range[breakeven_idx] if breakeven_idx else None

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=recall_range, y=net_savings_range,
        mode="lines", name="Net Savings",
        line=dict(color="#00e5ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.07)",
    ))
    fig_sens.add_hline(y=0, line=dict(color="#2a3a55", width=1, dash="dash"))
    if breakeven_recall:
        fig_sens.add_vline(
            x=breakeven_recall,
            line=dict(color="#ffd600", width=1.5, dash="dot"),
            annotation_text=f"Break-even @ {breakeven_recall:.0%}",
            annotation_font=dict(family="DM Mono", size=10, color="#ffd600"),
        )
    fig_sens.add_vline(
        x=detection_rate,
        line=dict(color="#00e676", width=2),
        annotation_text=f"Current model: {detection_rate:.0%}",
        annotation_font=dict(family="DM Mono", size=10, color="#00e676"),
    )
    fig_sens.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899aa", family="DM Mono"),
        xaxis=dict(title="Model Recall (Detection Rate)", tickformat=".0%", gridcolor="#1a2235"),
        yaxis=dict(title="Net Savings (USD)", gridcolor="#1a2235"),
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    # ── Fraud loss by cause of death ──────────────────────────────────────────
    st.markdown("#### 🏥 Fraud Exposure by Cause of Death")
    cod = df[df["is_fraud"] == 1].groupby("cause_of_death").agg(
        count=("is_fraud", "count"),
        exposure=("claim_amount_usd", "sum"),
    ).reset_index().sort_values("exposure", ascending=False)

    fig_cod = go.Figure(go.Bar(
        x=cod["cause_of_death"], y=cod["exposure"],
        marker=dict(
            color=cod["exposure"],
            colorscale=[[0, "#1a2235"], [1, "#ff6b2b"]],
            showscale=False,
        ),
        text=[f"${v:,.0f}" for v in cod["exposure"]],
        textposition="outside",
        textfont=dict(family="DM Mono", size=10),
        customdata=cod["count"],
        hovertemplate="<b>%{x}</b><br>Exposure: $%{y:,.0f}<br>Cases: %{customdata}<extra></extra>",
    ))
    fig_cod.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899aa", family="DM Mono"),
        yaxis=dict(gridcolor="#1a2235", title="Fraud Exposure (USD)"),
        xaxis=dict(tickangle=-30),
        height=340,
        margin=dict(l=0, r=0, t=20, b=60),
    )
    st.plotly_chart(fig_cod, use_container_width=True)

    # ── Loss ratio projection ─────────────────────────────────────────────────
    st.markdown("#### 📅 Projected Loss Ratio Improvement — 5-Year Horizon")

    years = list(range(2026, 2031))
    # Without ML: fraud creep 8% per year
    no_ml = [total_fraud_loss * (1.08 ** i) for i in range(5)]
    # With ML: savings compound as model improves
    with_ml = [max(total_fraud_loss * (1.08 ** i) - savings_from_detection * (1.05 ** i), 0)
               for i in range(5)]

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=years, y=no_ml, name="Without ML System",
        mode="lines+markers",
        line=dict(color="#ff1744", width=2, dash="dot"),
        marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(255,23,68,0.06)",
    ))
    fig_proj.add_trace(go.Scatter(
        x=years, y=with_ml, name="With FraudIQ ML",
        mode="lines+markers",
        line=dict(color="#00e676", width=2.5),
        marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
    ))
    fig_proj.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899aa", family="DM Mono"),
        xaxis=dict(gridcolor="#1a2235", title="Year", dtick=1),
        yaxis=dict(gridcolor="#1a2235", title="Fraud Loss (USD)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    savings_5yr = sum(no_ml) - sum(with_ml)
    st.markdown(f"""
    <div class="safe-alert">
      ✅ Over 5 years, the FraudIQ ML system is projected to save
      <b>${savings_5yr:,.0f}</b> in fraud losses across a portfolio of {portfolio_size:,} policies —
      supporting H₁ that ML-based systems significantly reduce fraud-related losses
      compared to traditional manual review methods.
    </div>
    """, unsafe_allow_html=True)
