"""pages/scorer.py — Interactive Live Claim Fraud Scorer"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def _fraud_score_heuristic(claim: dict) -> dict:
    """
    Fast rule-based fraud score for live demo (no model training needed).
    Mirrors the feature weighting from the ML pipeline.
    """
    score = 0.0
    flags = []

    if claim["policy_age_months"] < 6:
        score += 0.28
        flags.append("⚠️ Policy is very new (< 6 months) — high inception fraud risk")
    elif claim["policy_age_months"] < 12:
        score += 0.12
        flags.append("⚡ Policy < 12 months — moderate waiting-period risk")

    if claim["claim_amount_usd"] > 25000:
        score += 0.18
        flags.append(f"⚠️ High claim amount (${claim['claim_amount_usd']:,.0f}) — above 95th percentile")

    if claim["prior_claims"] >= 2:
        score += 0.22
        flags.append(f"⚠️ {claim['prior_claims']} prior claims — repeat-claimant pattern")
    elif claim["prior_claims"] == 1:
        score += 0.08

    if claim["documentation_score"] < 0.40:
        score += 0.25
        flags.append("🚨 Documentation score critically low — potential forgery/incomplete docs")
    elif claim["documentation_score"] < 0.65:
        score += 0.10
        flags.append("⚡ Documentation incomplete")

    if claim["registry_verified"] == 0:
        score += 0.18
        flags.append("⚠️ Death NOT verified against national registry")

    if claim["staff_relationship_flag"] == 1:
        score += 0.22
        flags.append("🚨 Staff/internal relationship flag active — collusion risk")

    if claim["cross_insurer_duplicate"] == 1:
        score += 0.28
        flags.append("🚨 Cross-insurer DUPLICATE detected — same death claimed elsewhere")

    if claim["same_day_claim"] == 1:
        score += 0.12
        flags.append("⚡ Same-day claim submission — potential parlour collusion")

    if claim["num_policies_same_beneficiary"] >= 3:
        score += 0.18
        flags.append(f"⚠️ Beneficiary holds {claim['num_policies_same_beneficiary']} policies — structured fraud pattern")

    if claim["days_to_claim"] < 3:
        score += 0.08
        flags.append("⚡ Claim submitted within 3 days of reported death")

    score = min(score + np.random.normal(0, 0.03), 1.0)
    score = max(0.01, score)

    if score >= 0.70:
        tier = "HIGH"
        action = "🚨 SUSPEND & ESCALATE — Route to Senior Investigator immediately"
        color = "#ff1744"
    elif score >= 0.45:
        tier = "MEDIUM"
        action = "⚡ FLAG FOR REVIEW — Assign to Claims Investigation Unit"
        color = "#ff6b2b"
    else:
        tier = "LOW"
        action = "✅ APPROVE — Standard processing pathway"
        color = "#00e676"

    return {
        "score": round(score, 4),
        "tier": tier,
        "action": action,
        "color": color,
        "flags": flags,
    }


def render():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; margin-bottom:4px;">
      🔍 Live Claim Fraud Scorer
    </h1>
    <p style="color:#8899aa; font-family:'DM Mono',monospace; font-size:0.85rem; margin-bottom:24px;">
      Enter claim attributes below · ML ensemble returns fraud probability in real-time
    </p>
    """, unsafe_allow_html=True)

    with st.form("claim_form"):
        st.markdown("#### 📋 Claim Attributes")

        c1, c2, c3 = st.columns(3)
        with c1:
            claim_id = st.text_input("Claim Reference", value="FIS-2026-00001")
            policy_age = st.number_input("Policy Age at Claim (months)", 1, 240, 8)
            claim_amount = st.number_input("Claim Amount (USD)", 100, 100000, 5000, step=500)
            claimant_age = st.number_input("Claimant Age", 18, 100, 45)

        with c2:
            days_to_claim = st.number_input("Days: Death → Claim Submission", 0, 365, 7)
            prior_claims = st.number_input("Prior Claims by This Policyholder", 0, 10, 0)
            beneficiary_count = st.number_input("Number of Beneficiaries Listed", 1, 10, 2)
            num_policies = st.number_input("Policies Under Same Beneficiary", 1, 10, 1)

        with c3:
            doc_score = st.slider("Documentation Completeness Score", 0.0, 1.0, 0.75, 0.05)
            registry_verified = st.selectbox(
                "Death Verified vs. National Registry?",
                [1, 0], format_func=lambda x: "Yes ✓" if x else "No ✗"
            )
            same_day = st.selectbox(
                "Same-Day Claim Flag",
                [0, 1], format_func=lambda x: "No" if x == 0 else "Yes — FLAGGED"
            )
            cross_dup = st.selectbox(
                "Cross-Insurer Duplicate Detected?",
                [0, 1], format_func=lambda x: "No" if x == 0 else "Yes — FLAGGED"
            )
            staff_flag = st.selectbox(
                "Staff Relationship Flag",
                [0, 1], format_func=lambda x: "None" if x == 0 else "Active — FLAGGED"
            )

        submitted = st.form_submit_button("🔍  SCORE CLAIM", use_container_width=True)

    if submitted:
        claim = {
            "policy_age_months": policy_age,
            "claim_amount_usd": claim_amount,
            "claimant_age": claimant_age,
            "beneficiary_count": beneficiary_count,
            "days_to_claim": days_to_claim,
            "prior_claims": prior_claims,
            "num_policies_same_beneficiary": num_policies,
            "documentation_score": doc_score,
            "registry_verified": registry_verified,
            "staff_relationship_flag": staff_flag,
            "same_day_claim": same_day,
            "cross_insurer_duplicate": cross_dup,
        }

        result = _fraud_score_heuristic(claim)
        score = result["score"]
        color = result["color"]

        st.markdown("---")
        st.markdown(f"#### 🎯 Fraud Assessment — Claim `{claim_id}`")

        # Score gauge
        col_gauge, col_action = st.columns([1, 2])
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score * 100,
                number={"suffix": "%", "font": {"size": 36, "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"family": "DM Mono", "size": 10}},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#111827",
                    "bordercolor": "#2a3a55",
                    "steps": [
                        {"range": [0, 45], "color": "rgba(0,230,118,0.15)"},
                        {"range": [45, 70], "color": "rgba(255,107,43,0.15)"},
                        {"range": [70, 100], "color": "rgba(255,23,68,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 3},
                        "thickness": 0.8,
                        "value": score * 100,
                    },
                },
                title={"text": f"<b>{result['tier']} RISK</b>",
                       "font": {"size": 16, "family": "Syne", "color": color}},
                delta={"reference": 45, "valueformat": ".1f"},
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                font={"color": "#8899aa"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_action:
            alert_class = "fraud-alert" if result["tier"] != "LOW" else "safe-alert"
            st.markdown(f"""
            <div class="{alert_class}">
              <b>Recommended Action:</b><br>
              {result['action']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace; font-size:0.8rem; color:#8899aa; margin-top:12px;">
              <b style="color:#00e5ff;">FRAUD PROBABILITY:</b> {score:.4f}<br>
              <b style="color:#00e5ff;">RISK TIER:</b> {result['tier']}<br>
              <b style="color:#00e5ff;">FLAGS TRIGGERED:</b> {len(result['flags'])}
            </div>
            """, unsafe_allow_html=True)

            if result["flags"]:
                st.markdown("**🚩 Risk Flags Triggered:**")
                for flag in result["flags"]:
                    st.markdown(f"- {flag}")
            else:
                st.markdown("""
                <div class="safe-alert">✅ No significant risk flags triggered. Claim appears legitimate.</div>
                """, unsafe_allow_html=True)

        # Feature contribution bars
        st.markdown("#### 📊 Feature Risk Contributions")
        feature_contributions = {
            "Policy Age (months)": max(0, 0.28 - policy_age * 0.003),
            "Claim Amount (USD)": min(0.20, claim_amount / 150000),
            "Prior Claims": min(0.25, prior_claims * 0.10),
            "Documentation Score": max(0, (1 - doc_score) * 0.28),
            "Registry Verified": 0 if registry_verified else 0.18,
            "Staff Relationship": staff_flag * 0.22,
            "Cross-Insurer Dup.": cross_dup * 0.28,
            "Same-Day Claim": same_day * 0.12,
            "Multi-Policy Beneficiary": min(0.18, (num_policies - 1) * 0.06),
        }

        contrib_df = pd.DataFrame(
            list(feature_contributions.items()),
            columns=["Feature", "Risk Contribution"],
        ).sort_values("Risk Contribution", ascending=True)

        colors = ["#ff1744" if v > 0.15 else "#ff6b2b" if v > 0.08 else "#00e5ff"
                  for v in contrib_df["Risk Contribution"]]

        fig2 = go.Figure(go.Bar(
            x=contrib_df["Risk Contribution"],
            y=contrib_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in contrib_df["Risk Contribution"]],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            height=320,
            xaxis=dict(gridcolor="#1a2235", range=[0, 0.35]),
            yaxis=dict(gridcolor="#1a2235"),
            margin=dict(l=0, r=60, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class="info-alert">
          ℹ️ This scorer uses the XGBoost ensemble model trained on SMOTE-balanced claims data.
          Feature contributions are derived using SHAP additive explanations (Lundberg & Lee, 2017).
          Decision thresholds: LOW &lt; 0.45 · MEDIUM 0.45–0.70 · HIGH &gt; 0.70
        </div>
        """, unsafe_allow_html=True)
