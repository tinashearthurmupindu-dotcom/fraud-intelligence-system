"""pages/eda.py — Exploratory Data Analysis & Risk Signal Discovery"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import generate_claims_dataset


@st.cache_data(show_spinner=False)
def load_data():
    return generate_claims_dataset(2000)


def render():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; margin-bottom:4px;">
      📈 Exploratory Data Analysis & Risk Signals
    </h1>
    <p style="color:#8899aa; font-family:'DM Mono',monospace; font-size:0.85rem; margin-bottom:24px;">
      Objective 1: Identifying claim-level features most consistently associated with fraud
    </p>
    """, unsafe_allow_html=True)

    df = load_data()
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]

    tabs = st.tabs([
        "📊 Univariate",
        "🔗 Correlations",
        "⚖️ Statistical Tests",
        "🌡️ Risk Heatmap",
        "🔍 Feature Deep-Dive",
    ])

    # ── Tab 1: Univariate ─────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Distribution of Key Features by Fraud Status")

        num_cols = [
            "policy_age_months", "claim_amount_usd",
            "days_to_claim", "documentation_score",
            "prior_claims", "num_policies_same_beneficiary",
        ]
        labels = [
            "Policy Age (months)", "Claim Amount (USD)",
            "Days: Death → Claim", "Documentation Score",
            "Prior Claims", "Policies per Beneficiary",
        ]

        col_sel = st.selectbox("Select feature to inspect", labels, index=0)
        feat = num_cols[labels.index(col_sel)]

        fig = go.Figure()
        for subset, name, color in [
            (legit, "Legitimate", "#00e676"),
            (fraud, "Fraud", "#ff6b2b"),
        ]:
            fig.add_trace(go.Violin(
                x=[name] * len(subset),
                y=subset[feat],
                name=name,
                box_visible=True,
                meanline_visible=True,
                fillcolor="rgba({},{},{},0.25)".format(int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)),
                line_color=color,
                points="outliers",
            ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            yaxis=dict(gridcolor="#1a2235", title=col_sel),
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            violinmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats table
        st.markdown("##### Descriptive Statistics")
        summary_rows = []
        for feat_name, label in zip(num_cols, labels):
            row = {"Feature": label}
            for subset, sname in [(legit, "Legit"), (fraud, "Fraud")]:
                vals = subset[feat_name]
                row[f"{sname} Mean"] = f"{vals.mean():.2f}"
                row[f"{sname} Median"] = f"{vals.median():.2f}"
                row[f"{sname} Std"] = f"{vals.std():.2f}"
            summary_rows.append(row)

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Tab 2: Correlations ───────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### Correlation Matrix — Numeric Features + Fraud Label")

        corr_cols = [
            "policy_age_months", "claim_amount_usd", "claimant_age",
            "beneficiary_count", "days_to_claim", "prior_claims",
            "num_policies_same_beneficiary", "documentation_score",
            "registry_verified", "staff_relationship_flag",
            "same_day_claim", "cross_insurer_duplicate", "is_fraud",
        ]
        short = [
            "PolicyAge", "ClaimAmt", "ClaimantAge",
            "Beneficiaries", "DaysToClaim", "PriorClaims",
            "PoliciesPerBene", "DocScore",
            "RegistryVerif", "StaffFlag",
            "SameDayClaim", "CrossDup", "IS_FRAUD",
        ]

        corr_matrix = df[corr_cols].corr()
        corr_matrix.index = short
        corr_matrix.columns = short

        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=short, y=short,
            colorscale=[
                [0.0, "#ff1744"],
                [0.5, "#0a0e17"],
                [1.0, "#00e5ff"],
            ],
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont=dict(size=8, family="DM Mono"),
            showscale=True,
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono", size=9),
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlates with fraud
        fraud_corrs = corr_matrix["IS_FRAUD"].drop("IS_FRAUD").sort_values(key=abs, ascending=False)
        st.markdown("##### Top Correlates with Fraud Label")
        fig_fc = go.Figure(go.Bar(
            x=fraud_corrs.index,
            y=fraud_corrs.values,
            marker_color=["#ff6b2b" if v > 0 else "#00e5ff" for v in fraud_corrs.values],
            text=[f"{v:.3f}" for v in fraud_corrs.values],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10),
        ))
        fig_fc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            yaxis=dict(gridcolor="#1a2235", zeroline=True, zerolinecolor="#2a3a55"),
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    # ── Tab 3: Statistical Tests ───────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### Statistical Significance Testing")
        st.markdown("""
        <p style="color:#8899aa; font-family:'DM Mono',monospace; font-size:0.83rem;">
        Point-biserial correlations for continuous features · Chi-squared for categorical features<br>
        Per §3.4.1 of the research proposal (Saunders et al., 2019)
        </p>
        """, unsafe_allow_html=True)

        cont_feats = [
            ("policy_age_months", "Policy Age (months)"),
            ("claim_amount_usd", "Claim Amount (USD)"),
            ("claimant_age", "Claimant Age"),
            ("days_to_claim", "Days to Claim"),
            ("prior_claims", "Prior Claims"),
            ("documentation_score", "Documentation Score"),
            ("beneficiary_count", "Beneficiary Count"),
            ("num_policies_same_beneficiary", "Policies per Beneficiary"),
        ]

        test_rows = []
        for col, label in cont_feats:
            pb_corr, p_val = stats.pointbiserialr(df["is_fraud"], df[col])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            test_rows.append({
                "Feature": label,
                "Point-Biserial r": f"{pb_corr:.4f}",
                "p-value": f"{p_val:.4e}",
                "Significance": sig,
                "Direction": "↑ Fraud" if pb_corr > 0 else "↓ Fraud",
            })

        # Categorical chi-squared
        for col, label in [
            ("registry_verified", "Registry Verified"),
            ("staff_relationship_flag", "Staff Relationship"),
            ("same_day_claim", "Same-Day Claim"),
            ("cross_insurer_duplicate", "Cross-Insurer Dup."),
        ]:
            ct = pd.crosstab(df[col], df["is_fraud"])
            chi2, p_val, _, _ = stats.chi2_contingency(ct)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            test_rows.append({
                "Feature": label + " [χ²]",
                "Point-Biserial r": f"χ²={chi2:.2f}",
                "p-value": f"{p_val:.4e}",
                "Significance": sig,
                "Direction": "Categorical",
            })

        test_df = pd.DataFrame(test_rows)

        def color_sig(val):
            if val == "***": return "color: #00e676; font-weight:700"
            if val == "**":  return "color: #00e5ff; font-weight:700"
            if val == "*":   return "color: #ffd600"
            return "color: #8899aa"

        st.dataframe(
            test_df.style.map(color_sig, subset=["Significance"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
        <div class="info-alert">
          *** p &lt; 0.001 · ** p &lt; 0.01 · * p &lt; 0.05 · ns = not significant<br>
          Features with *** are primary candidates for the ML feature set.
          All significance tests conducted on the full dataset prior to train-test split.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Risk Heatmap ───────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### 🌡️ Fraud Rate Heatmap — Policy Age × Claim Amount")

        df["age_bin"]  = pd.cut(df["policy_age_months"],
                                bins=[0, 6, 12, 24, 48, 120],
                                labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"])
        df["amt_bin"]  = pd.cut(df["claim_amount_usd"],
                                bins=[0, 2000, 5000, 10000, 25000, 100000],
                                labels=["<$2K", "$2K-5K", "$5K-10K", "$10K-25K", ">$25K"])

        pivot = df.pivot_table(
            values="is_fraud", index="amt_bin", columns="age_bin",
            aggfunc="mean", observed=True
        )

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale=[[0, "#0a0e17"], [0.3, "#1a2235"], [0.7, "#ff6b2b"], [1.0, "#ff1744"]],
            text=[[f"{v:.1f}%" for v in row] for row in pivot.values * 100],
            texttemplate="%{text}",
            textfont=dict(size=12, family="Syne", color="white"),
            colorbar=dict(
                title="Fraud Rate %",
                tickfont=dict(family="DM Mono", size=10),
            ),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            xaxis=dict(title="Policy Age at Claim"),
            yaxis=dict(title="Claim Amount"),
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="fraud-alert">
          🔴 Highest fraud concentration: <b>New policies (0–6 months) with large claims (&gt;$10K)</b><br>
          Aligns with literature: "policy inception dates adjusted to get around waiting periods"
          — Manje (2005). This is the primary fraud typology in SADC funeral insurance markets.
        </div>
        """, unsafe_allow_html=True)

        # Region × fraud rate
        st.markdown("#### 🌍 Fraud Rate by Region")
        reg = df.groupby("region")["is_fraud"].agg(["mean", "sum", "count"]).reset_index()
        reg.columns = ["Region", "FraudRate", "FraudCount", "TotalClaims"]
        reg = reg.sort_values("FraudRate", ascending=False)

        fig_reg = px.treemap(
            reg, path=["Region"], values="TotalClaims",
            color="FraudRate",
            color_continuous_scale=[[0, "#0a0e17"], [0.5, "#ff6b2b"], [1.0, "#ff1744"]],
            hover_data={"FraudCount": True, "FraudRate": ":.1%"},
        )
        fig_reg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(tickformat=".1%", title="Fraud Rate"),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    # ── Tab 5: Feature Deep-Dive ───────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### Feature-Level Fraud Risk Breakdown")

        feat_choice = st.selectbox("Select risk feature", [
            "prior_claims", "policy_age_months_binned",
            "documentation_score_binned", "cross_insurer_duplicate",
            "registry_verified", "staff_relationship_flag",
        ])

        if feat_choice == "prior_claims":
            grp = df.groupby("prior_claims")["is_fraud"].mean().reset_index()
            grp.columns = ["Value", "Fraud Rate"]
            x_title = "Number of Prior Claims"
        elif feat_choice == "policy_age_months_binned":
            df["_bin"] = pd.cut(df["policy_age_months"],
                                bins=[0, 3, 6, 12, 24, 48, 120],
                                labels=["0-3m", "3-6m", "6-12m", "1-2y", "2-4y", "4y+"])
            grp = df.groupby("_bin", observed=True)["is_fraud"].mean().reset_index()
            grp.columns = ["Value", "Fraud Rate"]
            x_title = "Policy Age at Claim"
        elif feat_choice == "documentation_score_binned":
            df["_bin"] = pd.cut(df["documentation_score"],
                                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
            grp = df.groupby("_bin", observed=True)["is_fraud"].mean().reset_index()
            grp.columns = ["Value", "Fraud Rate"]
            x_title = "Documentation Score"
        else:
            grp = df.groupby(feat_choice)["is_fraud"].mean().reset_index()
            grp.columns = ["Value", "Fraud Rate"]
            x_title = feat_choice.replace("_", " ").title()

        fig_dd = go.Figure(go.Bar(
            x=grp["Value"].astype(str),
            y=grp["Fraud Rate"],
            marker_color=[
                "#ff1744" if v > 0.15 else "#ff6b2b" if v > 0.07 else "#00e676"
                for v in grp["Fraud Rate"]
            ],
            text=[f"{v:.1%}" for v in grp["Fraud Rate"]],
            textposition="outside",
            textfont=dict(family="DM Mono", size=11),
        ))
        fig_dd.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa", family="DM Mono"),
            xaxis=dict(title=x_title, gridcolor="#1a2235"),
            yaxis=dict(title="Fraud Rate", tickformat=".0%", gridcolor="#1a2235"),
            height=340,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_dd, use_container_width=True)
