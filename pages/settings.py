"""pages/settings.py — System Configuration & About"""

import streamlit as st
import pandas as pd
from datetime import datetime


def render():
    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; margin-bottom:4px;">
      ⚙️ System Settings & About
    </h1>
    <p style="color:#8899aa; font-family:'DM Mono',monospace; font-size:0.85rem; margin-bottom:24px;">
      Configure thresholds, review data sources, and access research documentation
    </p>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🎛️ Thresholds", "📚 Research Info", "🔌 Data Sources", "📋 System Info"])

    # ── Thresholds ────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Detection Threshold Configuration")
        st.markdown("""
        <p style="color:#8899aa; font-family:'DM Mono',monospace; font-size:0.83rem;">
        Adjust the decision boundaries that map fraud probability scores to risk tiers.
        Lowering HIGH threshold increases sensitivity (more fraud caught) but increases false positives.
        </p>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            low_threshold  = st.slider("LOW → MEDIUM boundary", 0.20, 0.60, 0.45, 0.01,
                                       help="Claims below this score are approved automatically")
            high_threshold = st.slider("MEDIUM → HIGH boundary", 0.50, 0.90, 0.70, 0.01,
                                       help="Claims above this score are suspended immediately")
            max_auto_approve = st.number_input("Max auto-approve amount (USD)", 0, 50000, 2000, 500,
                                               help="Claims above this amount always require manual review regardless of score")

        with col2:
            st.markdown("##### Threshold Impact Preview")
            import numpy as np, plotly.graph_objects as go
            scores = np.linspace(0, 1, 200)
            fig_thresh = go.Figure()
            fig_thresh.add_vrect(x0=0, x1=low_threshold,
                                  fillcolor="rgba(0,230,118,0.10)", line_width=0,
                                  annotation_text="AUTO-APPROVE", annotation_position="top left",
                                  annotation_font=dict(color="#00e676", family="DM Mono", size=10))
            fig_thresh.add_vrect(x0=low_threshold, x1=high_threshold,
                                  fillcolor="rgba(255,107,43,0.10)", line_width=0,
                                  annotation_text="FLAG FOR REVIEW", annotation_position="top left",
                                  annotation_font=dict(color="#ff6b2b", family="DM Mono", size=10))
            fig_thresh.add_vrect(x0=high_threshold, x1=1.0,
                                  fillcolor="rgba(255,23,68,0.10)", line_width=0,
                                  annotation_text="SUSPEND", annotation_position="top left",
                                  annotation_font=dict(color="#ff1744", family="DM Mono", size=10))
            fig_thresh.add_vline(x=low_threshold, line=dict(color="#ff6b2b", width=2, dash="dash"))
            fig_thresh.add_vline(x=high_threshold, line=dict(color="#ff1744", width=2, dash="dash"))
            fig_thresh.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8899aa", family="DM Mono"),
                xaxis=dict(title="Fraud Score", range=[0, 1], gridcolor="#1a2235"),
                yaxis=dict(visible=False),
                height=200,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_thresh, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Alert & Notification Settings")
        col3, col4 = st.columns(2)
        with col3:
            st.toggle("Email alerts for HIGH risk claims", value=True)
            st.toggle("Daily fraud summary report", value=True)
            st.toggle("Weekly model performance digest", value=False)
            st.text_input("Alert email address", value="fraud.team@insurer.zm")
        with col4:
            st.toggle("Slack integration", value=False)
            st.toggle("Auto-generate investigation report (PDF)", value=True)
            st.toggle("Cross-insurer API sync (ZIBA registry)", value=False)
            st.number_input("Investigation SLA (hours)", 1, 168, 48)

        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved — thresholds updated across all scoring pipelines")

    # ── Research Info ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### 📚 Research Background")
        col_info1, col_info2 = st.columns(2)

        with col_info1:
            st.markdown("""
            <div style="background:#1a2235; border:1px solid #2a3a55; border-radius:10px; padding:20px;">
              <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#00e5ff; margin-bottom:12px;">
                Research Study
              </div>
              <div style="font-family:'DM Mono',monospace; font-size:0.82rem; color:#8899aa; line-height:1.9;">
                <b style="color:#e8edf5;">Title:</b> Using Machine Learning and Predictive Analytics in
                Detecting Fraudulent Funeral Insurance Claims in Zambia<br><br>
                <b style="color:#e8edf5;">Author:</b> Tinashe Arthur Mupindu<br>
                <b style="color:#e8edf5;">Student ID:</b> 2022996712<br>
                <b style="color:#e8edf5;">Degree:</b> MSc Applied Statistics<br>
                <b style="color:#e8edf5;">Institution:</b> University of Zambia (UNZA)<br>
                <b style="color:#e8edf5;">Department:</b> Mathematics & Statistics<br>
                <b style="color:#e8edf5;">Supervisors:</b> Dr Stanley Jere & Dr T Chowa<br>
                <b style="color:#e8edf5;">Submitted:</b> April 2026
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_info2:
            st.markdown("""
            <div style="background:#1a2235; border:1px solid #2a3a55; border-radius:10px; padding:20px;">
              <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#00e5ff; margin-bottom:12px;">
                Research Objectives & Hypotheses
              </div>
              <div style="font-family:'DM Mono',monospace; font-size:0.82rem; color:#8899aa; line-height:1.9;">
                <b style="color:#e8edf5;">RQ1:</b> Which claim-level features are most associated with fraud?<br>
                <b style="color:#e8edf5;">RQ2:</b> How do ML algorithms compare on recall & AUC-ROC?<br>
                <b style="color:#e8edf5;">RQ3:</b> What is the financial impact of ML-based detection?<br><br>
                <b style="color:#00e5ff;">H₁ (Accept):</b> ML models outperform logistic regression baseline<br>
                <b style="color:#00e5ff;">H₂ (Test):</b> SMOTE significantly improves minority-class recall<br>
                <b style="color:#00e5ff;">H₃ (Test):</b> ML detection reduces fraud losses vs manual review
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📖 Key References")
        refs = [
            ("Nabrawi & Alanazi (2023)", "Fraud detection in healthcare insurance using data mining",
             "Healthcare", "XGBoost AUC 0.97"),
            ("Debener, Heinke & Kriebel (2023)", "Detecting insurance fraud using ML — property/casualty",
             "Property Insurance", "RF+XGB ensemble"),
            ("Khalil (2025)", "Life insurance fraud detection using ML",
             "Life Insurance", "AUC 0.96"),
            ("Manje (2005)", "Funeral insurance fraud typologies in SADC",
             "Funeral Insurance", "5 fraud typologies"),
            ("Marumo (2016)", "Insurance fraud in the African context",
             "African Markets", "Qualitative review"),
            ("Lundberg & Lee (2017)", "A unified approach to interpreting model predictions (SHAP)",
             "ML Theory", "NeurIPS"),
            ("IAIS (2023)", "Issues Paper on AI in Insurance",
             "Regulation", "Explainability mandate"),
        ]

        ref_df = pd.DataFrame(refs, columns=["Citation", "Title", "Domain", "Key Contribution"])
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

    # ── Data Sources ──────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### 🔌 Data Sources & Integrations")

        sources = [
            {
                "Source": "Zambia IRA Regulatory Database",
                "Type": "Primary",
                "Status": "🔴 Simulated",
                "Description": "Licensed insurer data — surrenders, claims, complaints",
                "Rows": "N/A (proprietary)",
            },
            {
                "Source": "ZIBA Cross-Insurer Registry",
                "Type": "Primary",
                "Status": "🔴 Simulated",
                "Description": "National cross-insurer duplicate claim detection",
                "Rows": "N/A",
            },
            {
                "Source": "Synthetic SADC Claims Dataset",
                "Type": "Generated",
                "Status": "✅ Active",
                "Description": "2,000 synthetic claims generated per proposal §3.2",
                "Rows": "2,000",
            },
            {
                "Source": "National Death Registry (GRZ)",
                "Type": "External",
                "Status": "🔴 Simulated",
                "Description": "Death certificate verification",
                "Rows": "N/A",
            },
        ]
        st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="fraud-alert">
          ⚠️ <b>Data Privacy Notice:</b> This system uses a fully synthetic dataset generated
          per the ethical approval conditions of the UNZA Research Ethics Committee.
          No real policyholder data is stored or processed in this demo.
          All claims data is generated algorithmically to preserve privacy per Zambia's
          <i>Data Protection Act (2021)</i>.
        </div>
        """, unsafe_allow_html=True)

    # ── System Info ───────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### 📋 System Information")
        col_sys1, col_sys2 = st.columns(2)

        with col_sys1:
            st.markdown("""
            <div style="background:#1a2235; border:1px solid #2a3a55; border-radius:10px; padding:20px;">
              <div style="font-family:'Syne',sans-serif; color:#00e5ff; margin-bottom:10px;">Tech Stack</div>
              <div style="font-family:'DM Mono',monospace; font-size:0.82rem; color:#8899aa; line-height:2.0;">
                🐍 Python 3.11+<br>
                📊 Streamlit 1.35+<br>
                🤖 scikit-learn 1.4<br>
                ⚡ XGBoost 2.0<br>
                ⚖️ imbalanced-learn (SMOTE)<br>
                📈 Plotly 5.21<br>
                🔢 NumPy / Pandas<br>
                📐 SciPy (statistical tests)<br>
                🔍 SHAP 0.44
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sys2:
            st.markdown("""
            <div style="background:#1a2235; border:1px solid #2a3a55; border-radius:10px; padding:20px;">
              <div style="font-family:'Syne',sans-serif; color:#00e5ff; margin-bottom:10px;">Model Pipeline</div>
              <div style="font-family:'DM Mono',monospace; font-size:0.82rem; color:#8899aa; line-height:2.0;">
                📥 Data ingestion & validation<br>
                🔧 Feature engineering (12 features)<br>
                ⚖️ SMOTE balancing (train only)<br>
                📏 StandardScaler (LR, SVM)<br>
                🏋️ 5-model training suite<br>
                📊 10-fold stratified CV<br>
                🎯 Threshold optimisation<br>
                🧠 SHAP explainability layer<br>
                📤 REST API scoring endpoint
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace; font-size:0.8rem; color:#8899aa; text-align:center;">
          FraudIQ v2.0 · Built for UNZA MSc Applied Statistics Research 2026<br>
          System time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC<br>
          <span style="color:#00e5ff;">© 2026 Tinashe Arthur Mupindu — For academic research purposes only</span>
        </div>
        """, unsafe_allow_html=True)
