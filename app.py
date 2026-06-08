import streamlit as st

st.set_page_config(
    page_title="FraudIQ - Funeral Insurance Intelligence",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    background-color: #0a0e17 !important;
    color: #e8edf5;
  }
  .main {
    background-color: #0a0e17 !important;
  }
  .block-container {
    background-color: #0a0e17 !important;
    padding-top: 2rem;
  }
  [data-testid="stAppViewContainer"] {
    background-color: #0a0e17 !important;
  }
  [data-testid="stMain"] {
    background-color: #0a0e17 !important;
  }
  h1, h2, h3, h4, h5 {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    color: #ffffff;
  }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stToolbar"] { display: none; }
  div[data-testid="stDecoration"] { display: none; }
  div[data-testid="stStatusWidget"] { display: none; }
  div[data-testid="stSidebarNav"] { display: none !important; }
  button[data-testid="collapsedControl"] { display: none !important; }

  section[data-testid="stSidebar"] {
    background: #0d1b2a !important;
    border-right: 2px solid #00e5ff !important;
    display: block !important;
    visibility: visible !important;
    width: 280px !important;
    min-width: 280px !important;
    margin-left: 0 !important;
    transform: none !important;
  }
  section[data-testid="stSidebar"] > div {
    padding-top: 0px !important;
  }

  div[data-testid="stSidebar"] .stRadio > div {
    gap: 5px !important;
  }
  div[data-testid="stSidebar"] .stRadio > div > label {
    background: #1a2e45 !important;
    border: 1px solid #2a5a8a !important;
    border-radius: 8px !important;
    padding: 11px 16px !important;
    color: #ffffff !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    display: block !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    opacity: 1 !important;
  }
  div[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #1e3d5c !important;
    border-color: #00e5ff !important;
    color: #00e5ff !important;
  }
  div[data-testid="stSidebar"] .stRadio > div > label p {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    opacity: 1 !important;
  }
  div[data-testid="stSidebar"] .stRadio > div > label span {
    color: #ffffff !important;
    font-weight: 700 !important;
    opacity: 1 !important;
  }
  div[data-testid="stSidebar"] .stRadio > div > label div {
    color: #ffffff !important;
    opacity: 1 !important;
  }
  div[data-testid="stSidebar"] .stRadio input {
    display: none !important;
  }

  [data-testid="stMetric"] {
    background: #1a2e45;
    border: 1px solid #2a4a6b;
    border-radius: 10px;
    padding: 16px;
  }
  [data-testid="stMetricValue"] { color: #00e5ff !important; }
  [data-testid="stMetricLabel"] { color: #a0c8e8 !important; }

  .stButton > button {
    background: #00e5ff;
    color: #000;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-family: 'Manrope', sans-serif;
  }
  .stButton > button:hover { opacity: 0.85; }

  .fraud-alert {
    background: rgba(255,107,43,0.25);
    border-left: 5px solid #ff6b2b;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
  }
  .safe-alert {
    background: rgba(0,230,118,0.25);
    border-left: 5px solid #00e676;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
  }
  .info-alert {
    background: rgba(0,229,255,0.25);
    border-left: 5px solid #00e5ff;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #000000;
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
  }

  @media (max-width: 768px) {
    section[data-testid="stSidebar"] {
      width: 100vw !important;
    }
  }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 12px;
                border-bottom:2px solid #00e5ff; margin-bottom:16px;">
      <div style="font-family:'Manrope',sans-serif; font-size:2rem;
                  font-weight:800; color:#00e5ff; letter-spacing:3px;">
        FraudIQ
      </div>
      <div style="font-size:0.68rem; color:#a0c8e8; letter-spacing:2px;
                  margin-top:4px; font-family:'Manrope',sans-serif;
                  font-weight:600;">
        FUNERAL INSURANCE INTELLIGENCE
      </div>
    </div>
    <div style="font-size:0.76rem; color:#00e5ff;
                font-family:'Manrope',sans-serif; font-weight:800;
                margin-bottom:8px; padding:0 4px; letter-spacing:1px;">
      NAVIGATION
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Live Claim Scorer",
            "Model Training",
            "Model Benchmarks",
            "EDA and Risk Signals",
            "Batch Screening",
            "Actuarial Impact",
            "SHAP Explainer",
            "Settings",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin-top:20px; padding:14px; background:#1a2e45;
                border-radius:8px; border:1px solid #00e5ff;">
      <div style="color:#00e5ff; font-size:0.82rem; font-weight:800;
                  font-family:'Manrope',sans-serif; margin-bottom:10px;
                  letter-spacing:1px;">
        RESEARCH CONTEXT
      </div>
      <div style="font-size:0.82rem; font-family:'Manrope',sans-serif;
                  line-height:2.0; color:#ffffff;">
        <span style="color:#a0c8e8;">Author: </span>
        <span style="font-weight:700;">T.A. Mupindu</span><br>
        <span style="color:#a0c8e8;">ID: </span>
        <span style="font-weight:700;">2022996712</span><br>
        <span style="color:#a0c8e8;">Degree: </span>
        <span style="font-weight:700;">BSc Actuarial Science</span><br>
        <span style="color:#a0c8e8;">Dept: </span>
        <span style="font-weight:700;">MSAS, UNZA 2026</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

if page == "Dashboard":
    from pages.dashboard import render
elif page == "Live Claim Scorer":
    from pages.scorer import render
elif page == "Model Training":
    from pages.model_training import render
elif page == "Model Benchmarks":
    from pages.benchmarks import render
elif page == "EDA and Risk Signals":
    from pages.eda import render
elif page == "Batch Screening":
    from pages.batch import render
elif page == "Actuarial Impact":
    from pages.actuarial import render
elif page == "SHAP Explainer":
    from pages.shap_page import render
elif page == "Settings":
    from pages.settings import render
else:
    from pages.dashboard import render

render()