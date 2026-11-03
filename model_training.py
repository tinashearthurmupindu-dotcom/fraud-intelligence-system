"""pages/model_training.py — Model Training"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data_generator import generate_claims_dataset
from models.fraud_models import train_all_models, results_to_dataframe, FEATURE_COLS


@st.cache_data(show_spinner=False)
def get_data(n):
    return generate_claims_dataset(n)


@st.cache_resource(show_spinner=False)
def run_training(n):
    return train_all_models(get_data(n))


def render():
    st.markdown("## Model Training")
    st.caption(
        "Stratified 70/15/15 split. SMOTE is applied to the training set only. "
        "Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM."
    )

    n = st.slider("Number of synthetic claims", 1000, 5000, 2000, step=500)
    df = get_data(n)

    c1, c2, c3 = st.columns(3)
    c1.metric("Claims", f"{len(df):,}")
    c2.metric("Fraud rate", f"{df['is_fraud'].mean():.1%}")
    c3.metric("Features", len(FEATURE_COLS))

    if "trained_n" not in st.session_state:
        st.session_state["trained_n"] = None

    if st.button("Train all models", type="primary"):
        with st.spinner("Training models. This can take a minute..."):
            run_training(n)
        st.session_state["trained_n"] = n

    if st.session_state["trained_n"] != n:
        st.info("Click **Train all models** to start.")
        return

    results = run_training(n)
    table = results_to_dataframe(results)
    st.success("Training complete.")

    st.markdown("### Results (held-out test set)")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("### ROC curves")
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(x=r.fpr, y=r.tpr, mode="lines",
                                 name=f"{name} (AUC {r.auc_roc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(dash="dash", color="#556677")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8899aa"), height=420,
        xaxis=dict(title="False positive rate", gridcolor="#1a2235"),
        yaxis=dict(title="True positive rate", gridcolor="#1a2235"),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature importance")
    choices = [k for k, r in results.items() if r.feature_importance]
    if choices:
        pick = st.selectbox("Model", choices)
        imp = pd.Series(results[pick].feature_importance).abs().sort_values()
        bar = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker_color="#00e5ff"))
        bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8899aa"), height=420,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(bar, use_container_width=True)
