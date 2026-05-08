import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from app.utils.model_loader import (
    load_model,
    load_metadata,
    load_validation_results,
    load_shap_importance,
    load_train_data,
    load_test_data,
)
from app.tabs import user_tab, developer_tab

st.set_page_config(
    page_title="Procurement Lead Time Variance Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_metadata():
    return load_metadata()

@st.cache_resource
def get_validation_results():
    return load_validation_results()

@st.cache_resource
def get_shap_importance():
    return load_shap_importance()

@st.cache_resource
def get_train_data():
    return load_train_data()

@st.cache_resource
def get_test_data():
    return load_test_data()

model              = get_model()
feature_meta       = get_metadata()
validation_results = get_validation_results()
shap_importance    = get_shap_importance()
train_df           = get_train_data()
test_df            = get_test_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📦 Procurement Lead Time Variance Predictor")
st.divider()

# ── Tab routing ──────────────────────────────────────────────────────────────
tab_user, tab_dev = st.tabs(["👤 User — Order Predictor", "🛠️ Developer — Monitoring"])

with tab_user:
    user_tab.render(model, feature_meta, validation_results)

with tab_dev:
    developer_tab.render(
        model, feature_meta, validation_results,
        shap_importance, train_df, test_df
    )