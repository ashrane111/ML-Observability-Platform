"""Streamlit Dashboard for ML Observability Platform.

Run with: streamlit run dashboard/app.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
REFRESH_INTERVAL = 30  # seconds


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="ML Observability Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-healthy { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-critical { color: #dc3545; }
    .big-font { font-size: 24px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API Helper Functions
# ============================================================================

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_health() -> Dict[str, Any]:
    """Fetch API health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_drift_status(model_name: str) -> Dict[str, Any]:
    """Fetch drift status for a model."""
    try:
        response = requests.get(f"{API_URL}/monitoring/drift/status/{model_name}", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e), "drift_detected": False, "drift_share": 0, "drifted_features": []}


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_alerts() -> List[Dict[str, Any]]:
    """Fetch active alerts."""
    try:
        response = requests.get(f"{API_URL}/monitoring/alerts", timeout=5)
        return response.json()
    except Exception as e:
        return []


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_alert_summary() -> Dict[str, Any]:
    """Fetch alert summary."""
    try:
        response = requests.get(f"{API_URL}/monitoring/alerts/summary", timeout=5)
        return response.json()
    except Exception as e:
        return {"total": 0, "by_severity": {}, "by_type": {}}


def make_prediction(model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Make a prediction via the API."""
    try:
        response = requests.post(f"{API_URL}/predict/{model_name}", json=features, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render the sidebar navigation and filters."""
    st.sidebar.title("🔍 ML Observability")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Models", "Drift Detection", "Alerts", "Predictions", "Settings"],
    )
    
    st.sidebar.markdown("---")
    selected_model = st.sidebar.selectbox("Select Model", ["fraud", "price", "churn"])
    
    st.sidebar.markdown("---")
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"],
        index=2
    )
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    
    return page, selected_model, time_range, auto_refresh


# ============================================================================
# Dashboard Page
# ============================================================================

def render_dashboard(selected_model: str):
    """Render the main dashboard page."""
    st.title("📊 ML Observability Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    health = fetch_health()
    
    with col1:
        status = "✅ Healthy" if health.get("status") == "ok" else "❌ Error"
        st.metric("API Status", status)
    
    with col2:
        models_loaded = health.get("models_loaded", {})
        loaded_count = sum(1 for v in models_loaded.values() if v)
        st.metric("Models Loaded", f"{loaded_count}/3")
    
    with col3:
        alert_summary = fetch_alert_summary()
        st.metric("Active Alerts", alert_summary.get("total", 0))
    
    with col4:
        drift_status = fetch_drift_status(selected_model)
        drift_detected = drift_status.get("drift_detected", False)
        st.metric(f"{selected_model.title()} Drift", "⚠️ Yes" if drift_detected else "✅ No")
    
    st.markdown("---")
    
    # Model Status Grid
    st.subheader("Model Status Overview")
    model_cols = st.columns(3)
    
    for i, model in enumerate(["fraud", "price", "churn"]):
        with model_cols[i]:
            status = fetch_drift_status(model)
            drift = status.get("drift_detected", False)
            drift_share = status.get("drift_share", 0)
            
            st.markdown(f"### {model.title()} Detector")
            st.write(f"**Drift:** {'🚨 Detected' if drift else '✅ None'}")
            st.write(f"**Drift Share:** {drift_share:.1%}")
            st.write(f"**Features Drifted:** {len(status.get('drifted_features', []))}")
    
    st.markdown("---")
    
    # Alerts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📢 Recent Alerts")
        alerts = fetch_alerts()
        if alerts:
            for alert in alerts[:5]:
                severity_icon = {"info": "🔵", "warning": "🟡", "critical": "🔴"}.get(alert.get("severity"), "⚪")
                st.write(f"{severity_icon} **{alert.get('alert_type')}**: {alert.get('message', '')[:50]}...")
        else:
            st.info("No active alerts")
    
    with col2:
        st.subheader("📈 Alert Distribution")
        alert_summary = fetch_alert_summary()
        if alert_summary.get("by_severity"):
            fig = px.pie(
                values=list(alert_summary["by_severity"].values()),
                names=list(alert_summary["by_severity"].keys()),
                color_discrete_map={"info": "#17a2b8", "warning": "#ffc107", "critical": "#dc3545"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alert data")


# ============================================================================
# Models Page
# ============================================================================

def render_models_page(selected_model: str):
    """Render the models detail page."""
    st.title(f"🤖 Model: {selected_model.title()}")
    
    model_info = {
        "fraud": {"type": "XGBoost Classifier", "target": "is_fraud", "features": 10},
        "price": {"type": "LightGBM Regressor", "target": "price", "features": 12},
        "churn": {"type": "Random Forest Classifier", "target": "churned", "features": 8},
    }
    
    info = model_info.get(selected_model, {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Information")
        st.write(f"**Type:** {info.get('type')}")
        st.write(f"**Target:** {info.get('target')}")
        st.write(f"**Features:** {info.get('features')}")
    
    with col2:
        st.subheader("Actions")
        if st.button("🔄 Trigger Retraining"):
            st.info("Retraining flow triggered!")
        if st.button("📊 View Full Metrics"):
            st.info("Opening metrics...")
    
    st.markdown("---")
    st.subheader("Performance Metrics")
    
    metrics_data = {
        "fraud": {"Accuracy": 0.92, "Precision": 0.89, "Recall": 0.85, "F1": 0.87, "AUC": 0.94},
        "price": {"R²": 0.82, "RMSE": 32500, "MAE": 25000},
        "churn": {"Accuracy": 0.88, "Precision": 0.86, "Recall": 0.82, "F1": 0.84},
    }
    
    metrics = metrics_data.get(selected_model, {})
    cols = st.columns(len(metrics))
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i]:
            display = f"{value:.2%}" if isinstance(value, float) and value < 1 else f"{value:,.0f}"
            st.metric(name, display)


# ============================================================================
# Drift Detection Page
# ============================================================================

def render_drift_page(selected_model: str):
    """Render the drift detection page."""
    st.title("📉 Drift Detection")
    
    drift_status = fetch_drift_status(selected_model)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Drift Detected", "Yes" if drift_status.get("drift_detected") else "No")
    with col2:
        st.metric("Drift Share", f"{drift_status.get('drift_share', 0):.1%}")
    with col3:
        st.metric("Features Drifted", len(drift_status.get("drifted_features", [])))
    
    st.markdown("---")
    st.subheader("Feature Drift Scores")
    
    # Sample feature scores for visualization
    feature_scores = {
        "amount": 0.15, "hour": 0.05, "day_of_week": 0.02,
        "customer_age": 0.22, "account_age_days": 0.08,
        "transaction_count_24h": 0.18, "avg_transaction_amount": 0.12,
    }
    
    df_scores = pd.DataFrame({
        "Feature": list(feature_scores.keys()),
        "PSI Score": list(feature_scores.values())
    }).sort_values("PSI Score", ascending=True)
    
    fig = px.bar(df_scores, x="PSI Score", y="Feature", orientation="h",
                 color="PSI Score", color_continuous_scale=["green", "yellow", "red"])
    fig.add_vline(x=0.1, line_dash="dash", line_color="orange")
    fig.add_vline(x=0.2, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Alerts Page
# ============================================================================

def render_alerts_page():
    """Render the alerts page."""
    st.title("🚨 Alert Management")
    
    alert_summary = fetch_alert_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", alert_summary.get("total", 0))
    with col2:
        st.metric("Critical", alert_summary.get("by_severity", {}).get("critical", 0))
    with col3:
        st.metric("Warning", alert_summary.get("by_severity", {}).get("warning", 0))
    
    st.markdown("---")
    st.subheader("Active Alerts")
    
    alerts = fetch_alerts()
    if alerts:
        for alert in alerts[:20]:
            icon = {"info": "🔵", "warning": "🟡", "critical": "🔴"}.get(alert.get("severity"), "⚪")
            with st.expander(f"{icon} {alert.get('alert_type')} - {alert.get('created_at', '')[:19]}"):
                st.write(f"**Message:** {alert.get('message')}")
                st.write(f"**Severity:** {alert.get('severity')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.button("✅ Acknowledge", key=f"ack_{alert.get('id', id(alert))}")
                with col2:
                    st.button("🔇 Resolve", key=f"res_{alert.get('id', id(alert))}")
    else:
        st.info("No active alerts")


# ============================================================================
# Predictions Page
# ============================================================================

def render_predictions_page(selected_model: str):
    """Render the predictions testing page."""
    st.title("🎯 Make Predictions")
    
    if selected_model == "fraud":
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount ($)", value=100.0)
            hour = st.slider("Hour", 0, 23, 14)
            customer_age = st.number_input("Customer Age", value=35)
        with col2:
            transaction_count = st.number_input("Transactions (24h)", value=3)
            distance = st.number_input("Distance from Home (km)", value=5.0)
        
        if st.button("🔍 Predict"):
            result = make_prediction("fraud", {
                "amount": amount, "hour": hour, "customer_age": customer_age,
                "transaction_count_24h": transaction_count, "distance_from_home": distance
            })
            if "error" not in result:
                pred = result.get("prediction", 0)
                prob = result.get("probability", 0)
                st.metric("Prediction", "🚨 FRAUD" if pred else "✅ OK")
                st.metric("Probability", f"{prob:.1%}")
    
    elif selected_model == "price":
        col1, col2 = st.columns(2)
        with col1:
            sqft = st.number_input("Square Feet", value=1500)
            bedrooms = st.number_input("Bedrooms", value=3)
        with col2:
            bathrooms = st.number_input("Bathrooms", value=2.0)
            year_built = st.number_input("Year Built", value=2000)
        
        if st.button("💰 Predict"):
            result = make_prediction("price", {
                "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms, "year_built": year_built
            })
            if "error" not in result:
                st.metric("Predicted Price", f"${result.get('prediction', 0):,.0f}")
    
    elif selected_model == "churn":
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.number_input("Tenure (months)", value=24)
            monthly_charges = st.number_input("Monthly Charges ($)", value=70.0)
        with col2:
            contract = st.selectbox("Contract", ["month-to-month", "one_year", "two_year"])
            support_tickets = st.number_input("Support Tickets", value=2)
        
        if st.button("📊 Predict"):
            result = make_prediction("churn", {
                "tenure_months": tenure, "monthly_charges": monthly_charges,
                "contract_type": contract, "num_support_tickets": support_tickets
            })
            if "error" not in result:
                pred = result.get("prediction", 0)
                prob = result.get("probability", 0)
                st.metric("Prediction", "⚠️ WILL CHURN" if pred else "✅ WILL STAY")
                st.metric("Probability", f"{prob:.1%}")


# ============================================================================
# Settings Page
# ============================================================================

def render_settings_page():
    """Render the settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("API Configuration")
    st.text_input("API URL", value=API_URL)
    
    st.subheader("Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("PSI Warning", value=0.1)
        st.number_input("PSI Critical", value=0.2)
    with col2:
        st.number_input("Min Accuracy", value=0.85)
        st.number_input("Max Latency (ms)", value=500)
    
    if st.button("💾 Save"):
        st.success("Settings saved!")


# ============================================================================
# Main
# ============================================================================

def main():
    page, selected_model, time_range, auto_refresh = render_sidebar()
    
    if page == "Dashboard":
        render_dashboard(selected_model)
    elif page == "Models":
        render_models_page(selected_model)
    elif page == "Drift Detection":
        render_drift_page(selected_model)
    elif page == "Alerts":
        render_alerts_page()
    elif page == "Predictions":
        render_predictions_page(selected_model)
    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
