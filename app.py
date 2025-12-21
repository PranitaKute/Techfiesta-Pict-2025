# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime
import time

# PAGE CONFIG
st.set_page_config(
    page_title="Adaptive Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

# CONFIG (Production-like)
FRAUD_HIGH_THRESHOLD = 0.70
FRAUD_MEDIUM_THRESHOLD = 0.40
ANOMALY_THRESHOLD = 1.0

# SESSION STATE
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "txn_id" not in st.session_state:
    st.session_state.txn_id = 0

# CUSTOM CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.4);
}
.high { color: #ef4444; font-weight: bold; }
.medium { color: #f59e0b; font-weight: bold; }
.low { color: #22c55e; font-weight: bold; }
.signal-box {
    background-color: #111827;
    padding: 14px;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    fraud_classifier = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    behaviour_model = load_model("models/autoencoder.h5", compile=False)
    return fraud_classifier, scaler, behaviour_model

fraud_classifier, scaler, behaviour_model = load_models()

# HEADER
st.title("🛡️ Adaptive AI-Based Fraud Detection Platform")
st.caption("Real-time Behaviour Analysis • Risk Scoring • Automated Decisions")

# SIDEBAR — SYSTEM HEALTH
st.sidebar.subheader("📈 System Health")
st.sidebar.metric("Transactions / min", np.random.randint(80, 160))
st.sidebar.metric("Model Latency (ms)", np.random.randint(25, 70))
st.sidebar.metric("Anomaly Rate", f"{np.random.uniform(1.2, 4.8):.2f}%")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Overview",
    "💳 Transaction Analysis",
    "📊 Alert Timeline",
    "🌐 Streaming Architecture"
])

# TAB 1 — OVERVIEW
with tab1:
    st.markdown("""
**Problem**  
Traditional fraud systems are rule-based and fail against new attack patterns.

**Solution**  
This platform combines:
- ML-based fraud classification
- Behaviour anomaly detection
- Risk-based decision engine

**Outcome**  
✔ Real-time detection  
✔ Adaptive intelligence  
✔ Automated actioning
    """)

# TAB 2 — TRANSACTION ANALYSIS
with tab2:
    mode = st.radio("Transaction Mode", ["Manual Entry", "Live Simulation"], horizontal=True)

    col1, col2 = st.columns(2)

    with col1:
        if mode == "Live Simulation":
            amount = np.random.uniform(100, 900)
            balance = np.random.uniform(10000, 120000)
            distance = np.random.randint(20, 400)
            hour = datetime.now().hour
            daily_count = np.random.randint(2, 8)
        else:
            amount = st.number_input("Transaction Amount ($)", 1.0, 5000.0, 250.0)
            balance = st.number_input("Account Balance ($)", 0.0, 200000.0, 50000.0)
            distance = st.slider("Transaction Distance (km)", 1, 5000, 500)
            hour = st.slider("Transaction Hour", 0, 23, 14)
            daily_count = st.slider("Daily Transaction Count", 1, 50, 5)

    with col2:
        txn_type = st.selectbox("Transaction Type", ["POS", "Online", "ATM Withdrawal", "Bank Transfer"])
        device = st.selectbox("Device Type", ["Mobile", "Laptop", "Tablet"])
        country = st.selectbox("Country", ["India", "USA", "UK", "Australia"])
        location_map = {
            "India": ["Mumbai", "Pune", "Delhi"],
            "USA": ["New York", "Chicago", "San Francisco"],
            "UK": ["London", "Manchester"],
            "Australia": ["Sydney", "Melbourne"]
        }
        location = st.selectbox("City", location_map[country])
        merchant = st.selectbox("Merchant Category", ["Electronics", "Clothing", "Travel", "Restaurants"])
        auth = st.selectbox("Authentication Method", ["OTP", "Password", "Biometric"])

    analyze = st.button(" Analyze Transaction")

# PROCESS TRANSACTION
if analyze:
    with st.spinner("Analyzing transaction..."):
        time.sleep(1)

    is_night = 1 if hour >= 22 or hour <= 5 else 0

    transaction = pd.DataFrame([{
        "Transaction_Amount": amount,
        "Transaction_Type": txn_type,
        "Account_Balance": balance,
        "Device_Type": device,
        "Location": location,
        "Merchant_Category": merchant,
        "Daily_Transaction_Count": daily_count,
        "Avg_Transaction_Amount_7d": amount,
        "Failed_Transaction_Count_7d": 1,
        "Card_Type": "Visa",
        "Card_Age": 120,
        "Transaction_Distance": distance,
        "Authentication_Method": auth,
        "Is_Weekend": 0,
        "Transaction_Hour": hour,
        "Transaction_Day": 15,
        "Is_Night_Transaction": is_night
    }])

    behaviour_features = [
        "Transaction_Amount", "Account_Balance", "Daily_Transaction_Count",
        "Avg_Transaction_Amount_7d", "Transaction_Distance",
        "Card_Age", "Is_Weekend", "Transaction_Hour", "Is_Night_Transaction"
    ]

    scaled_features = scaler.transform(transaction[behaviour_features])
    reconstructed = behaviour_model.predict(scaled_features)
    anomaly_score = float(np.mean(np.square(scaled_features - reconstructed)))
    anomaly_flag = anomaly_score > ANOMALY_THRESHOLD

    transaction["Anomaly_Score"] = anomaly_score
    transaction["Anomaly_Flag"] = int(anomaly_flag)


    fraud_probability = fraud_classifier.predict_proba(transaction)[0][1]

    # SOFT RISK LOGIC (KEY FIX)
    soft_risk_score = 0.0

    if 400 <= amount <= 800:
        soft_risk_score += 0.15

    if 5 <= daily_count <= 7:
        soft_risk_score += 0.10

    if 100 <= distance <= 300:
        soft_risk_score += 0.10

    effective_risk = fraud_probability + soft_risk_score

    # FINAL DECISION ENGINE
    if effective_risk > FRAUD_HIGH_THRESHOLD or anomaly_flag:
        risk, action = "High", "BLOCK"
    elif effective_risk > FRAUD_MEDIUM_THRESHOLD:
        risk, action = "Medium", "OTP"
    else:
        risk, action = "Low", "ALLOW"

    st.session_state.txn_id += 1
    st.session_state.alert_history.append({
        "Txn ID": st.session_state.txn_id,
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Risk": risk,
        "Action": action,
        "Fraud Probability": f"{fraud_probability:.2%}"
    })

    st.subheader("🤖 Risk Decision")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{fraud_probability:.2%}")
    c2.metric("Behaviour Deviation", f"{anomaly_score:.3f}")
    c3.metric("Final Risk Level", risk)

    # Align gauge with final decision
    gauge_value = effective_risk * 100

    if anomaly_flag:
        gauge_value = max(gauge_value, 85)  # force visual high-risk

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={"text": "Decision Risk Score (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "white"},
            "steps": [
                {"range": [0, 40], "color": "#22c55e"},   # Green → Low
                {"range": [40, 70], "color": "#facc15"}, # Yellow → Medium
                {"range": [70, 100], "color": "#ef4444"} # Red → High
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": gauge_value
            }
        }
    ))


    st.plotly_chart(fig, use_container_width=True)


    st.subheader("⚠️ Risk Signals")
    if anomaly_flag:
        st.markdown('<div class="signal-box">Unusual transaction behaviour detected</div>', unsafe_allow_html=True)
    if distance > 1000:
        st.markdown('<div class="signal-box">Transaction from distant location</div>', unsafe_allow_html=True)
    if is_night:
        st.markdown('<div class="signal-box">Night-time transaction</div>', unsafe_allow_html=True)
    if fraud_probability > 0.6:
        st.markdown('<div class="signal-box">Matches known fraud patterns</div>', unsafe_allow_html=True)

# TAB 3 — ALERT TIMELINE
with tab3:
    st.subheader("📊 Alert History")
    if st.session_state.alert_history:
        df = pd.DataFrame(st.session_state.alert_history)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📤 Export Alerts",
            df.to_csv(index=False),
            "fraud_alerts.csv",
            "text/csv"
        )
    else:
        st.info("No alerts generated yet")

# TAB 4 — STREAMING ARCHITECTURE
with tab4:
    st.markdown("""
**Production Streaming Design**

Transaction Stream → Kafka  
Kafka → Fraud Detection Microservice  
ML Engine → Risk Decision Engine  
Actions → Block / OTP / Allow  
Alerts → Dashboard + Monitoring  

✔ Scalable  
✔ Cloud-ready  
✔ Real-time
    """)
