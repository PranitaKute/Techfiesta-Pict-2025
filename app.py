import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Adaptive Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.4);
}

.metric-title {
    font-size: 14px;
    color: #9CA3AF;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
}

.low { color: #22c55e; }
.medium { color: #f59e0b; }
.high { color: #ef4444; }

.alert-high {
    background-color: #7f1d1d;
    padding: 15px;
    border-radius: 12px;
    font-weight: bold;
    color: white;
}

.explain-box {
    background-color: #111827;
    padding: 15px;
    border-left: 5px solid #6366f1;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    rf_model = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    autoencoder = load_model("models/autoencoder.h5", compile=False)
    return rf_model, scaler, autoencoder

rf_model, scaler, autoencoder = load_models()

# ===============================
# HEADER
# ===============================
st.title("🛡️ Adaptive AI-Based Fraud Detection & Risk Management System")
st.caption("Hybrid AI • Behaviour Analysis • Explainable Decisions")

# ===============================
# TABS
# ===============================
tab1, tab2, tab4 = st.tabs([
    "📌 Problem Context",
    "🧾 Transaction Simulation",
    "🧠 Explainability & Action"
])

# ===============================
# TAB 1 — PROBLEM CONTEXT
# ===============================
with tab1:
    st.subheader("🚨 Real-World Problem")

    st.markdown("""
    💳 **Banks lose billions every year due to fraud**  
    ❌ Traditional systems react *after* money is lost  
    ❌ Rule-based logic fails on new fraud patterns  

    ✅ **Our Solution**
    - Real-time fraud risk prediction
    - Behaviour-based anomaly detection
    - Explainable AI decisions
    - Risk-based action engine
    """)

    st.success("Designed for scalable, real-time banking systems")

# ===============================
# TAB 2 — TRANSACTION SIMULATION
# ===============================
with tab2:
    st.subheader("🧾 Simulate a Transaction")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount ($)", 1.0, 5000.0, 250.0)
        balance = st.number_input("Account Balance ($)", 0.0, 200000.0, 50000.0)
        distance = st.slider("Transaction Distance (km)", 1, 5000, 500)
        hour = st.slider("Transaction Hour", 0, 23, 14)
        daily_count = st.slider("Daily Transaction Count", 1, 50, 5)

    with col2:
        txn_type = st.selectbox("Transaction Type", ["POS", "Online", "ATM Withdrawal", "Bank Transfer"])
        device = st.selectbox("Device Type", ["Mobile", "Laptop", "Tablet"])
        location = st.selectbox("Location", ["Mumbai", "New York", "Sydney", "London"])
        merchant = st.selectbox("Merchant Category", ["Electronics", "Clothing", "Travel", "Restaurants"])
        auth = st.selectbox("Authentication Method", ["OTP", "Password", "Biometric"])

    simulate = st.button("🚀 Analyze Transaction")
    result_container = st.container()

# ===============================
# PROCESS TRANSACTION
# ===============================
if simulate:
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    is_weekend = 0

    input_df = pd.DataFrame([{
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
        "Is_Weekend": is_weekend,
        "Transaction_Hour": hour,
        "Transaction_Day": 15,
        "Is_Night_Transaction": is_night
    }])

    ae_features = [
        'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count',
        'Avg_Transaction_Amount_7d', 'Transaction_Distance',
        'Card_Age', 'Is_Weekend', 'Transaction_Hour', 'Is_Night_Transaction'
    ]

    scaled = scaler.transform(input_df[ae_features])
    recon = autoencoder.predict(scaled)
    anomaly_score = np.mean(np.square(scaled - recon))
    anomaly_flag = int(anomaly_score > 1.0)

    input_df["Anomaly_Score"] = anomaly_score
    input_df["Anomaly_Flag"] = anomaly_flag

    fraud_prob = rf_model.predict_proba(input_df)[0][1]

    if fraud_prob > 0.7 or anomaly_flag:
        risk = "High"
    elif fraud_prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # ===============================
    # AI RISK ENGINE (INLINE)
    # ===============================
    with result_container:
        st.subheader("🤖 AI Risk Decision")

        if risk == "High":
            st.markdown('<div class="alert-high">🚨 HIGH RISK TRANSACTION DETECTED</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fraud Probability</div>
            <div class="metric-value">{fraud_prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Anomaly Score</div>
            <div class="metric-value">{anomaly_score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value {risk.lower()}">{risk}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(fraud_prob * 100))

        # Risk Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 40], "color": "green"},
                    {"range": [40, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"},
                ],
            },
            title={"text": "Risk Score (%)"}
        ))
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# TAB 4 — EXPLAINABILITY
# ===============================
with tab4:
    st.subheader("🧠 Explainability & Recommended Action")

    st.markdown("""
    <div class="explain-box">
    <b>Why this transaction is risky?</b><br>
    • Unusual transaction amount or frequency<br>
    • Location / device deviation<br>
    • Behaviour anomaly detected by Autoencoder
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    <b>AI Decision Strategy</b><br>
    • Supervised Random Forest for known fraud patterns<br>
    • Autoencoder for unknown / zero-day fraud
    </div>
    """, unsafe_allow_html=True)

    st.success("🔔 Action Engine: BLOCK / OTP / ALLOW based on risk score")
