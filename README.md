# AI-Based Fraud Detection System

## Problem Statement

Financial fraud is a major challenge for banks and digital payment systems. Every year, financial institutions lose huge amounts of money due to fraudulent transactions.

Most traditional fraud detection systems are rule-based, which means:

- Fraud is often detected after the loss  
- Many genuine transactions are incorrectly blocked (high false positives)  
- Decisions are hard to explain to users or auditors  

---

## Our Approach

To solve this problem, we designed an adaptive AI-based fraud detection system that works in real time and focuses on risk assessment instead of yes/no decisions.

Our system:

- Generates a risk score for every transaction  
- Uses both machine learning and anomaly detection  
- Provides clear explanations for every decision  
- Helps in taking smarter and faster actions on transactions  

---

## Key Features

- Risk-Based Fraud Scoring instead of binary labels  
- Hybrid AI Model  
  - Random Forest for known fraud patterns  
  - Autoencoder for detecting unusual or unknown behavior  
- Explainable AI Outputs for transparency  
- Real-Time Transaction Simulation  
- Scalable and Cloud-Ready Design  

---

## System Architecture
```text
User Transaction
        ↓
Feature Engineering Layer
        ↓
Autoencoder (Anomaly Detection)
        +
Random Forest (Fraud Classification)
        ↓
Adaptive Risk Engine
        ↓
Explainability & Action Layer
```

---

## Dataset Strategy

- Transaction data is designed based on real-world banking behavior  
- Synthetic data is generated to handle rare and edge fraud cases  
- Data leakage features are removed to ensure fair evaluation  
- In real deployment, the model can be trained using actual bank transaction logs  

---

## Tech Stack Used

- Frontend: Streamlit  
- Machine Learning Models: Random Forest, Autoencoder using TensorFlow  
- Backend Logic: Python  
- Data Processing: Pandas, NumPy  
- Model Storage: Joblib, Keras  
- Deployment: Cloud-compatible (AWS / Google Cloud)  

---

## Installation & How to Run the Project

1️. Clone the repository
```bash
git clone https://github.com/PranitaKute/Techfiesta-Pict-2025.git
cd Fraud_Detection_V2
```

2️. Create virtual environment (Python 3.10 recommended)
```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

3️. Install required libraries
```bash
pip install -r requirements.txt
```

4️. Run the application
```bash
streamlit run app.py
```

Explainability

- To ensure trust and transparency, the system explains every fraud decision using:

- Behaviour-based anomaly scores

- Model-generated risk probabilities

- Simple, human-readable reasons

- Recommended actions such as Allow / OTP / Block

Ethics & Security Considerations

- No personal or sensitive user data is exposed

- All risk decisions are explainable

- Designed to minimize bias and reduce false positives

- Secure and controlled model execution

Future Enhancements

- Real-time transaction streaming using Kafka

- Microservices using FastAPI

- Full cloud deployment

- Automated model retraining pipelines

- Role-based dashboards for banks and admins
