import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# =========================
# Create models folder
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# Load Dataset
# =========================
df = pd.read_csv("synthetic_fraud_dataset.csv")

# =========================
# Feature Engineering
# =========================
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Transaction_Hour'] = df['Timestamp'].dt.hour
df['Transaction_Day'] = df['Timestamp'].dt.day
df['Is_Night_Transaction'] = df['Transaction_Hour'].apply(
    lambda x: 1 if x >= 22 or x <= 5 else 0
)

df.drop(['Transaction_ID', 'User_ID', 'Timestamp'], axis=1, inplace=True)

# =========================
# Remove Leakage Features
# =========================
leakage_features = [
    'Previous_Fraudulent_Activity',
    'Risk_Score',
    'IP_Address_Flag'
]
df.drop(leakage_features, axis=1, inplace=True)

# =========================
# Autoencoder Training
# =========================
auto_features = [
    'Transaction_Amount',
    'Account_Balance',
    'Daily_Transaction_Count',
    'Avg_Transaction_Amount_7d',
    'Transaction_Distance',
    'Card_Age',
    'Is_Weekend',
    'Transaction_Hour',
    'Is_Night_Transaction'
]

df_normal = df[df['Fraud_Label'] == 0][auto_features]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_normal)

joblib.dump(scaler, "models/scaler.pkl")

input_dim = df_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

autoencoder.fit(
    df_scaled,
    df_scaled,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    shuffle=True
)

autoencoder.save("models/autoencoder.h5")

# =========================
# Generate Anomaly Scores
# =========================
df_auto_all = scaler.transform(df[auto_features])
recon = autoencoder.predict(df_auto_all)

df['Anomaly_Score'] = np.mean(np.square(df_auto_all - recon), axis=1)
threshold = np.percentile(df['Anomaly_Score'], 95)
df['Anomaly_Flag'] = (df['Anomaly_Score'] > threshold).astype(int)

# =========================
# Random Forest Training
# =========================
X = df.drop('Fraud_Label', axis=1)
y = df['Fraud_Label']

categorical_cols = [
    'Transaction_Type',
    'Device_Type',
    'Location',
    'Merchant_Category',
    'Card_Type',
    'Authentication_Method'
]

numerical_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('rf', rf)
])

pipeline.fit(X, y)

joblib.dump(pipeline, "models/rf_model.pkl")

print("✅ Training completed. Models saved in /models folder.")
