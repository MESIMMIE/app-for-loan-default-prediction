# app.py
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("🚀 Gen Z Loan Default Predictor 💰")

# Load models
scaler = joblib.load("scaler.pkl")
dnn_model = keras.models.load_model("loan_default_model.h5")
xgb_model = joblib.load("xgb_model.pkl")

# User input fields
income = st.number_input("💵 Annual Income ($)", min_value=0, value=50000)
credit = st.number_input("🏦 Credit Amount ($)", min_value=0, value=100000)
birth_years = st.number_input("🎂 Age (Years)", min_value=18, value=30)
employment_years = st.number_input("💼 Employment Duration (Years)", min_value=0, value=5)

# Prediction
if st.button("🚀 Predict Loan Default 💳"):
    try:
        input_data = np.array([[income, credit, birth_years, employment_years]])
        input_scaled = scaler.transform(input_data)

        dnn_pred = dnn_model.predict(input_scaled)[0, 0]
        xgb_pred = xgb_model.predict_proba(input_scaled)[:, 1][0]

        st.subheader(f"🧠 DNN Prediction: {'❌ Default' if dnn_pred > 0.5 else '✅ No Default'} ({dnn_pred:.2f})")
        st.subheader(f"📊 XGBoost Prediction: {'❌ Default' if xgb_pred > 0.5 else '✅ No Default'} ({xgb_pred:.2f})")

    except Exception as e:
        st.error(f"Prediction Failed: {e}")
