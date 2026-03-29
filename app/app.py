import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# =========================
# Load model & scaler safely
# =========================
BASE_DIR = os.path.dirname(__file__)

scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "fraud_model.pkl"), "rb"))

# =========================
# UI
# =========================
st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is fraudulent or legitimate")

st.markdown("---")

# =========================
# Manual Prediction
# =========================
st.subheader("🔹 Manual Prediction")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0)
    time = st.number_input("Transaction Time", min_value=0.0)

with col2:
    st.info("V1–V28 are hidden (default = 0)")

# 🔥 Threshold slider (NEW)
threshold = st.slider("Fraud Detection Sensitivity", 0.0, 1.0, 0.3)

if st.button("Predict"):
    try:
        input_data = np.array([time] + [0]*28 + [amount]).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # 👉 Use probability
        proba = model.predict_proba(input_data)[0][1]

        if proba > threshold:
            st.error(f"🚨 Fraud Detected! (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {proba:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# =========================
# Bulk Prediction (CSV)
# =========================
st.subheader("📂 Bulk Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📊 Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Run Bulk Prediction", key="bulk_predict"):
        try:
            # Convert to numpy
            X = scaler.transform(df.values)

            # 👉 Use probability instead of direct prediction
            proba = model.predict_proba(X)

            preds = (proba[:, 1] > threshold).astype(int)

            result_df = df.copy()
            result_df["Fraud_Probability"] = proba[:, 1]

            result_df["Prediction"] = preds
            result_df["Prediction"] = result_df["Prediction"].map({
                0: "Legitimate",
                1: "Fraud"
            })

            st.success("✅ Bulk prediction completed!")

            st.write("📊 Final Results:")
            st.dataframe(result_df)

        except Exception as e:
            st.error(f"Error: {e}")
