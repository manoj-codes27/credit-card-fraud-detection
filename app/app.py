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

if st.button("Predict"):
    try:
        input_data = np.array([time] + [0]*28 + [amount]).reshape(1, -1)
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

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

    # 👉 Use a unique key to avoid button state issues
    if st.button("Run Bulk Prediction", key="bulk_predict"):
        try:
            # Convert to numpy (prevents feature name mismatch)
            X = scaler.transform(df.values)

            # Predict
            preds = model.predict(X)

            # Create result dataframe (IMPORTANT FIX)
            result_df = df.copy()
            result_df["Prediction"] = preds
            result_df["Prediction"] = result_df["Prediction"].map({
                0: "Legitimate",
                1: "Fraud"
            })

            st.success("✅ Bulk prediction completed!")

            # 👉 Show ONLY final result (no confusion)
            st.write("📊 Final Results:")
            st.dataframe(result_df)

        except Exception as e:
            st.error(f"Error: {e}")
