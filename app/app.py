import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# =========================
# Load model & scaler safely
# =========================
BASE_DIR = os.path.dirname(__file__)

scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
model_path = os.path.join(BASE_DIR, "fraud_model.pkl")

scaler = pickle.load(open(scaler_path, "rb"))
model = pickle.load(open(model_path, "rb"))

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
    st.info("Note: V1–V28 are anonymized features")

# Default V1–V28
v_features = [0.0] * 28

if st.button("Predict"):
    try:
        input_data = np.array([time, amount] + v_features).reshape(1, -1)
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

    if st.button("Run Bulk Prediction"):
        try:
            # Expected column order (VERY IMPORTANT)
            expected_columns = [
                'Time', 'Amount',
                'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                'V11','V12','V13','V14','V15','V16','V17','V18',
                'V19','V20','V21','V22','V23','V24','V25','V26','V27','V28'
            ]

            # Add missing columns (if any)
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0

            # Reorder columns correctly
            df = df[expected_columns]

            # Scale
            X = scaler.transform(df)

            # Predict
            preds = model.predict(X)

            df["Prediction"] = preds
            df["Prediction"] = df["Prediction"].map({
                0: "Legitimate",
                1: "Fraud"
            })

            st.success("✅ Bulk prediction completed!")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Error: {e}")

                st.success("✅ Bulk prediction completed!")
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error: {e}")
