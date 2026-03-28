import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("fraud_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is fraudulent or legitimate")

st.markdown("---")

# OPTION 1: Manual input
st.subheader("🔹 Manual Prediction")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0)
    time = st.number_input("Transaction Time", min_value=0.0)

with col2:
    st.info("Note: V1–V28 are anonymized features")

v_features = [0] * 28

if st.button("Predict"):
    input_data = np.array([time, amount] + v_features).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

st.markdown("---")

# OPTION 2: CSV Upload
st.subheader("📂 Bulk Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📊 Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Run Bulk Prediction"):
        X = df.copy()

        # Ensure same format
        X = scaler.transform(X)

        preds = model.predict(X)

        df["Prediction"] = preds
        df["Prediction"] = df["Prediction"].map({0: "Legitimate", 1: "Fraud"})

        st.write("✅ Results:")
        st.dataframe(df)

        st.success("Bulk prediction completed!")
