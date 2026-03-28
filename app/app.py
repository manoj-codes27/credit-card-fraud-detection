import streamlit as st
import numpy as np
import pickle

st.write("UPDATED VERSION")
# Load model and scaler
scaler = pickle.load(open("app/scaler.pkl", "rb"))
model = pickle.load(open("app/fraud_model.pkl", "rb"))

# Title
st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is fraudulent or legitimate")

st.markdown("---")

# Input section
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0)
    time = st.number_input("Transaction Time", min_value=0.0)

with col2:
    st.info("Note: V1–V28 are anonymized features from dataset")

# Use default values for V1–V28
v_features = [0] * 28

# Prediction
if st.button("Predict"):
    input_data = np.array([time, amount] + v_features).reshape(1, -1)

    # Apply scaling
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
        st.write("⚠️ This transaction is likely suspicious.")
    else:
        st.success("✅ Legitimate Transaction")
        st.write("👍 This transaction appears safe.")
