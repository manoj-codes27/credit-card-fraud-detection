import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("app/fraud_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details to predict whether it's fraud or not")

# Example inputs (simplified)
amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0.0)

# Dummy inputs for V1–V28 (since dataset is anonymized)
v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    v_features.append(val)

# Prediction
if st.button("Predict"):
    input_data = np.array([time, amount] + v_features).reshape(1, -1)
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
