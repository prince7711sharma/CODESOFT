import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Theme: page settings
st.set_page_config(page_title="💳 Credit Card Fraud Detector", layout="wide")

# 💠 Custom CSS for theme and nav-style bars
st.markdown("""
    <style>
        body {
            background-color: #f2f7ff;
        }
        .main {
            background-color: #f2f7ff;
        }
        h1 {
            color: #003366;
        }
        .stButton > button {
            background-color: #0066cc;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        .stButton > button:hover {
            background-color: #005bb5;
        }
        .section {
            background-color: #e8f0fe;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# 📢 Title
st.markdown("<h1 style='text-align: center;'>🔍 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter transaction details below to check for fraud in real-time.</p>", unsafe_allow_html=True)
st.markdown("---")

# 🔹 Split into 3 columns: left-navbar, center, right-navbar
left_bar, center, right_bar = st.columns([1.4, 1.2, 1.4])

# 📦 Features
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
midpoint = len(features) // 2
left_features = features[:midpoint]
right_features = features[midpoint:]

user_input = {}

# 🔵 Left "Navbar" Inputs
with left_bar:
    st.markdown("### 📘 Input Panel A")
    with st.container():
        for feature in left_features:
            user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f", key=f"{feature}_left")

# 🟡 Right "Navbar" Inputs
with right_bar:
    st.markdown("### 📙 Input Panel B")
    with st.container():
        for feature in right_features:
            user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f", key=f"{feature}_right")

# 🎯 Center Prediction Panel
with center:
    st.markdown("### 🧪 Prediction Center")
    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][int(prediction)]

        st.markdown("---")
        if prediction == 1:
            st.error("🚨 **Fraudulent Transaction Detected!**")
        else:
            st.success("✅ **Legitimate Transaction**")

        st.markdown(f"**🔎 Confidence Score:** `{prob:.2%}`")
