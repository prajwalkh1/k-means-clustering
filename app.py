import streamlit as st
import numpy as np
import json
import os

st.title("💼 Customer Segmentation (K-Means)")
# Load model
with open("model.json", "r") as f:
    data = json.load(f)

centers = np.array(data["centers"])
mean = np.array(data["mean"])
scale = np.array(data["scale"])

# Inputs
income = st.number_input("Annual Income", 0.0, 1000000.0, 50000.0)
purchase = st.number_input("Purchase Amount", 0.0, 10000.0, 200.0)
frequency = st.number_input("Purchase Frequency", 0.0, 100.0, 10.0)
loyalty = st.slider("Loyalty Score", 0.0, 10.0, 5.0)

# Predict
if st.button("Predict Segment"):
    x = np.array([income, purchase, frequency, loyalty])
    x_scaled = (x - mean) / scale

    distances = np.linalg.norm(centers - x_scaled, axis=1)
    cluster = np.argmin(distances)

    st.success(f"Cluster: {cluster}")

# Show graphs
st.subheader("📊 Visual Insights")

if os.path.exists("outputs"):
    for img in os.listdir("outputs"):
        st.image(f"outputs/{img}")