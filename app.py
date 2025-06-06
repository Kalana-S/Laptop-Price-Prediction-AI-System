import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("laptop_price_predictor.pickle", "rb") as file:
    model = pickle.load(file)

st.title("💻 Laptop Price Predictor")

st.sidebar.header("Enter Laptop Specifications")

# Categorical inputs
company = st.sidebar.selectbox("Company", ['Dell', 'Apple', 'HP', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Other'])
type_name = st.sidebar.selectbox("Type", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook'])
cpu = st.sidebar.selectbox("CPU Brand", ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD', 'Other'])
gpu = st.sidebar.selectbox("GPU Brand", ['Intel', 'Nvidia', 'AMD'])
os = st.sidebar.selectbox("Operating System", ['Windows', 'Mac', 'Linux', 'Other'])

# Numerical inputs
ram = st.sidebar.slider("RAM (GB)", 2, 64, 8)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1, value=1.5)
touchscreen = st.sidebar.selectbox("Touchscreen", ['No', 'Yes'])
ips = st.sidebar.selectbox("IPS Display", ['No', 'Yes'])
st.sidebar.markdown("### 💾 Memory (Storage Type in GB)")
hdd = st.sidebar.slider("HDD (GB)", 0, 2000, 0, step=128)
ssd = st.sidebar.slider("SSD (GB)", 0, 2000, 256, step=128)
flash = st.sidebar.slider("Flash Storage (GB)", 0, 1024, 0, step=64)
hybrid = st.sidebar.slider("Hybrid Storage (GB)", 0, 2000, 0, step=128)

# Convert to binary
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Create input dataframe
input_data = {
    'Ram': ram,
    'Weight': weight,
    'Touchscreen': touchscreen,
    'IPS': ips,
    'HDD': hdd,
    'SSD': ssd,
    'Flash_Storage': flash,
    'Hybrid': hybrid,
    f'Company_{company}': 1,
    f'TypeName_{type_name}': 1,
    f'cpu_name_{cpu}': 1,
    f'Gpu_name_{gpu}': 1,
    f'OpSys_{os}': 1
}

input_df = pd.DataFrame([input_data])

# Ensure all required columns exist
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# Arrange columns in exact model order
input_df = input_df[model.feature_names_in_]

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: €{prediction:.2f}")
