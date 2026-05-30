import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="DriveValue",
    page_icon="🚗",
    layout="centered"
)

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model/car_price_model.joblib")

@st.cache_resource
def load_features():
    return joblib.load("model/feature_columns.joblib")

pipeline = load_model()
feature_cols = load_features()

NUMERIC_FEATURES     = feature_cols["numeric"]
CATEGORICAL_FEATURES = feature_cols["categorical"]
ALL_FEATURES         = feature_cols["all"]

# ── Header ──────────────────────────────────────────────────
st.title("🚗 DriveValue")
st.markdown("#### Nigerian Used Car Price Predictor")
st.markdown("Fill in your car details below to get an instant price estimate.")
st.divider()

# ── Input Form ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Brand (Make)", [
        "Toyota", "Honda", "Mercedes-Benz", "BMW", "Ford",
        "Hyundai", "Kia", "Lexus", "Nissan", "Volkswagen",
        "Audi", "Land Rover", "Jeep", "Chevrolet", "Peugeot",
        "Mitsubishi", "Suzuki", "Mazda", "Infiniti", "Acura"
    ])
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2018)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000, step=5000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid", "CNG"])

with col2:
    gear_type = st.selectbox("Gear Type", ["Automatic", "Manual"])
    condition = st.selectbox("Condition", ["Brand New", "Foreign Used", "Nigerian Used"])
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    registered_city = st.selectbox("Registered City", [
        "Lagos", "Abuja", "Port Harcourt", "Kano", "Ibadan",
        "Enugu", "Kaduna", "Benin City", "Warri", "Owerri"
    ])

st.divider()

# ── Predict ─────────────────────────────────────────────────
if st.button("💰 Estimate Price", use_container_width=True, type="primary"):
    car_age = 2025 - year

    input_data = {
        "car_age": car_age,
        "Mileage": mileage,
        "Engine Size": engine_size,
        "Horse Power": 150,
        "Number of Cylinders": 4,
        "Seats": 5,
        "Make": make,
        "fuel type": fuel_type,
        "gear type": gear_type,
        "Condition": condition,
        "Drivetrain": "FWD",
        "Registered city": registered_city,
        "Selling Condition": "Good",
        "Bought Condition": "Foreign Used"
    }

    row = pd.DataFrame([input_data])

    # Keep only columns the model was trained on
    row = row[[c for c in ALL_FEATURES if c in row.columns]]

    # Add any missing columns with defaults
    for col in ALL_FEATURES:
        if col not in row.columns:
            row[col] = np.nan

    row = row[ALL_FEATURES]

    predicted_price = pipeline.predict(row)[0]

    st.success(f"### Estimated Price: ₦{predicted_price:,.0f}")

    low  = predicted_price * 0.90
    high = predicted_price * 1.10
    st.info(f"**Likely price range:** ₦{low:,.0f} — ₦{high:,.0f}")

    st.markdown("---")
    st.markdown("**Key factors affecting this estimate:**")
    st.markdown(f"- Car age: **{car_age} years** (newer = higher value)")
    st.markdown(f"- Mileage: **{mileage:,} km** (lower = higher value)")
    st.markdown(f"- Condition: **{condition}** (Brand New > Foreign Used > Nigerian Used)")
    st.markdown(f"- Brand: **{make}**")

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.caption("DriveValue — Built with scikit-learn, SHAP and Streamlit | Nigerian Car Prices Dataset")
