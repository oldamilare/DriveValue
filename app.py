import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "car_price_model.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

st.set_page_config(
    page_title="DriveValue",
    page_icon="🚗",
    layout="centered"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea55;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-price {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    return joblib.load(FEATURE_PATH)

pipeline = load_model()
feature_cols = load_features()

NUMERIC_FEATURES     = feature_cols["numeric"]
CATEGORICAL_FEATURES = feature_cols["categorical"]
ALL_FEATURES         = feature_cols["all"]

# ── Header ──────────────────────────────────────────────────
st.markdown('<p class="hero-title">🚗 DriveValue</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Nigerian Used Car Price Predictor — powered by Machine Learning</p>', unsafe_allow_html=True)
st.divider()

# ── Input Form ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("🏷️ Brand (Make)", [
        "Toyota", "Honda", "Mercedes-Benz", "BMW", "Ford",
        "Hyundai", "Kia", "Lexus", "Nissan", "Volkswagen",
        "Audi", "Land Rover", "Jeep", "Chevrolet", "Peugeot",
        "Mitsubishi", "Suzuki", "Mazda", "Infiniti", "Acura"
    ])
    year = st.number_input("📅 Year of Manufacture", min_value=1990, max_value=2025, value=2018)
    mileage = st.number_input("🛣️ Mileage (km)", min_value=0, max_value=500000, value=50000, step=5000)
    fuel_type = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid", "CNG"])

with col2:
    gear_type = st.selectbox("⚙️ Gear Type", ["Automatic", "Manual"])
    condition = st.selectbox("✨ Condition", ["Brand New", "Foreign Used", "Nigerian Used"])
    engine_size = st.number_input("🔧 Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    registered_city = st.selectbox("📍 Registered City", [
        "Lagos", "Abuja", "Port Harcourt", "Kano", "Ibadan",
        "Enugu", "Kaduna", "Benin City", "Warri", "Owerri",
        "Calabar", "Jos", "Ilorin", "Uyo", "Asaba",
        "Abeokuta", "Onitsha", "Aba", "Maiduguri", "Sokoto",
        "Zaria", "Akure", "Bauchi", "Makurdi", "Yola",
        "Other"
    ])

st.divider()

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
    row = row[[c for c in ALL_FEATURES if c in row.columns]]
    for col in ALL_FEATURES:
        if col not in row.columns:
            row[col] = np.nan
    row = row[ALL_FEATURES]

    predicted_price = pipeline.predict(row)[0]
    low  = predicted_price * 0.90
    high = predicted_price * 1.10

    st.markdown(f"""
    <div class="result-box">
        <div style="color:#888; font-size:1rem; margin-bottom:0.5rem">Estimated Market Price</div>
        <div class="result-price">₦{predicted_price:,.0f}</div>
        <div style="color:#888; font-size:0.9rem; margin-top:0.5rem">
            Likely range: ₦{low:,.0f} — ₦{high:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Key factors affecting this estimate:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Car Age", f"{car_age} years", delta="newer = higher value")
        st.metric("Mileage", f"{mileage:,} km", delta="lower = higher value")
    with col_b:
        st.metric("Condition", condition)
        st.metric("Brand", make)

st.divider()
st.caption("DriveValue — Built with scikit-learn, SHAP & Streamlit | Nigerian Car Prices Dataset")