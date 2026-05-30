# DriveValue 🚗

A machine learning web application that predicts used car prices in Nigeria based on brand, year, mileage, condition, and other features.

## Live App
[DriveValue on Streamlit](https://drivevalue.streamlit.app)

## Project Structure
DriveValue/
├── app.py
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_model_explainability.ipynb
├── model/
│   ├── car_price_model.joblib
│   ├── preprocessor.joblib
│   └── feature_columns.joblib
└── assets/
├── shap_bar.png
├── shap_summary.png
└── shap_waterfall.png

## Workflow
1. **EDA** — Explored price distributions, mileage vs price, brand frequency and correlation heatmaps
2. **Modelling** — scikit-learn Pipeline with ColumnTransformer, OneHotEncoder and Random Forest Regressor
3. **Explainability** — SHAP TreeExplainer with bar, beeswarm and waterfall plots
4. **Deployment** — Streamlit app deployed on Streamlit Cloud

## Key Technical Decisions
- **Random Forest** chosen for its robustness to outliers and built-in feature importances
- **Pipeline** chains imputation, encoding, scaling and model into one object
- **SHAP** explains each prediction — showing which features pushed price up or down
- **car_age** engineered from year of manufacture (2025 - year)

## Dataset
Nigerian Used Car Prices — [Kaggle](https://www.kaggle.com/datasets/makindekayode/nigerian-car-prices-dataset)

## Tech Stack
- Python, scikit-learn, pandas, numpy
- SHAP, matplotlib, seaborn
- Streamlit
