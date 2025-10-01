import joblib
import numpy as np
import pandas as pd
import os

# Get absolute path to model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# Load all necessary joblib files
model = joblib.load(os.path.join(MODEL_DIR, "final_xgb_model.joblib"))
log_transform_cols = joblib.load(os.path.join(MODEL_DIR, "log_transform_cols.joblib"))
scaled_cols = joblib.load(os.path.join(MODEL_DIR, "scaled_cols.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.joblib"))

# List of expected features
feature_order = [
    'Genre','BudgetUSD', 'One_Week_SalesUSD', 'IMDbRating',
    'Director', 'LeadActor'
]


def preprocess_input(data: dict) -> pd.DataFrame:
    """Convert raw input dict into a properly formatted DataFrame for prediction."""
    df = pd.DataFrame([data])

    # 2️⃣ Log transform skewed columns
    for col in log_transform_cols:
        df[col] = np.log1p(df[col])

    # 3️⃣ Scale numeric columns
    df[scaled_cols] = scaler.transform(df[scaled_cols])

    # 4️⃣ Convert categorical columns to category dtype
    df['Genre'] = df['Genre'].astype('category')
    df['Director'] = df['Director'].astype('category')
    df['LeadActor'] = df['LeadActor'].astype('category')

    # 5️⃣ Reorder columns to match training
    df = df[feature_order]

    return df


def predict_boxoffice(data: dict) -> float:
    """Run prediction pipeline."""
    processed_df = preprocess_input(data)
    prediction = model.predict(processed_df)[0]
    return float(prediction)
