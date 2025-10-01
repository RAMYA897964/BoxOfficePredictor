# backend/app/main.py

import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# ----------------------------
# 1. Load model and scalers
# ----------------------------
#MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model")
#model = joblib.load(os.path.join(MODEL_DIR, "final_xgb_model.joblib"))
#scaler = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.joblib"))
#model = joblib.load("C:\\BoxOfficePredictor\\boxoffice-predictor\\backend\\final_xgb_model.joblib")
#scaler = joblib.load("C:\\BoxOfficePredictor\\boxoffice-predictor\\backend\\standard_scaler.joblib")  # scaler should be trained on BudgetUSD, One_Week_SalesUSD, IMDbRating
model = joblib.load("final_xgb_model.joblib")
scaler = joblib.load("standard_scaler.joblib")


# ----------------------------
# 2. Define FastAPI app
# ----------------------------
app = FastAPI(title="BoxOffice Predictor API")

# ----------------------------# 3. Define request schema
# ----------------------------
class MovieData(BaseModel):
    Genre: str
    BudgetUSD: float
    One_Week_SalesUSD: float
    IMDbRating: float
    Director: str
    LeadActor: str

# ----------------------------
# 4. Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(data: MovieData):
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([data.dict()])

        # Log transform numeric columns
        for col in ["BudgetUSD", "One_Week_SalesUSD"]:
            df[col] = np.log1p(df[col])

        # Standardize numeric columns
        scaled_cols = ["BudgetUSD", "One_Week_SalesUSD", "IMDbRating"]
        df[scaled_cols] = scaler.transform(df[scaled_cols])

        # Handle categorical column
        df["Genre"] = df["Genre"].astype("category")
        df["Director"] = df["Director"].astype("category")
        df["LeadActor"] = df["LeadActor"].astype("category")

        # Ensure correct feature order
        feature_order = ['Genre', 'BudgetUSD', 'One_Week_SalesUSD', 'IMDbRating', 'Director', 'LeadActor']
        df = df[feature_order]

        # Predict
        prediction = model.predict(df)

        # Convert to float
        return {"prediction": float(prediction[0])}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
