from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

# BASE_DIR points to project root
BASE_DIR = Path(__file__).resolve().parent.parent  # disease-prediction-ml/

# Model & scaler paths
model_file = BASE_DIR / "models/diabetes_model.pkl"
scaler_file = BASE_DIR / "models/scaler.pkl"

# Load model & scaler
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Mount static files (web folder)
app.mount("/web", StaticFiles(directory=BASE_DIR / "web"), name="web")

# Request schema
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Predict endpoint
@app.post("/predict")
def predict(patient: PatientData):
    data = pd.DataFrame([patient.dict()])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0].tolist()
    return {"prediction": int(pred), "probability": prob}