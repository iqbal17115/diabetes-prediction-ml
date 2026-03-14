import pandas as pd
import joblib
from pathlib import Path

# 1️⃣ Paths
BASE_DIR = Path(__file__).resolve().parent.parent
model_file = BASE_DIR / "models/diabetes_model.pkl"
scaler_file = BASE_DIR / "models/scaler.pkl"

# Load model & scaler
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Columns must match the training features
new_data = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 32
}])

# Scale new data
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

print("Prediction (0=No, 1=Yes):", prediction[0])
print("Prediction probability [No, Yes]:", probability[0])