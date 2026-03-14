# Diabetes Prediction ML App

A machine learning web application that predicts whether a person has diabetes based on medical attributes.

## Features
- Data preprocessing
- Machine learning model training
- FastAPI REST API
- Web interface for prediction

## Tech Stack
- Python
- Scikit-learn
- FastAPI
- Bootstrap
- Pandas
- Joblib

## Dataset Features
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Run Locally

Clone repository

git clone https://github.com/iqbal17115/diabetes-prediction-ml.git

Install dependencies

pip install -r requirements.txt

Run API

uvicorn api.main:app --reload

Open browser

http://127.0.0.1:8000/web/index.html