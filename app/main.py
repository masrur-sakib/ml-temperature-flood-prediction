# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# Load model and preprocessing components
model = tf.keras.models.load_model('data/flood_prediction_model.keras')
label_encoder = joblib.load('data/label_encoder.pkl')
scaler = joblib.load('data/scaler.pkl')

app = FastAPI()

# Define input schema
class FloodPredictionRequest(BaseModel):
    station_name: str
    year: int
    month: int

# Prediction endpoint
@app.post('/predict_flood')
def predict_flood(request: FloodPredictionRequest):
    # Preprocess inputs
    station_encoded = label_encoder.transform([request.station_name])[0]
    year_scaled, month_scaled = scaler.transform([[request.year, request.month]])[0]

    # Model input
    model_input = np.array([[station_encoded, year_scaled, month_scaled]])

    # Make prediction
    prediction = model.predict(model_input)[0][0]

    # Return 1 if probability > 0.5, else 0
    flood_prediction = int(prediction > 0.5)
    print(flood_prediction)
    return {"flood": flood_prediction}
