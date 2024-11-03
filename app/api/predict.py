# app/api/predict.py
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from app.models.load_model import load_prediction_model

# Load the model once when the module is imported
model = load_prediction_model('data/weather_prediction_model.keras')

router = APIRouter()

class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int

@router.get("/")
def root():
    return {"name": "env ai server"}

@router.post("/predict")
def predict_weather(request: PredictionRequest):
    # Preprocess input for prediction (e.g., scaling or feature engineering)
    # Example:
    future_data = np.array([[request.year, request.month, request.day, request.hour]])  # Adjust as needed
    prediction = model.predict(future_data)
    return {"temperature": prediction[0][0], "precipitation": prediction[0][1]}
