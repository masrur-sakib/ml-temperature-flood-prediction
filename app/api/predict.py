# app/api/predict.py
from fastapi import APIRouter, HTTPException
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

class InputData(BaseModel):
    features: list

@router.get("/")
def root():
    return {"name": "env ai server"}

@router.post("/predict")
async def predict(data: InputData):
    try:
        # Reshape the input data to match the expected shape (None, 24, 7)
        input_data = np.array(data.features)
        input_data = input_data.reshape(1, 24, 7)  # Reshape to match model's expected input

        # Make prediction
        prediction = model.predict(input_data)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
