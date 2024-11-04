import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd

class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int

app = FastAPI()
model = tf.keras.models.load_model("data/weather_prediction_model.keras")
historical_data = pd.read_csv('data/sylhet_weather_preprocessed.csv')

# Define allowed origins (modify as needed)
origins = [
    "http://127.0.0.1:5173",  # React app running locally
    "http://localhost:5173"
]

# Add CORS middleware to allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow listed origins only
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

# Load scaling factors
with open('data/scaling_factors.json', 'r') as f:
    scaling_factors = json.load(f)
temperature_mean = scaling_factors['means']['T2M']
temperature_std = scaling_factors['stds']['T2M']
precipitation_mean = scaling_factors['means']['PRECTOTCORR']
precipitation_std = scaling_factors['stds']['PRECTOTCORR']

def get_recent_sequence(year, month, day, hour, n_steps=24):
    recent_data = historical_data.tail(n_steps).values
    return recent_data[:, :-2].reshape(1, n_steps, 7)

@app.post("/predict")
async def predict_weather(request: PredictionRequest):
    input_sequence = get_recent_sequence(request.year, request.month, request.day, request.hour)
    prediction = model.predict(input_sequence)
    normalized_temperature, normalized_precipitation = prediction[0]

    # Denormalize predictions using saved mean and std values
    temperature = (normalized_temperature * temperature_std) + temperature_mean
    precipitation = (normalized_precipitation * precipitation_std) + precipitation_mean

    return {
        "temperature": float(temperature),
        "precipitation": float(precipitation)
    }
