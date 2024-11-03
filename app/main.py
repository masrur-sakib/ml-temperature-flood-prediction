from fastapi import FastAPI
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

# Calculate mean and standard deviation for denormalization
temperature_mean, temperature_std = historical_data['T2M'].mean(), historical_data['T2M'].std()
precipitation_mean, precipitation_std = historical_data['PRECTOTCORR'].mean(), historical_data['PRECTOTCORR'].std()

def get_recent_sequence(year, month, day, hour, n_steps=24):
    recent_data = historical_data.tail(n_steps).values
    return recent_data[:, :-2].reshape(1, n_steps, 7)

@app.post("/predict")
async def predict_weather(request: PredictionRequest):
    input_sequence = get_recent_sequence(request.year, request.month, request.day, request.hour)
    prediction = model.predict(input_sequence)
    normalized_temperature, normalized_precipitation = prediction[0]

    # Denormalize predictions
    temperature = (normalized_temperature * temperature_std) + temperature_mean
    precipitation = (normalized_precipitation * precipitation_std) + precipitation_mean

    return {"temperature": float(temperature), "precipitation": float(precipitation)}

