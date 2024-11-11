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

# Load the original data to get the date information
original_data = pd.read_csv('data/sylhet_weather_2001_2024_hourly.csv')
preprocessed_data = pd.read_csv('data/sylhet_weather_preprocessed.csv')

# Create a datetime index for the preprocessed data
preprocessed_data['datetime'] = pd.to_datetime({
    'year': original_data['YEAR'],
    'month': original_data['MO'],
    'day': original_data['DY'],
    'hour': original_data['HR']
})
preprocessed_data.set_index('datetime', inplace=True)

# Load scaling factors
with open('data/scaling_factors.json', 'r') as f:
    scaling_factors = json.load(f)

def prepare_input_sequence(year, month, day, hour, n_steps=24):
    # Convert request date to datetime
    target_date = pd.to_datetime(f'{year}-{month}-{day} {hour}:00:00')

    # Find data from similar dates in previous years
    similar_dates = []
    for prev_year in range(year-5, year):  # Use data from previous 5 years
        similar_date = target_date.replace(year=prev_year)
        mask = (preprocessed_data.index >= similar_date - pd.Timedelta(hours=n_steps)) & \
               (preprocessed_data.index < similar_date)
        if mask.any():
            sequence = preprocessed_data[mask].values
            if len(sequence) == n_steps:  # Only use complete sequences
                similar_dates.append(sequence)

    if not similar_dates:
        # Fallback to last available sequence if no similar dates found
        sequence = preprocessed_data.tail(n_steps).values
        similar_dates.append(sequence)

    # Use the most recent similar sequence
    recent_data = similar_dates[-1]

    # Select only the required features (first 7 columns)
    input_sequence = recent_data[:, :7]  # T2M,RH2M,PRECTOTCORR,PS,WS10M,Hour_Sin,Hour_Cos

    return input_sequence.reshape(1, n_steps, 7)

@app.get("/")
def root():
    return {"name": "env ai server"}

# Add CORS middleware
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_weather(request: PredictionRequest):
    # Prepare input sequence
    input_sequence = prepare_input_sequence(
        request.year,
        request.month,
        request.day,
        request.hour
    )

    # Make prediction
    prediction = model.predict(input_sequence)
    temperature_pred, precipitation_pred = prediction[0]

    # Denormalize predictions
    temperature = (temperature_pred * scaling_factors['stds']['T2M']) + scaling_factors['means']['T2M']
    precipitation = (precipitation_pred * scaling_factors['stds']['PRECTOTCORR']) + scaling_factors['means']['PRECTOTCORR']

    return {
        "temperature": float(temperature),
        "precipitation": float(precipitation)
    }
