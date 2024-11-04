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
print("Available columns:", historical_data.columns.tolist())
historical_data['datetime'] = pd.to_datetime({
    'year': historical_data['YEAR'],
    'month': historical_data['MO'],
    'day': historical_data['DY'],
    'hour': historical_data['HR']
})
historical_data.set_index('datetime', inplace=True)

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
    # Convert request date to datetime
    target_date = pd.to_datetime(f'{year}-{month}-{day} {hour}:00:00')

    # Find data from similar dates in previous years
    similar_dates = []
    for prev_year in range(year-5, year):  # Use data from previous 5 years
        similar_date = target_date.replace(year=prev_year)
        mask = (historical_data.index >= similar_date - pd.Timedelta(hours=n_steps)) & \
               (historical_data.index < similar_date)
        if mask.any():
            similar_dates.extend(historical_data[mask].values.tolist())

    if not similar_dates:
        # Fallback to last available sequence if no similar dates found
        similar_dates = historical_data.tail(n_steps).values.tolist()

    # Use the most recent similar sequence
    recent_data = np.array(similar_dates[-n_steps:])
    print("Input sequence data for prediction:", recent_data)  # Debug print
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
