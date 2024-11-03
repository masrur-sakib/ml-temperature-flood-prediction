from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Define the request body
class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int
    hour: int

app = FastAPI()
model = tf.keras.models.load_model("data/weather_prediction_model.keras")

@app.post("/predict")
async def predict_weather(request: PredictionRequest):
    # Example dummy sequence (replace with actual pre-processing if available)
    # Here we create a dummy sequence of shape (1, 24, 7)
    input_sequence = np.random.rand(1, 24, 7)  # replace with actual past data or pre-processed data

    # Make a prediction
    try:
        prediction = model.predict(input_sequence)
        temperature, precipitation = prediction[0]  # unpack results
        return {
            "temperature": float(temperature),
            "precipitation": float(precipitation)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
