# app/models/load_model.py
from tensorflow.keras.models import load_model

def load_prediction_model(model_path):
    return load_model(model_path)
