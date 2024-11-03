import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, :-2])
        y.append(data[i + n_steps, -2:])
    return np.array(X), np.array(y)

def train_and_save_model(data_path, model_path, n_steps=24):
    data = pd.read_csv(data_path)
    X, y = create_sequences(data.values, n_steps)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(32, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    model.save(model_path)

# Train and save the model
train_and_save_model('data/sylhet_weather_preprocessed.csv', 'data/weather_prediction_model.keras')
