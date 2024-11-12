import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score
import json

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

    # Create model using Input layer
    inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(128, activation='relu', return_sequences=True)(inputs)
    x = LSTM(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(2)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Rest of the configuration remains the same
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=optimizer, loss='mse')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    model.save(model_path)

def calculate_accuracy(y_true, y_pred, tolerance_percentage=10):
    """
    Calculate accuracy based on predictions within tolerance percentage of true values
    """
    within_tolerance = 0
    total_predictions = len(y_true) * y_true.shape[1]

    for i in range(y_true.shape[1]):
        # Calculate tolerance range for each true value
        tolerance = np.abs(y_true[:, i] * (tolerance_percentage / 100))

        # Count predictions within tolerance
        within_tolerance += np.sum(
            np.abs(y_true[:, i] - y_pred[:, i]) <= tolerance
        )

    accuracy = (within_tolerance / total_predictions) * 100
    return accuracy

def evaluate_model(model, X_test, y_test, scaling_factors_path='data/scaling_factors.json'):
    # Make predictions
    y_pred = model.predict(X_test)

    # Load scaling factors for denormalization
    with open(scaling_factors_path, 'r') as f:
        scaling_factors = json.load(f)

    # Denormalize predictions and actual values
    for i in range(2):  # Assuming last 2 columns are your target variables
        y_pred[:, i] = (y_pred[:, i] * scaling_factors['stds']['T2M']) + scaling_factors['means']['T2M']
        y_test[:, i] = (y_test[:, i] * scaling_factors['stds']['T2M']) + scaling_factors['means']['T2M']

    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)

    # Calculate custom accuracy
    accuracy = calculate_accuracy(y_test, y_pred, tolerance_percentage=10)

    print(f"\nModel Performance Metrics:")
    print(f"R-squared Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"Accuracy (within 10% tolerance): {accuracy:.2f}%")

    return r2, accuracy

# Train and save the model
if __name__ == "__main__":
    # Train and save the model
    data_path = 'data/sylhet_weather_preprocessed.csv'
    model_path = 'data/weather_prediction_model.keras'

    print("Training model...")
    train_and_save_model(data_path, model_path)

    print("\nEvaluating model...")
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Prepare data for evaluation
    data = pd.read_csv(data_path)
    X, y = create_sequences(data.values, n_steps=24)
    split = int(len(X) * 0.8)
    _, X_test = X[:split], X[split:]
    _, y_test = y[:split], y[split:]

    # Evaluate the model
    r2, accuracy = evaluate_model(model, X_test, y_test)
    print("accuracy", r2)
