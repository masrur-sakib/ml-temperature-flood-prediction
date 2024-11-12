import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_flood_model(data_path, model_path):
    # Load preprocessed data
    data = pd.read_csv(data_path)

    # Drop rows with NaN values in the target column 'flood'
    data = data.dropna(subset=['flood'])

    X = data[['station_name', 'year', 'month']].values
    y = data['flood'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model with Input layer
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile and train the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

    # Evaluate the model on the test set
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy on test data: {accuracy * 100:.2f}%")

    # Save the model
    model.save(model_path)

if __name__ == "__main__":
    train_flood_model('data/flood_data_preprocessed.csv', 'data/flood_prediction_model.keras')
