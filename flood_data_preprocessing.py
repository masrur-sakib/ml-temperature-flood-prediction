import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_flood_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Select relevant columns and target column
    data = data[['station_name', 'year', 'month', 'flood']]  # Assuming 'flood' is a binary column for flood occurrence

    # Encode categorical data (station_name)
    label_encoder = LabelEncoder()
    data['station_name'] = label_encoder.fit_transform(data['station_name'])

    # Scaling numerical features
    scaler = StandardScaler()
    data[['year', 'month']] = scaler.fit_transform(data[['year', 'month']])

    # Save encoders and scaler for later use in prediction
    joblib.dump(label_encoder, 'data/label_encoder.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')

    # Save the processed data
    data.to_csv('data/flood_data_preprocessed.csv', index=False)

if __name__ == "__main__":
    preprocess_flood_data('data/flood_data.csv')
