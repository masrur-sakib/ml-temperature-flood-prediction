# app/utils/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    # Load data
    data = pd.read_csv(input_file)
    # Drop rows with missing values
    data = data.dropna()
    # Scale numeric columns
    scaler = StandardScaler()
    data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']] = scaler.fit_transform(data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']])
    # Save preprocessed data
    data.to_csv(output_file, index=False)
