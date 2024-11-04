import pandas as pd
import numpy as np
import json

def preprocess_data(file_path, output_path='data/sylhet_weather_preprocessed.csv'):
    data = pd.read_csv(file_path)
    data = data[['YEAR', 'MO', 'DY', 'HR', 'T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']]

    # Calculate and store mean and std for denormalization
    means = data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']].mean()
    stds = data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']].std()

    # Normalize the data
    data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']] = (data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']] - means) / stds

    # Save processed data
    data.to_csv(output_path, index=False)

    # Save means and stds for later use
    scaling_factors = {'means': means.to_dict(), 'stds': stds.to_dict()}
    with open('data/scaling_factors.json', 'w') as f:
        json.dump(scaling_factors, f)

# Run preprocessing
preprocess_data('data/sylhet_weather_2001_2024_hourly.csv')
