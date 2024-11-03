import pandas as pd
import numpy as np

def preprocess_data(file_path, n_steps=24):
    data = pd.read_csv(file_path)
    data = data[['YEAR', 'MO', 'DY', 'HR', 'T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']]
    # Normalize the data
    data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']] = data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']].apply(lambda x: (x - x.mean()) / x.std())
    return data

# Save processed data
processed_data = preprocess_data('data/sylhet_weather_2001_2024_hourly.csv')
processed_data.to_csv('data/sylhet_weather_preprocessed.csv', index=False)
