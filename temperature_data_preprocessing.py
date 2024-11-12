import pandas as pd
import numpy as np
import json

def preprocess_data(file_path, output_path='data/sylhet_weather_preprocessed.csv'):
    data = pd.read_csv(file_path)

    # Add cyclical time features
    data['Hour_Sin'] = np.sin(2 * np.pi * data['HR']/24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['HR']/24)
    data['Month_Sin'] = np.sin(2 * np.pi * data['MO']/12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['MO']/12)

    # Drop original time columns
    data = data.drop(['YEAR', 'MO', 'DY', 'HR'], axis=1)

    # Handle any missing values
    data = data.ffill()

    # Calculate and store mean and std for denormalization
    means = data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']].mean()
    stds = data[['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']].std()

    # Normalize the numerical columns
    columns_to_normalize = ['T2M', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M']
    data[columns_to_normalize] = (data[columns_to_normalize] - means) / stds

    # Save processed data
    data.to_csv(output_path, index=False)

    # Save scaling factors
    scaling_factors = {'means': means.to_dict(), 'stds': stds.to_dict()}
    with open('data/scaling_factors.json', 'w') as f:
        json.dump(scaling_factors, f)

# Run preprocessing
preprocess_data('data/sylhet_weather_2001_2024_hourly.csv')
