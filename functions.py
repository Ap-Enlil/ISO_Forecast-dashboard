import pandas as pd
import numpy as np
import requests
from io import StringIO

#=======================
# Data Functions
#=======================
def download_data(filenames):
    """Download and concatenate data files."""
    dfs = []
    for filename in filenames:
        url = f"https://www.eia.gov/electricity/wholesalemarkets/csv/{filename}"
        response = requests.get(url)
        response.raise_for_status()
        dfs.append(pd.read_csv(StringIO(response.text), skiprows=3))
    return pd.concat(dfs, ignore_index=True)

#=======================
# Processing Functions
#=======================

def preprocess_spp(data):
    """SPP-specific data processing with persistence forecast."""
    df = data

    # Use the expected column name for Central Time
    required_columns = ['Local Timestamp Central Time (Interval Beginning)',
                        'SPP Total Actual Load (MW)', 'SPP Total Forecast Load (MW)']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise KeyError(f"Missing required columns for SPP processing: {missing}")

    # Convert and sort timestamps (handling timezone internally)
    df['Timestamp'] = pd.to_datetime(df['Local Timestamp Central Time (Interval Beginning)'])

    # Remove timezone information
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

    df = df.sort_values('Timestamp').set_index('Timestamp')

    # Column standardization (if needed)
    df = df.rename(columns={
        'SPP Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
        'SPP Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
    })

    # Forecast calculations
    df['Forecast Error (MW)'] = df['SystemTotal Forecast Load (MW)'] - df['TOTAL Actual Load (MW)']
    df['MAPE (%)'] = (abs(df['Forecast Error (MW)']) / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100

    # Persistence forecast calculations
    df['Persistence Forecast (MW)'] = df['TOTAL Actual Load (MW)'].shift(24)
    df['Persistence Error (MW)'] = df['Persistence Forecast (MW)'] - df['TOTAL Actual Load (MW)']

    # Calculate Persistence MAPE safely
    load_values = df['TOTAL Actual Load (MW)'].replace(0, np.nan)  # Avoid division by zero
    df['Persistence MAPE (%)'] = (abs(df['Persistence Error (MW)']) / load_values) * 100

    # Rolling metrics
    df['Rolling MAPE (30D)'] = df['MAPE (%)'].rolling('30D').mean()
    df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

    return df
