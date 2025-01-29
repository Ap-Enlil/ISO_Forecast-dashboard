import pandas as pd
import numpy as np
import requests
from io import StringIO


#=======================
# Data Functions
#=======================
def download_data(filenames):
    """Download and concatenate data files"""
    dfs = []
    for filename in filenames:
        url = f"https://www.eia.gov/electricity/wholesalemarkets/csv/{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            dfs.append(pd.read_csv(StringIO(response.text), skiprows=3))
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return None  # Or handle the error as appropriate for your application
        except pd.errors.ParserError as e:
            print(f"Error parsing {filename}: {e}")
            return None
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return None

#=======================
# Processing Functions
#=======================

def preprocess_spp(data):
    """SPP-specific data processing with persistence forecast"""
    if data is None:
        return None
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

    return df.dropna()
def preprocess_miso(data):
    """MISO-specific data processing with persistence forecast"""
    if data is None:
        return None
    df = data

    # Use the original expected column name (even though the data is in Eastern Time)
    required_columns = ['Local Timestamp Central Time (Interval Beginning)',  # We'll handle the timezone internally
                        'MISO Total Actual Load (MW)', 'MISO Total Forecast Load (MW)']

    if not all(col in df.columns for col in required_columns):
        # Rename the Eastern Time column to the expected Central Time name TEMPORARILY
        df = df.rename(columns={'Local Timestamp Eastern Standard Time (Interval Beginning)': 'Local Timestamp Central Time (Interval Beginning)'})
        # Recheck if all required columns are present after renaming
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise KeyError(f"Missing required columns for MISO processing: {missing}")

    # Convert and sort timestamps
    df['Timestamp'] = pd.to_datetime(df['Local Timestamp Central Time (Interval Beginning)'])

    # Remove timezone information
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

    df = df.sort_values('Timestamp').set_index('Timestamp')

    # Column standardization (if needed) - modify as necessary
    df = df.rename(columns={
        'MISO Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
        'MISO Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
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

    return df.dropna()
def preprocess_pjm(data):
    """PJM-specific data processing with persistence forecast"""
    if data is None:
        return None
    df = data

    # Ensure required columns are present
    required_columns = ['Local Timestamp Eastern Time (Interval Beginning)',
                        'PJM Total Actual Load (MW)', 'PJM Total Forecast Load (MW)']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise KeyError(f"Missing required columns for PJM processing: {missing}")

    # Convert and sort timestamps
    df['Timestamp'] = pd.to_datetime(df['Local Timestamp Eastern Time (Interval Beginning)'])
    df = df.sort_values('Timestamp').set_index('Timestamp')

    # Column standardization (if needed) - modify as necessary
    df = df.rename(columns={
        'PJM Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
        'PJM Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
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

    return df.dropna()
def preprocess_ercot(data):
    """Enhanced ERCOT processing with persistence forecast"""
    if data is None:
        return None
    df = data

    # Ensure required columns are present
    required_columns = ['Local Timestamp Central Time (Interval Beginning)',
                        'SystemTotal Forecast Load (MW)', 'TOTAL Actual Load (MW)']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise KeyError(f"Missing required columns for ERCOT processing: {missing}")

    # Convert and sort timestamps
    df['Timestamp'] = pd.to_datetime(df['Local Timestamp Central Time (Interval Beginning)'])
    df = df.sort_values('Timestamp').set_index('Timestamp')

    # System forecast calculations
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

    return df.dropna()

def preprocess_caiso(data):
    """CAISO-specific data processing (Revised)"""
    if data is None:
        return None
    df = data

    # Ensure required columns are present
    required_columns = ['Local Timestamp Pacific Time (Interval Beginning)',
                        'CAISO Total Actual Load (MW)', 'CAISO Total Forecast Load (MW)']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise KeyError(f"Missing required columns for CAISO processing: {missing}")

    # Convert and sort timestamps
    df['Timestamp'] = pd.to_datetime(df['Local Timestamp Pacific Time (Interval Beginning)'])
    df = df.sort_values('Timestamp').set_index('Timestamp')

    # Column standardization
    df = df.rename(columns={
        'CAISO Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
        'CAISO Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
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

    return df.dropna()





#=======================
# ISO Configuration
#=======================
ISO_CONFIG = {
    'SPP': {
        'filenames': ["spp_load-temp_hr_2024.csv", "spp_load-temp_hr_2025.csv"],
        'processor': preprocess_spp
    },
    'MISO': {
        'filenames': ["miso_load-temp_hr_2024.csv", "miso_load-temp_hr_2025.csv"],
        'processor': preprocess_miso
    },
    'ERCOT': {
        'filenames': ["ercot_load-temp_hr_2024.csv", "ercot_load-temp_hr_2025.csv"],
        'processor': preprocess_ercot
    },
    'CAISO': {
        'filenames': ["caiso_load-temp_hr_2024.csv", "caiso_load-temp_hr_2025.csv"],
        'processor': preprocess_caiso
    },
    'PJM': {
        'filenames': ["pjm_load-temp_hr_2024.csv", "pjm_load-temp_hr_2025.csv"],
        'processor': preprocess_pjm
    }
}
