import pandas as pd
import numpy as np
import requests
from io import StringIO
import pytz
from sklearn.ensemble import IsolationForest

# ------------------------------
# ISO configuration dictionary
# ------------------------------
ISO_CONFIG = {
    'SPP': {
        'filenames': ["spp_load-temp_hr_2024.csv", "spp_load-temp_hr_2025.csv"],
        'timezone': 'America/Chicago',
        'required_columns': [
            'Local Timestamp Central Time (Interval Beginning)',
            'SPP Total Actual Load (MW)',
            'SPP Total Forecast Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Central Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',  # After renaming
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'SPP Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'SPP Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    },
    'MISO': {
        'filenames': ["miso_load-temp_hr_2024.csv", "miso_load-temp_hr_2025.csv"],
        'timezone': 'America/New_York',
        'required_columns': [
            'Local Timestamp Central Time (Interval Beginning)',
            'TOTAL Actual Load (MW)',
            'SystemTotal Forecast Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Central Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'Local Timestamp Eastern Standard Time (Interval Beginning)':
                'Local Timestamp Central Time (Interval Beginning)',
            'MISO Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'MISO Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    },
    'ERCOT': {
        'filenames': ["ercot_load-temp_hr_2024.csv", "ercot_load-temp_hr_2025.csv"],
        'timezone': 'America/Chicago',
        'required_columns': [
            'Local Timestamp Central Time (Interval Beginning)',
            'SystemTotal Forecast Load (MW)',
            'TOTAL Actual Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Central Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)'
    },
    'CAISO': {
        'filenames': ["caiso_load-temp_hr_2024.csv", "caiso_load-temp_hr_2025.csv"],
        'timezone': 'America/Los_Angeles',
        'required_columns': [
            'Local Timestamp Pacific Time (Interval Beginning)',
            'CAISO Total Actual Load (MW)',
            'CAISO Total Forecast Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Pacific Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'CAISO Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'CAISO Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    },
    'PJM': {
        'filenames': ["pjm_load-temp_hr_2024.csv", "pjm_load-temp_hr_2025.csv"],
        'timezone': 'America/New_York',
        'required_columns': [
            'Local Timestamp Eastern Time (Interval Beginning)',
            'PJM Total Actual Load (MW)',
            'PJM Total Forecast Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Eastern Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'PJM Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'PJM Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    },
    'PJM_Dominion': {
        'filenames': ["pjm_load-temp_hr_2024.csv", "pjm_load-temp_hr_2025.csv"],
        'timezone': 'America/New_York',
        'required_columns': [
            'Local Timestamp Eastern Time (Interval Beginning)',
            'Duke Energy Ohio/Kentucky Actual Load (MW)',
            'Duke Energy Ohio/Kentucky Forecast Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Eastern Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'Duke Energy Ohio/Kentucky Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'Duke Energy Ohio/Kentucky Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    },
    'PJM_Duquesne': {
        'filenames': ["pjm_load-temp_hr_2024.csv", "pjm_load-temp_hr_2025.csv"],
        'timezone': 'America/New_York',
        'required_columns': [
            'Local Timestamp Eastern Time (Interval Beginning)',
            'Duquesne Light Actual Load (MW)',
            'Duquesne Light Forecast Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Eastern Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'Duquesne Light Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'Duquesne Light Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    },
    'PJM_EastKentucky': {
        'filenames': ["pjm_load-temp_hr_2024.csv", "pjm_load-temp_hr_2025.csv"],
        'timezone': 'America/New_York',
        'required_columns': [
            'Local Timestamp Eastern Time (Interval Beginning)',
            'East Kentucky Power Coop Actual Load (MW)',
            'East Kentucky Power Coop Forecast Load (MW)'
        ],
        'timestamp_column': 'Local Timestamp Eastern Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'East Kentucky Power Coop Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'East Kentucky Power Coop Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    }
}


# ------------------------------
# Data Downloading
# ------------------------------
def download_data(filenames):
    """
    Download and concatenate CSV data files.
    """
    dfs = []
    for filename in filenames:
        url = f"https://www.eia.gov/electricity/wholesalemarkets/csv/{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            dfs.append(pd.read_csv(StringIO(response.text), skiprows=3))
        except (requests.exceptions.RequestException, pd.errors.ParserError) as e:
            print(f"Error downloading/parsing {filename}: {e}")
            return None
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ------------------------------
# Preprocessing ISO Data
# ------------------------------
def preprocess_iso_data(data, iso_key):
    """
    Preprocess raw data for a given ISO:
      - Rename columns as needed.
      - Convert timestamp to datetime and sort.
      - Calculate forecast error and percentage error.
      - Calculate rolling metrics.
    """
    if data is None:
        return None

    df = data.copy()
    config = ISO_CONFIG[iso_key]

    # Check for required columns; if not all found, try renaming first.
    if not all(col in df.columns for col in config['required_columns']):
        if 'rename_map' in config:
            df = df.rename(columns=config['rename_map'])
            if not all(col in df.columns for col in config['required_columns']):
                missing = [col for col in config['required_columns'] if col not in df.columns]
                raise KeyError(f"Missing required columns for {iso_key} after renaming: {missing}")
        else:
            missing = [col for col in config['required_columns'] if col not in df.columns]
            raise KeyError(f"Missing required columns for {iso_key}: {missing}")

    # Convert timestamp column to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df[config['timestamp_column']])
    df = df.sort_values('Timestamp').set_index('Timestamp')

    # If needed, rename columns using the mapping
    if 'rename_map' in config:
        df = df.rename(columns=config['rename_map'])
    # Remove outliers from the actual load column
    actual_col = config['actual_column']
    forecast_col = config['forecast_column']
    # Calculate forecast error and percentage errors
    df['Forecast Error (MW)'] = df[config['forecast_column']] - df[config['actual_column']]
    df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df[config['actual_column']]).replace(np.inf, np.nan) * 100
    df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df[config['actual_column']]).replace(np.inf, np.nan) * 100

    # Remove rows where the absolute percentage error exceeds 10%
    df = df[df['APE (%)'] <= 10]

    # Rolling metrics (using time-based windows)
    df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
    df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

    return df.dropna()


# ------------------------------
# Load Data for All ISOs
# ------------------------------
def load_all_iso_data():
    """
    Download and preprocess data for all ISOs.
    Returns:
        dict: {iso_key: DataFrame or None}
    """
    iso_data = {}
    for iso_key, cfg in ISO_CONFIG.items():
        raw_data = download_data(cfg['filenames'])
        if raw_data is not None:
            try:
                df = preprocess_iso_data(raw_data, iso_key)
            except KeyError as e:
                print(f"Error processing data for {iso_key}: {e}")
                df = None
            iso_data[iso_key] = df
        else:
            iso_data[iso_key] = None
    return iso_data


# ------------------------------
# Ensure Uniform Hourly Datetime Index
# ------------------------------
def ensure_uniform_hourly_index(df, iso_key):
    """
    Ensure the DataFrame has a complete hourly index in UTC.
      - Remove duplicates.
      - Localize timestamps with proper DST handling.
      - Convert to UTC.
      - Reindex to a full hourly range and interpolate numeric columns.
    """
    config = ISO_CONFIG[iso_key]
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Localize timestamps if not already localized
    try:
        if df.index.tz is None:
            df = df.tz_localize(
                config['timezone'],
                ambiguous='infer',
                nonexistent='shift_forward'
            )
        else:
            df = df.tz_convert(config['timezone'])
    except pytz.exceptions.AmbiguousTimeError:
        df = df.tz_localize(
            config['timezone'],
            ambiguous=False,
            nonexistent='shift_forward'
        )

    # Convert to UTC
    df = df.tz_convert('UTC')

    # Reindex to a complete hourly range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H', tz='UTC')
    df = df.reindex(full_range)
    
    # Interpolate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    return df
import numpy as np
import pandas as pd
