import pandas as pd
import numpy as np
import requests
from io import StringIO
import pytz

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
def preprocess_iso_data(data, iso_key):
    """
    Generic preprocessing function for all ISOs.

    """
    if data is None:
        return None
    df = data
    config = ISO_CONFIG[iso_key]

    
    if iso_key == "ERCOT":
        # Fetch both actual and forecast data
        print("YoMAMA")
        print(df.columns)
        
    # Ensure required columns are present
    if not all(col in df.columns for col in config['required_columns']):
        missing = [col for col in config['required_columns'] if col not in df.columns]
        
        # Attempt to rename columns if a mapping is provided
        if 'rename_map' in config:
            df = df.rename(columns=config['rename_map'])
            # Recheck if all required columns are present after renaming
            if not all(col in df.columns for col in config['required_columns']):
                missing = [col for col in config['required_columns'] if col not in df.columns]
                raise KeyError(f"Missing required columns for {iso_key} processing after renaming: {missing}")
        else:
            raise KeyError(f"Missing required columns for {iso_key} processing: {missing}")

    # Convert and sort timestamps
    df['Timestamp'] = pd.to_datetime(df[config['timestamp_column']])
    df = df.sort_values('Timestamp').set_index('Timestamp')

    # Column standardization (if needed)
    if 'rename_map' in config:
        df = df.rename(columns=config['rename_map'])

    # Forecast calculations
    df['Forecast Error (MW)'] = df[config['forecast_column']] - df[config['actual_column']]
    df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df[config['actual_column']]).replace(np.inf, np.nan) * 100
    df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df[config['actual_column']]).replace(np.inf, np.nan) * 100

    # Rolling metrics
    df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
    df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

    return df.dropna()

def load_all_iso_data():
    """
    Download and preprocess data for ALL ISOs at once.
    Returns a dict of DataFrames, keyed by ISO name.
    e.g. {'SPP': df_spp, 'ERCOT': df_ercot, ...}
    """
    iso_data = {}
    for iso_key, cfg in ISO_CONFIG.items():
        raw_data = download_data(cfg['filenames'])
        if raw_data is not None:
            df = preprocess_iso_data(raw_data, iso_key)
            iso_data[iso_key] = df
        else:
            iso_data[iso_key] = None
    return iso_data
ISO_CONFIG = {
    'SPP': {
        'filenames': ["spp_load-temp_hr_2024.csv", "spp_load-temp_hr_2025.csv"],
        'timezone': 'America/Chicago',
        'required_columns': ['Local Timestamp Central Time (Interval Beginning)', 'SPP Total Actual Load (MW)', 'SPP Total Forecast Load (MW)'],
        'timestamp_column': 'Local Timestamp Central Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',  # Corrected after rename
        'forecast_column': 'SystemTotal Forecast Load (MW)',  # Corrected after rename
        'rename_map': {
            'SPP Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'SPP Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    },
    'MISO': {
        'filenames': ["miso_load-temp_hr_2024.csv", "miso_load-temp_hr_2025.csv"],
        'timezone': 'America/New_York',

        # REQUIRED COLUMNS *after* rename
        'required_columns': [
            'Local Timestamp Central Time (Interval Beginning)',
            'TOTAL Actual Load (MW)',
            'SystemTotal Forecast Load (MW)'
        ],

        # The column that you reference later for DateTime:
        'timestamp_column': 'Local Timestamp Central Time (Interval Beginning)',

        # The final column names used in your forecast-error calculations:
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',

        # The old => new mapping so your code can unify the columns
        'rename_map': {
            'Local Timestamp Eastern Standard Time (Interval Beginning)':
                'Local Timestamp Central Time (Interval Beginning)',

            'MISO Total Actual Load (MW)':
                'TOTAL Actual Load (MW)',

            'MISO Total Forecast Load (MW)':
                'SystemTotal Forecast Load (MW)'
        }
    },

    'ERCOT': {
        'filenames': ["ercot_load-temp_hr_2024.csv", "ercot_load-temp_hr_2025.csv"],
        'timezone': 'America/Chicago',
        'required_columns': ['Local Timestamp Central Time (Interval Beginning)', 'SystemTotal Forecast Load (MW)', 'TOTAL Actual Load (MW)'],
        'timestamp_column': 'Local Timestamp Central Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)'
    },

    'CAISO': {
        'filenames': ["caiso_load-temp_hr_2024.csv", "caiso_load-temp_hr_2025.csv"],
        'timezone': 'America/Los_Angeles',
        'required_columns': ['Local Timestamp Pacific Time (Interval Beginning)', 'CAISO Total Actual Load (MW)', 'CAISO Total Forecast Load (MW)'],
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
        'required_columns': ['Local Timestamp Eastern Time (Interval Beginning)', 'PJM Total Actual Load (MW)', 'PJM Total Forecast Load (MW)'],
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
        'required_columns': ['Local Timestamp Eastern Time (Interval Beginning)', 'Duke Energy Ohio/Kentucky Actual Load (MW)', 'Duke Energy Ohio/Kentucky Forecast Load (MW)'],
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
        'required_columns': ['Local Timestamp Eastern Time (Interval Beginning)', 'Duquesne Light Actual Load (MW)', 'Duquesne Light Forecast Load (MW)'],
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
        'required_columns': ['Local Timestamp Eastern Time (Interval Beginning)', 'East Kentucky Power Coop Actual Load (MW)', 'East Kentucky Power Coop Forecast Load (MW)'],
        'timestamp_column': 'Local Timestamp Eastern Time (Interval Beginning)',
        'actual_column': 'TOTAL Actual Load (MW)',
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        'rename_map': {
            'East Kentucky Power Coop Actual Load (MW)': 'TOTAL Actual Load (MW)',
            'East Kentucky Power Coop Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
        }
    }
}

def ensure_uniform_hourly_index(df, iso_key):
    """
    Robust timezone handling with explicit DST ambiguity resolution
    """
    config = ISO_CONFIG[iso_key]
    
    # 1. Remove duplicates first
    df = df[~df.index.duplicated(keep='first')]
    
    # 2. Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 3. Handle timezone conversion with DST parameters
    try:
        if df.index.tz is None:
            # Localize with explicit DST handling
            df = df.tz_localize(
                config['timezone'],
                ambiguous='infer',  # Let pandas infer DST based on timestamp order
                nonexistent='shift_forward'  # Handle spring-forward transitions
            )
        else:
            df = df.tz_convert(config['timezone'])
    except pytz.exceptions.AmbiguousTimeError:
        # Fallback for ambiguous times: assume non-DST (standard time)
        df = df.tz_localize(
            config['timezone'],
            ambiguous=False,
            nonexistent='shift_forward'
        )
    
    # 4. Convert to UTC for uniform handling
    df = df.tz_convert('UTC')
    
    # 5. Create complete UTC index
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='H',
        tz='UTC'
    )
    
    # 6. Reindex and interpolate
    df = df.reindex(full_range)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    
    return df