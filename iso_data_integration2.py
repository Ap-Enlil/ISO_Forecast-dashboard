import os
import pandas as pd
import numpy as np
import requests
from io import StringIO
import pytz
from sklearn.ensemble import IsolationForest
import streamlit as st

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
        }#,
                # --- Price Data Files for ERCOT ---
        #'rt_filenames': ["spp_lmp_rt_15min_hubs_2024Q1.csv","ercot_lmp_rt_15min_hubs_2024Q2.csv","ercot_lmp_rt_15min_hubs_2024Q3.csv","ercot_lmp_rt_15min_hubs_2024Q4.csv", "ercot_lmp_rt_15min_hubs_2025Q1.csv"],
        #'da_filenames': ["ercot_lmp_da_hr_hubs_2024.csv", "ercot_lmp_da_hr_hubs_2025.csv"],
        # Specify the price column name (for both RT and DA data)
        #'price_column': 'Bus average LMP'
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
        'forecast_column': 'SystemTotal Forecast Load (MW)',
        # --- Price Data Files for ERCOT ---
        'rt_filenames': ["ercot_lmp_rt_15min_hubs_2024Q1.csv","ercot_lmp_rt_15min_hubs_2024Q2.csv","ercot_lmp_rt_15min_hubs_2024Q3.csv","ercot_lmp_rt_15min_hubs_2024Q4.csv", "ercot_lmp_rt_15min_hubs_2025Q1.csv"],
        'da_filenames': ["ercot_lmp_da_hr_hubs_2024.csv", "ercot_lmp_da_hr_hubs_2025.csv"],
        # Specify the price column name (for both RT and DA data)
        'price_column': 'Bus average LMP'
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
    base_url = "https://www.eia.gov/electricity/wholesalemarkets/csv/"
    for filename in filenames:
        url = f"{base_url}{filename}"
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

    # Convert timestamp column to datetime and set as index.
    df['Timestamp'] = pd.to_datetime(df[config['timestamp_column']])
    df = df.sort_values('Timestamp').set_index('Timestamp')

    # Rename columns if a rename_map is provided.
    if 'rename_map' in config:
        df = df.rename(columns=config['rename_map'])
    
    # Calculate forecast error and percentage errors.
    df['Forecast Error (MW)'] = df[config['forecast_column']] - df[config['actual_column']]
    df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df[config['actual_column']]).replace(np.inf, np.nan) * 100
    df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df[config['actual_column']]).replace(np.inf, np.nan) * 100

    # Remove rows where the absolute percentage error exceeds 10%.
    df = df[df['APE (%)'] <= 10]

    # Calculate rolling metrics.
    df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
    df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

    return df.dropna()


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
    # Remove duplicate indices (initial pass).
    df = df[~df.index.duplicated(keep='first')]

    # Ensure index is datetime.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Localize timestamps if not already localized.
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

    # Convert to UTC.
    df = df.tz_convert('UTC')
    
    # Remove duplicates again (in case duplicates were introduced during localization/conversion).
    df = df[~df.index.duplicated(keep='first')]

    # Reindex to a complete hourly range.
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h', tz='UTC')
    df = df.reindex(full_range)
    
    # Interpolate numeric columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    return df
def add_price_data_to_existing_df(existing_df, iso_key, target_column="LMP Difference (USD)"):
    """
    Adds real-time (RT) and day-ahead (DA) price data to an existing DataFrame.

    Args:
        existing_df (pd.DataFrame): The DataFrame to add price data to.
        iso_key (str): The key representing the ISO (e.g., "PJM", "CAISO").
        target_column (str): The name of the column to store the price difference.

    Returns:
        pd.DataFrame: The DataFrame with the added price data, or the original
                      DataFrame if an error occurs.
    """
    config = ISO_CONFIG.get(iso_key)
    if config is None:
        print(f"Error: Configuration for ISO '{iso_key}' not found.")
        st.error(f"Configuration for ISO '{iso_key}' not found.")  # Display error in Streamlit
        return existing_df

    print(f"Configuration for {iso_key}: {config}")

    base_url = "https://www.eia.gov/electricity/wholesalemarkets/csv/"
    price_col = config.get('price_column', 'Bus average LMP')
    print(f"Price column being used: {price_col}")

    # ------------------------------
    # Process Real-Time (RT) Price Data
    # ------------------------------
    rt_dfs = []
    for filename in config.get("rt_filenames", []):
        url = f"{base_url}{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            rt_df = pd.read_csv(StringIO(response.text), skiprows=3, on_bad_lines='skip')

            timestamp_col = 'UTC Timestamp (Interval Ending)'
            if timestamp_col not in rt_df.columns:
                print(f"File {filename} is missing '{timestamp_col}'. Columns found: {rt_df.columns.tolist()}")
                st.error(f"Missing timestamp column in RT file: {filename}")  # Streamlit error
                continue  # Skip to the next file if timestamp is missing

            rt_df[timestamp_col] = pd.to_datetime(rt_df[timestamp_col], errors='coerce', utc=True)
            rt_df = rt_df.set_index(timestamp_col)
            rt_dfs.append(rt_df)
            print(f"Successfully loaded RT file {filename}")

        except Exception as e:
            print(f"Error downloading or processing RT price file {filename}: {e}")
            st.error(f"Error processing RT file: {filename}")  # Streamlit error

    if not rt_dfs:
        print("No RT price data loaded. Price data merge skipped.")
        return existing_df

    rt_df = pd.concat(rt_dfs)
    rt_numeric = rt_df.select_dtypes(include=[np.number])
    rt_df_hourly = rt_numeric.resample('h').mean()

    if price_col in rt_df_hourly.columns:
        rt_df_hourly = rt_df_hourly.rename(columns={price_col: "RT " + price_col})
        print(f"RT price column renamed to: RT {price_col}")
    else:
        print(f"RT price column '{price_col}' not found in the data.")
        print(f"Columns in rt_df_hourly: {rt_df_hourly.columns}")
        st.error(f"RT price column '{price_col}' not found in the data after processing.")

    # ------------------------------
    # Process Day-Ahead (DA) Price Data
    # ------------------------------
    da_dfs = []
    for filename in config.get("da_filenames", []):
        url = f"{base_url}{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            da_df = pd.read_csv(StringIO(response.text), skiprows=3, on_bad_lines='skip')

            timestamp_col = 'UTC Timestamp (Interval Ending)'
            if timestamp_col not in da_df.columns:
                print(f"File {filename} is missing '{timestamp_col}'. Columns found: {da_df.columns.tolist()}")
                st.error(f"Missing timestamp column in DA file: {filename}")
                continue

            da_df[timestamp_col] = pd.to_datetime(da_df[timestamp_col], errors='coerce', utc=True)
            da_df = da_df.set_index(timestamp_col)
            da_dfs.append(da_df)
            print(f"Successfully loaded DA file {filename}")

        except Exception as e:
            print(f"Error downloading or processing DA price file {filename}: {e}")
            st.error(f"Error processing DA file: {filename}")

    if not da_dfs:
        print("No DA price data loaded. Price data merge skipped.")
        return existing_df

    da_df = pd.concat(da_dfs)
    da_numeric = da_df.select_dtypes(include=[np.number])
    da_df_hourly = da_numeric.resample('h').mean()

    if price_col in da_df_hourly.columns:
        da_df_hourly = da_df_hourly.rename(columns={price_col: "DA " + price_col})
        print(f"DA price column renamed to: DA {price_col}")
    else:
        print(f"DA price column '{price_col}' not found in the data.")
        print(f"Columns in da_df_hourly: {da_df_hourly.columns}")
        st.error(f"DA price column '{price_col}' not found in the data after processing.")

    # ------------------------------
    # Merge Price Data with the Existing DataFrame
    # ------------------------------
    print(f"Columns in existing_df before RT merge: {existing_df.columns}")
    print(f"Columns in rt_df_hourly before merge: {rt_df_hourly.columns}")
    try:
        merged_df = existing_df.merge(rt_df_hourly[['RT ' + price_col]], left_index=True, right_index=True, how='left')
        print(f"Columns in merged_df after RT merge: {merged_df.columns}")

        print(f"Columns in da_df_hourly before merge: {da_df_hourly.columns}")
        merged_df = merged_df.merge(da_df_hourly[['DA ' + price_col]], left_index=True, right_index=True, how='left')
        print(f"Columns in merged_df after DA merge: {merged_df.columns}")

    except Exception as e:
        print("Error during merging price data:", e)
        st.error("Error during merging price data. Check console for details.")
        return existing_df

    # Compute the target column as the difference between RT and DA prices, handling missing columns.
    rt_col_name = "RT " + price_col
    da_col_name = "DA " + price_col

    if rt_col_name in merged_df.columns and da_col_name in merged_df.columns:
        merged_df[target_column] = merged_df[rt_col_name] - merged_df[da_col_name]
    else:
        print(f"Error: Could not calculate '{target_column}' because one or both of the following columns are missing: '{rt_col_name}', '{da_col_name}'")
        st.error(f"Could not calculate price difference. Check console for details.")

    return merged_df



# ------------------------------
# Load Data for All ISOs
# ------------------------------
def load_all_iso_data():
    """
    Download, preprocess, and (if available) merge price data for all ISOs.
    Returns:
        dict: {iso_key: DataFrame or None}
    """
    iso_data = {}
    for iso_key, cfg in ISO_CONFIG.items():
        raw_data = download_data(cfg['filenames'])
        if raw_data is not None:
            try:
                # Preprocess the load data.
                df = preprocess_iso_data(raw_data, iso_key)
                # Ensure a uniform hourly UTC index.
                df = ensure_uniform_hourly_index(df, iso_key)
                # If price data is configured (e.g. for ERCOT), merge it.
                # The new function 'add_price_data_to_existing_df' takes care of
                # downloading the price files, resampling RT data, and merging.
                if "rt_filenames" in cfg and "da_filenames" in cfg:
                    df = add_price_data_to_existing_df(df, iso_key, target_column="LMP Difference (USD)")
            except KeyError as e:
                print(f"Error processing data for {iso_key}: {e}")
                df = None
            iso_data[iso_key] = df
        else:
            iso_data[iso_key] = None
    return iso_data



# Example: To load and inspect the data for ERCOT (with price data merged)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_iso_data = load_all_iso_data()
    ercot_data = all_iso_data.get("ERCOT")
    print(ercot_data["LMP Difference (USD)"]   )

        # Create a scatter plot:
    plt.figure(figsize=(10, 6))
    plt.scatter(ercot_data["LMP Difference (USD)"], ercot_data["Forecast Error (MW)"], alpha=0.6, edgecolor='k')
    plt.xlabel("DA - RT Price Difference (USD)")
    plt.ylabel("Forecast Error (MW)")
    plt.title("Forecast Error vs DA-RT Price Difference (ERCOT)")
    plt.grid(True)
    plt.show()
