import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import requests
import json
import time
import datetime

def authenticate(username, password):
                  # Hardcoded value
    AUTH_URL = (
        "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
        "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
        "?username={username}"
        "&password={password}"
        "&grant_type=password"
        "&scope=openid+fec253ea-0d06-4272-a5e6-b478baeecd70+offline_access"
        "&client_id=fec253ea-0d06-4272-a5e6-b478baeecd70"
        "&response_type=id_token"
    )
    try:
        response = requests.post(AUTH_URL.format(username=username, password=password))
        if response.status_code != 200:
            st.error(f"Authentication failed: {response.status_code} {response.text}")
            return None

        response_json = response.json()
        access_token = response_json.get("access_token")
        if not access_token:
            st.error("No access token returned. Check credentials.")
            return None

        return access_token

    except requests.exceptions.RequestException as e:
        st.error(f"Request error during authentication: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"JSON decode error during authentication: {e}")
        return None

def fetch_data(access_token, delivery_date_from, delivery_date_to, subscription_key):
    BASE_URL = "https://api.ercot.com/api/public-reports/np4-732-cd/wpp_hrly_avrg_actl_fcast"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "deliveryDateFrom": delivery_date_from,
        "deliveryDateTo": delivery_date_to
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(BASE_URL, headers=headers, params=params)
            if response.status_code == 200:
                try:
                    data = response.json()
                    return data
                except json.JSONDecodeError:
                    st.error("Invalid JSON returned by the API.")
                    return None
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                time.sleep(retry_after)  # No debug output
            else:
                st.error(f"API request failed: {response.status_code} {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Request error during data fetch: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)

    st.error("Max retries exceeded while fetching data.")
    return None

def get_data_for_delivery_date(access_token, target_date_str, subscription_key):
    data = fetch_data(access_token, target_date_str, target_date_str, subscription_key)
    if not data or 'data' not in data:
        st.error(f"No data or 'data' key found for {target_date_str}")
        return None

    fields = data.get('fields', [])
    if not fields:
        st.error(f"No 'fields' key found for {target_date_str}")
        return None

    columns = [field['name'] for field in fields]
    try:
        df = pd.DataFrame(data['data'], columns=columns)
    except Exception as e:
        st.error(f"Error creating DataFrame: {e}, Data: {data['data']}")
        return None

    if df.empty:
        st.error(f"DataFrame is empty for {target_date_str}")
        return None

    try:
        df['datetime'] = pd.to_datetime(df['deliveryDate']) + pd.to_timedelta(df['hourEnding'], unit='h')
        if 'postedDatetime' in df.columns:
            df['postedDatetime'] = pd.to_datetime(df['postedDatetime'])
        else:
            st.error(f"postedDatetime column not found for {target_date_str}")
            return None

        df = df.set_index('datetime')
        df = df.sort_values('postedDatetime', ascending=False)
        df = df[~df.index.duplicated(keep='first')]
        return df
    except KeyError as e:
        st.error(f"KeyError in get_data_for_delivery_date: {e}. Columns: {df.columns}")
        return None
    except Exception as e:
        st.error(f"Unexpected error in get_data_for_delivery_date: {e}")
        return None

def update_csv(filename, access_token, subscription_key, default_start_date="2024-12-20"):
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename, parse_dates=['datetime'])
            if 'deliveryDateRequested' in existing_df.columns:
                last_date = pd.to_datetime(existing_df['deliveryDateRequested']).max()
            else:
                last_date = existing_df['datetime'].max().normalize()
        except Exception as e:
            st.error(f"Error reading existing CSV: {e}")
            return None
    else:
        existing_df = pd.DataFrame()
        last_date = pd.to_datetime(default_start_date) - pd.Timedelta(days=1)

    new_start_date = last_date + pd.Timedelta(days=1)
    new_end_date = pd.Timestamp.today().normalize()

    if new_start_date > new_end_date:
        return existing_df

    new_data_frames = []
    for target_date in pd.date_range(new_start_date, new_end_date):
        target_date_str = target_date.strftime("%Y-%m-%d")
        df_new = get_data_for_delivery_date(access_token, target_date_str, subscription_key)
        if df_new is not None:
            df_new['deliveryDateRequested'] = target_date_str
            new_data_frames.append(df_new.reset_index())
        else:
             st.write(f"No data fetched for {target_date_str}")
        time.sleep(2)

    if new_data_frames:
        new_df = pd.concat(new_data_frames, ignore_index=True)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.drop_duplicates(inplace=True)

        try:
            updated_df.to_csv(filename, index=False)
        except Exception as e:
            st.error(f"Error saving to CSV: {e}")
            return None
        return updated_df
    else:
        return existing_df

def run_analysis(df, clustering_start_date, clustering_end_date, display_start_date, display_end_date):
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')

    cluster_mask = (df["datetime"] >= clustering_start_date) & (df["datetime"] <= clustering_end_date)
    clustering_df = df.loc[cluster_mask].sort_values("datetime").copy()

    if clustering_df.empty:
        st.error(f"No data found for clustering between {clustering_start_date.date()} and {clustering_end_date.date()}.")
        return

    clustering_df["Error"] = clustering_df["genSystemWide"] - clustering_df["WGRPPSystemWide"]
    clustering_df["RampRate"] = clustering_df["genSystemWide"].diff()
    clustering_df.dropna(subset=["Error", "RampRate"], inplace=True)

    features = clustering_df[["Error", "genSystemWide", "RampRate"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    try:
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clustering_df["Cluster"] = kmeans.fit_predict(features_scaled)
    except Exception as e:
        st.error(f"Error during KMeans clustering: {e}")
        return

    cluster_stats = clustering_df.groupby("Cluster")[["genSystemWide", "Error", "RampRate"]].mean()
    st.write("Cluster Statistics (from clustering period):") # Keep this output
    st.dataframe(cluster_stats)

    curtailment_cluster = cluster_stats["Error"].idxmin()
    clustering_df["Error_Class"] = clustering_df["Cluster"].apply(
        lambda x: "Curtailment" if x == curtailment_cluster else "Normal Forecast Error"
    )

    display_mask = (df["datetime"] >= display_start_date) & (df["datetime"] <= display_end_date)
    filtered_df = df.loc[display_mask].sort_values("datetime").copy()

    if filtered_df.empty:
        st.warning(f"No data found for display between {display_start_date.date()} and {display_end_date.date()}.")
        return

    try:
        filtered_df = filtered_df.merge(
            clustering_df[["datetime", "Error_Class"]],
            on="datetime",
            how="left"
        )
        filtered_df["Error_Class"].fillna("Normal Forecast Error", inplace=True)
    except KeyError as e:
        st.error(f"KeyError during merge: {e}. clustering_df columns: {list(clustering_df.columns)}, filtered_df columns: {list(filtered_df.columns)}")
        return
    except Exception as e:
        st.error(f"Error merging clustering results: {e}")
        return

    filtered_df["genSystemWide"] = pd.to_numeric(filtered_df["genSystemWide"], errors='coerce')
    filtered_df["WGRPPSystemWide"] = pd.to_numeric(filtered_df["WGRPPSystemWide"], errors='coerce')

    before_drop = len(filtered_df)
    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
    filtered_df.dropna(subset=["genSystemWide", "WGRPPSystemWide"], inplace=True)
    after_drop = len(filtered_df)
    if before_drop != after_drop:
        st.write(f"Dropped {before_drop - after_drop} rows with NaN or infinite in genSystemWide/WGRPPSystemWide.")

    filtered_df["Error"] = filtered_df["genSystemWide"] - filtered_df["WGRPPSystemWide"]
    mask_nonzero = (filtered_df["genSystemWide"] != 0)

    overall_MAPE = (
        np.abs(filtered_df.loc[mask_nonzero, "Error"] / filtered_df.loc[mask_nonzero, "genSystemWide"]).mean() * 100
        if mask_nonzero.any() else 0
    )

    non_curt_mask = (filtered_df["Error_Class"] != "Curtailment") & mask_nonzero
    mape_non_curt = (
        np.abs(filtered_df.loc[non_curt_mask, "Error"] / filtered_df.loc[non_curt_mask, "genSystemWide"]).mean() * 100
        if non_curt_mask.any() else 0
    )

    curt_mask = (filtered_df["Error_Class"] == "Curtailment") & mask_nonzero
    mape_curt = (
        np.abs(filtered_df.loc[curt_mask, "Error"] / filtered_df.loc[curt_mask, "genSystemWide"]).mean() * 100
        if curt_mask.any() else 0
    )

    curtail_volume = (
        (filtered_df.loc[curt_mask, "WGRPPSystemWide"] - filtered_df.loc[curt_mask, "genSystemWide"]).clip(lower=0).sum()
        if curt_mask.any() else 0
    )

    st.write(f"Overall MAPE: {overall_MAPE:.2f}%")  # Keep
    st.write(f"MAPE Outside Curtailment Time: {mape_non_curt:.2f}%") # Keep
    st.write(f"MAPE for Curtailment Time: {mape_curt:.2f}%") # Keep
    st.write(f"Total Curtailment Volume (MW): {curtail_volume:.2f}") # Keep

    def detect_daily_ramp_events_and_accuracy(df, actual_col, forecast_col, ramp_increase_threshold=0.25, window_hours=3):

        df['date'] = df['datetime'].dt.date
        daily_results = []
        for day, group in df.groupby('date'):
            group = group.sort_values('datetime')
            daily_min_actual = group[actual_col].min()
            subset = group[group[actual_col] == daily_min_actual]
            if subset.empty:
                continue # no error message needed here
            min_time = subset['datetime'].iloc[0]
            window_end_time = min_time + pd.Timedelta(hours=window_hours)
            window_data = group[(group['datetime'] >= min_time) & (group['datetime'] <= window_end_time)]
            if window_data.empty:
                continue # no error message
            window_max_actual = window_data[actual_col].max()
            ramp_actual = window_max_actual - daily_min_actual
            ramp_event = ramp_actual >= daily_min_actual * ramp_increase_threshold
            if ramp_event:
                closest_min_idx = (group['datetime'] - min_time).abs().argsort()[:1]
                forecast_at_min = group.iloc[closest_min_idx][forecast_col].values[0]
                window_max_forecast = window_data[forecast_col].max()
                ramp_forecast = window_max_forecast - forecast_at_min
                ramp_error = abs(ramp_forecast - ramp_actual)
                ramp_accuracy = max(0, (1 - (ramp_error / ramp_actual)) * 100)
            else:
                window_max_forecast = np.nan
                ramp_forecast = np.nan
                ramp_error = np.nan
                ramp_accuracy = np.nan

            daily_results.append({
                'date': day,
                'daily_min_actual': daily_min_actual,
                'min_time': min_time,
                'window_max_actual': window_max_actual,
                'window_max_forecast': window_max_forecast,
                'ramp_actual': ramp_actual,
                'ramp_forecast': ramp_forecast,
                'ramp_event': ramp_event,
                'ramp_error': ramp_error,
                'ramp_accuracy': ramp_accuracy
            })
        return pd.DataFrame(daily_results)

    daily_ramp_df = detect_daily_ramp_events_and_accuracy(
        filtered_df.copy(),
        actual_col='genSystemWide',
        forecast_col='STWPFSystemWide',  # Corrected forecast column
        ramp_increase_threshold=0.2,
        window_hours=3
    )

    # Use .empty attribute, not method.  Also check for NaN values.
    valid_ramps = daily_ramp_df[daily_ramp_df['ramp_event'] & daily_ramp_df['ramp_accuracy'].notna()]
    overall_ramp_accuracy = valid_ramps['ramp_accuracy'].mean() if not valid_ramps.empty else 0  # Removed ()
    st.write(f"Overall Ramp Accuracy: {overall_ramp_accuracy:.2f}%") #keep

    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(filtered_df["datetime"], filtered_df["genSystemWide"], label="genSystemWide", linewidth=2, color="green")
    ax2.plot(filtered_df["datetime"], filtered_df["WGRPPSystemWide"], label="WGRPPSystemWide", linewidth=2, color="blue")
    ax2.fill_between(filtered_df["datetime"], filtered_df["genSystemWide"], alpha=0.3, color="green")
    curt_points = filtered_df[filtered_df["Error_Class"] == "Curtailment"]
    ax2.scatter(curt_points["datetime"], curt_points["genSystemWide"], color="red", label="Curtailment Candidate", zorder=5)
    ax2.set_title(f"Wind Generation vs. Time with Curtailment Candidates\n({display_start_date.date()} to {display_end_date.date()})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Generation (MW)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    fig1, ax1 = plt.subplots(figsize=(15, 6))
    for label, group in filtered_df.groupby("Error_Class"):
        ax1.scatter(group["datetime"], group["Error"], label=label, s=10)
    ax1.set_title("Forecast Error Time Series Classified by Unsupervised Clustering")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Error (genSystemWide - WGRPPSystemWide)")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)




    fig3, ax3 = plt.subplots(figsize=(15, 6))
    ax3.plot(filtered_df['datetime'], filtered_df['genSystemWide'], label='Actual Generation (genSystemWide)', color='blue')
    for idx, row in daily_ramp_df.iterrows():
        ax3.plot(row['min_time'], row['daily_min_actual'], 'ro', markersize=8, label='Daily Minimum' if idx == 0 else "")
        if row['ramp_event']:
            day_data = filtered_df[filtered_df['datetime'].dt.date == row['date']]
            window_data = day_data[(day_data['datetime'] >= row['min_time']) & (day_data['datetime'] <= row['min_time'] + pd.Timedelta(hours=3))]
            if not window_data.empty:
                window_max_time = window_data[window_data['genSystemWide'] == row['window_max_actual']]['datetime'].iloc[0]
                ax3.plot(window_max_time, row['window_max_actual'], 'gs', markersize=8, label='Ramp Event (Actual)' if idx == 0 else "")
    ax3.set_title("Actual Generation with Detected Daily Ramp Events")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Generation (MW)")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(15, 6))
    ax4.bar(valid_ramps['date'].astype(str), valid_ramps['ramp_accuracy'], color='green')
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Ramp Accuracy (%)")
    ax4.set_title("Daily Ramp Forecast Accuracy\n(Comparing Forecast Ramp vs. Actual Ramp)")
    plt.xticks(rotation=45)
    ax4.grid(True)
    st.pyplot(fig4)

    overall_rmse = np.sqrt(mean_squared_error(filtered_df.loc[mask_nonzero, "genSystemWide"], filtered_df.loc[mask_nonzero, "WGRPPSystemWide"]))
    rmse_non_curt = np.sqrt(mean_squared_error(filtered_df.loc[non_curt_mask, "genSystemWide"], filtered_df.loc[non_curt_mask, "WGRPPSystemWide"]))
    st.write(f"Overall RMSE: {overall_rmse:.2f} MW") # Keep
    st.write(f"RMSE Outside Curtailment Time: {rmse_non_curt:.2f} MW") # Keep
def render_weather_tab(start_date, end_date):
    st.header("ERCOT Wind Power Forecast Analysis")

    username = "antoine.bertoncello@gmail.com"
    password = "nEUQUEN2025!"
    subscription_key = "8daa9c431fc34bb09734abdb83791b5f"

    filename = "delivery_data_last_year_filtered.csv"

    display_start_datetime = pd.to_datetime(start_date)
    display_end_datetime = pd.to_datetime(end_date)
    clustering_start_datetime = display_start_datetime
    clustering_end_datetime = display_end_datetime

    if st.button("Run Analysis"):
        access_token = authenticate(username, password)
        if access_token:
            updated_df = update_csv(filename, access_token, subscription_key)
            if updated_df is not None and not updated_df.empty:
                run_analysis(updated_df, clustering_start_datetime, clustering_end_datetime, display_start_datetime, display_end_datetime)
            else:
                st.error("No data available for analysis.")
        else:
            st.error("Authentication failed. Check credentials.")