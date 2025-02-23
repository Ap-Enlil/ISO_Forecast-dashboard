
import streamlit as st
st.set_page_config(page_title="Model Forecast Comparison", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Import our own modules (assuming these are in the same directory or accessible)
from iso_data_integration import ISO_CONFIG, load_all_iso_data, ensure_uniform_hourly_index
from metrics_calculation import compute_iso_metrics
from rendering import load_data, get_global_date_range, render_comparison_tab, render_iso_analysis_tab
from rendering_Weather2Pow import render_weather_tab
from rendering_long_term_forecast import render_long_term_tab
from functions import  load_config
from long_term_forecast_data import load_data_long_term_ercot
# -- Set page configuration

# ------------------------------
# Custom CSS (same as before, no changes needed here)
# ------------------------------
st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background: #f0f2f6;
    }
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        padding: 20px;
    }
    /* Headings */
    h1, h2, h3, h4 {
        color: #333333;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-size: 16px;
    }
    /* DataFrame styling */
    .dataframe-container {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.title("Forecast Model Registry – Comparison & Analysis")
    st.markdown("Welcome to the forecast analysis tool. Use the sidebar to select the date range for forecast performance.")

    # Load ISO data
    iso_data_dict = load_data()

    # Load ISO Config
    ISO_CONFIG = load_config()
    if ISO_CONFIG is None:
        st.error("Could not load ISO configuration.")
        return

    # Filter for long-term ISOs
    long_term_isos = {}
    for iso_name, config in ISO_CONFIG.items():
        if config.get("timeframe") == "long":
            long_term_isos[iso_name] = config

    # Convert the dict keys to a list
    long_term_iso_options = list(long_term_isos.keys())

    # --- Sidebar for Date Selection only ---
    with st.sidebar:
        st.header("Global Date Selection")

        # 1) Automatically pick first ISO (or None if list is empty)
        if not long_term_iso_options:
            st.warning("No Long-Term ISO configurations found.")
            selected_iso_name = None
        else:
            # Automatically choose the first ISO from the list—no selectbox
            selected_iso_name = long_term_iso_options[0]
            #st.info(f"Using default ISO: {selected_iso_name}")

        # 2) Global date range from all iso_data_dict
        global_min, global_max = get_global_date_range(iso_data_dict)
        if global_min is None or global_max is None:
            st.error("No valid ISO data available to set date range.")
            start_date, end_date = None, None
        else:
            default_start = global_max - pd.Timedelta(days=30)
            if default_start < global_min:
                default_start = global_min
            start_date, end_date = st.slider(
                "Select Date Range",
                min_value=global_min,
                max_value=global_max,
                value=(default_start, global_max),
                format="YYYY-MM-DD"
            )
    # --- End Sidebar ---

    # Create the tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Comparison (Table)", "Single ISO Analysis", "Weather Data Analysis", "Long Term Forecast"]
    )

    # Tab 1
    with tab1:
        if selected_iso_name and start_date and end_date:
            render_comparison_tab(iso_data_dict, start_date, end_date)
        elif not long_term_iso_options:
            st.warning("No Long-Term ISO configurations found to display in Comparison Tab.")
        elif not (start_date and end_date):
            st.warning("No valid date range available. Please check your data sources.")
        else:
            st.warning("Please ensure a valid date range is selected.")

    # Tab 2
    with tab2:
        render_iso_analysis_tab(iso_data_dict, start_date, end_date)

    # Tab 3
    with tab3:
        render_weather_tab(start_date, end_date)

    # Tab 4
    with tab4:
        # if you have a specific function for the long-term tab
        render_long_term_tab()  

if __name__ == "__main__":
    main()
