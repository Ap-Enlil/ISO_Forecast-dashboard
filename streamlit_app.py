
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
from rendering import load_data, get_global_date_range, render_comparison_tab, render_iso_analysis_tab, render_weather_tab
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
# ------------------------------
# Import Rendering Functions from the Other Module
# ------------------------------
from rendering import load_data, get_global_date_range, render_comparison_tab, render_iso_analysis_tab, render_weather_tab
def main():
    st.title("Forecast Model Registry – Comparison & Analysis")
    st.markdown("Welcome to the forecast analysis tool. Use the sidebar to select the date range for global and ISO-specific forecast performance.")

    # Load ISO data (for tabs 1-3) - load ALL iso data first, for date range calculation
    iso_data_dict = load_data()

    # Load ISO Config to filter for long-term ISOs
    ISO_CONFIG = load_config()
    if ISO_CONFIG is None:
        st.error("Could not load ISO configuration.")
        return

    long_term_isos = {} # Dictionary to store long-term ISO names and their configs
    for iso_name, config in ISO_CONFIG.items():
        if config.get("timeframe") == "long":
            long_term_isos[iso_name] = config

    # --- Sidebar for ISO Selection and Global Date Selection (for Tabs 1-3) ---
    with st.sidebar:
        st.header("ISO and Date Selection")

        # --- ISO Selection (FILTERED for Long-Term ISOs) ---
        long_term_iso_options = list(long_term_isos.keys()) # Use filtered list
        if not long_term_iso_options:
            st.warning("No Long-Term ISO configurations found.")
            selected_iso_name = None # Handle case with no long-term ISOs
        else:
            selected_iso_name = st.selectbox(
                "Select Long-Term ISO for Comparison", # Updated label
                options=long_term_iso_options,
                index=0 if long_term_iso_options else 0 # Default to first if available
            )

        st.header("Global Date Selection") # Moved below ISO selection
        # Global date range (for tabs 1-3) - moved inside sidebar for clarity
        global_min, global_max = get_global_date_range(iso_data_dict) # Still use ALL iso_data for date range
        if global_min is None or global_max is None:
            st.error("No valid ISO data available to set date range.")
            start_date, end_date = None, None # Handle case with no date range
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


    tab1, tab2, tab3, tab4 = st.tabs(["Comparison (Table)", "Single ISO Analysis", "Weather Data Analysis", "Long Term Forecast"])

    with tab1:
        if selected_iso_name and start_date and end_date: # Check if selected_iso_name and dates are valid
            render_comparison_tab(iso_data_dict, start_date, end_date) # Pass selected_iso_name
        elif not long_term_iso_options: # Check for long_term_iso_options instead of iso_options
            st.warning("No Long-Term ISO configurations found to display in Comparison Tab.")
        elif not (start_date and end_date):
            st.warning("No valid date range available. Please check your data sources.")
        else:
            st.warning("Please select a Long-Term ISO and ensure valid date range is available.") # Updated warning

    with tab2:
        render_iso_analysis_tab(iso_data_dict, start_date, end_date) # Keep as is, for all ISOs
    with tab3:
        render_weather_tab() # Keep as is
    with tab4:
        render_long_term_tab()  # Long-term tab, with its OWN sidebar, keep as is

if __name__ == "__main__":
    main()