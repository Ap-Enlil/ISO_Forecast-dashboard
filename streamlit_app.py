
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

    # Load ISO data (for tabs 1-3)
    iso_data_dict = load_data()

    # Global date range (for tabs 1-3)
    global_min, global_max = get_global_date_range(iso_data_dict)
    if global_min is None or global_max is None:
        st.error("No valid ISO data available.")
        return

    # --- Sidebar for Global Date Selection (for Tabs 1-3) ---
    with st.sidebar:
        st.header("Global Date Selection")
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

    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Comparison (Table)", "Single ISO Analysis", "Weather Data Analysis", "Long Term Forecast"])

    with tab1:
        render_comparison_tab(iso_data_dict, start_date, end_date)
    with tab2:
        render_iso_analysis_tab(iso_data_dict, start_date, end_date)
    with tab3:
        #render_weather_tab()
    with tab4:
        render_long_term_tab()  # Long-term tab, with its OWN sidebar


if __name__ == "__main__":
    main()