import streamlit as st


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.colors import LinearSegmentedColormap

# Import our own modules
from iso_data_integration import ISO_CONFIG, load_all_iso_data, ensure_uniform_hourly_index
from metrics_calculation import compute_iso_metrics
from rendering import load_data, get_global_date_range, render_comparison_tab, render_iso_analysis_tab, render_weather_tab

# --
# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="ISO Forecast Comparison", layout="wide")

# ------------------------------
# Custom CSS for a Modern Look
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
    st.title("ISO Load Forecast – Comparison & Analysis")
    st.markdown("Welcome to the ISO forecast analysis tool. Use the sidebar to select the date range to explore global as well as ISO-specific forecast performance.")

    # Load ISO data
    iso_data_dict = load_data()

    # Get the global date range from the loaded data
    global_min, global_max = get_global_date_range(iso_data_dict)
    if global_min is None or global_max is None:
        st.error("No valid ISO data available.")
        return

    # ------------------------------
    # Global Date Range Selector in Sidebar
    # ------------------------------
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

    # ------------------------------
    # Create Tabs and Render Content
    # ------------------------------
    tab1, tab2, tab3 = st.tabs(["Comparison (Mega Table)", "Single ISO Analysis", "Weather Data Analysis"])
    with tab1:
        render_comparison_tab(iso_data_dict, start_date, end_date)
    with tab2:
        render_iso_analysis_tab(iso_data_dict, start_date, end_date)
    with tab3:
        render_weather_tab()

if __name__ == "__main__":
    main()