import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os
import json # <--- ADD THIS LINE
from long_term_forecast_data import load_data_long_term_ercot # Commented out during debugging if not needed.
from functions import load_config
# Import our own modules (assuming these are in the same directory or accessible)
from iso_data_integration import load_all_iso_data, ensure_uniform_hourly_index
from metrics_calculation import compute_iso_metrics
from rendering import load_data, get_global_date_range, render_comparison_tab, render_iso_analysis_tab



def plot_long_term_forecasts(actuals_peak, actuals_energy, forecast_series, forecast_series_energy, selected_forecasts, plot_type):
    """Generates and displays the long-term forecast plots."""

    fig, ax = plt.subplots(figsize=(12, 6))  # Create figure and axes explicitly

    if plot_type == 'Peak Demand':
        # Plot actual summer peak
        actual_years = sorted(actuals_peak.keys())
        actual_peaks = [actuals_peak[yr] for yr in actual_years]
        ax.plot(actual_years, actual_peaks, color='black', marker='o', linewidth=2, label='Actual')
        ax.set_ylabel('Summer Peak Demand (MW)')
        ax.set_title('ERCOT Actual Summer Peak Demand and Yearly Forecasts')
    elif plot_type == 'Energy':
        # Plot actual total energy
        years_energy = sorted(actuals_energy.keys())
        energies = [actuals_energy[yr] for yr in years_energy]
        ax.plot(years_energy, energies, color='black', marker='o', linewidth=2, label='Actual Energy')
        ax.set_ylabel('Energy (TWh)')
        ax.set_title('ERCOT Actual Total Energy Demand (TWh) and Forecasts')
    else:
        st.error("Invalid plot_type.  Must be 'Peak Demand' or 'Energy'.")
        return

    # Colormap for forecast series
    cmap = plt.get_cmap('viridis')
    all_release_years = sorted(forecast_series.keys() if plot_type == 'Peak Demand' else forecast_series_energy.keys())
    if not all_release_years:
        st.warning("No forecast data available for the selected plot type.")
        st.pyplot(fig)  # Display the empty plot
        return

    norm = plt.Normalize(min(all_release_years), max(all_release_years))

    # Plot selected forecast series
    for rel_year in selected_forecasts:
        if plot_type == 'Peak Demand':
            if rel_year in forecast_series:
                series = sorted(forecast_series[rel_year], key=lambda x: x[0])
                if series: #Check if the series is not empty
                    f_years, f_peaks = zip(*series)  # Unzip the series into separate lists
                    ax.plot(f_years, f_peaks, color=cmap(norm(rel_year)),
                            linestyle='--', linewidth=1.5, alpha=0.7,
                            label=f'Forecast {rel_year}')
                else:
                    st.warning(f"No data available for forecast year {rel_year} and plot type {plot_type}.")
        elif plot_type == 'Energy':
            if rel_year in forecast_series_energy:
                series = sorted(forecast_series_energy[rel_year], key=lambda x: x[0])
                if series:  # Check if series is not empty
                  f_years, f_energies = zip(*series) # Unzip
                  ax.plot(f_years, f_energies, color=cmap(norm(rel_year)),
                          linestyle='--', linewidth=1.5, alpha=0.7,
                          label=f'Forecast Energy {rel_year}')
                else:
                   st.warning(f"No data available for forecast year {rel_year} and plot type {plot_type}.")

    ax.set_xlabel('Year')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True)
    fig.tight_layout()  # Use fig.tight_layout()
    st.pyplot(fig)  # Pass the figure to st.pyplot
    
def calculate_mape_long_term(actuals_peak, actuals_energy, forecast_series, forecast_series_energy):
    """Calculates and returns MAPE for long-term forecasts."""

    horizon_errors_peak = defaultdict(list)
    horizon_errors_energy = defaultdict(list)

    max_actual_peak_year = max(actuals_peak.keys()) if actuals_peak else 0
    max_actual_energy_year = max(actuals_energy.keys()) if actuals_energy else 0

    for rel_year, forecasts in forecast_series.items():
        for target_year, forecast_peak in forecasts:
            if target_year in actuals_peak and target_year <= max_actual_peak_year:
                horizon = target_year - rel_year
                if horizon > 5:
                    continue
                error = abs(forecast_peak - actuals_peak[target_year]) / actuals_peak[target_year] * 100
                horizon_errors_peak[horizon].append(error)

    for rel_year, forecasts in forecast_series_energy.items():
        for target_year, forecast_energy in forecasts:
            if target_year in actuals_energy and target_year <= max_actual_energy_year:
                horizon = target_year - rel_year
                if horizon > 5:
                    continue
                error = abs(forecast_energy - actuals_energy[target_year]) / actuals_energy[target_year] * 100
                horizon_errors_energy[horizon].append(error)

    mape_peak_all = np.nanmean([item for sublist in horizon_errors_peak.values() for item in sublist]) if horizon_errors_peak else np.nan
    mape_energy_all = np.nanmean([item for sublist in horizon_errors_energy.values() for item in sublist]) if horizon_errors_energy else np.nan

    return {"Peak Demand MAPE (%)": mape_peak_all, "Energy MAPE (%)": mape_energy_all} # Return a dictionary




def render_long_term_tab():
    """Renders the long-term forecast analysis tab, allowing ISO selection."""
    st.title('Long-Term Forecasting Analysis')

    # --- Find all ISOs with timeframe="long" ---
    long_term_isos = {}  # Dictionary to store ISO names and their configs
    ISO_CONFIG=load_config() # Load config here, inside the function!
    if ISO_CONFIG is None: # Handle case where load_config fails (returns None)
        return

    for iso_name, config in ISO_CONFIG.items():
        if config.get("timeframe") == "long":
            long_term_isos[iso_name] = config

    if not long_term_isos:
        st.warning("No ISO configurations found with timeframe='long'.")
        return

    # --- User selection of ISO ---
    selected_iso_name = st.selectbox(
        "Select ISO for Long-Term Analysis",
        options=list(long_term_isos.keys()), # Provide ISO names as options
        index=0 # Optionally set a default selection (e.g., the first one)
    )

    selected_config = long_term_isos[selected_iso_name] # Get the config for the selected ISO

    # Update tab title to include selected ISO
    st.subheader(f"Long-Term Forecasts for {selected_iso_name}")


    # --- Dynamically Load Data based on Selected Config ---
    function_name = selected_config.get("function")
    if function_name:
        try:
            data_loading_function = globals()[function_name]
            actuals_peak, actuals_energy, forecast_series, forecast_series_energy = data_loading_function() #  Uncomment if load_data_long_term_ercot is causing issues
            # actuals_peak, actuals_energy, forecast_series, forecast_series_energy = {}, {}, {}, {} # Initialize empty dicts for debugging
            st.write("Data loading function called.") # Indicate function call
            print("Loaded actuals_peak:", actuals_peak) # Debug print
            print("Loaded actuals_energy:", actuals_energy) # Debug print
            print("Loaded forecast_series:", forecast_series) # Debug print
            print("Loaded forecast_series_energy:", forecast_series_energy) # Debug print

        except KeyError:
            st.error(f"Error: Function '{function_name}' not found.")
            return
        except Exception as e:
            st.error(f"Error executing function '{function_name}': {e}")
            return
    else:
        st.warning(f"No 'function' specified in configuration for {selected_iso_name}.")
        return


    # --- Handle empty forecast_series (same as before) ---
    if forecast_series:
        all_forecasts = sorted(forecast_series.keys())
        default_forecasts = all_forecasts
    else:
        all_forecasts = ["No forecasts available"]
        default_forecasts = []
        st.warning("No forecast data could be loaded.")


    # --- Sidebar moved INSIDE the tab (same as before) ---
    with st.expander('Plot Options'):
        plot_type = st.selectbox('Select Plot Type', ['Peak Demand', 'Energy'])
        selected_forecasts = st.multiselect('Select Forecasts to Display', all_forecasts, default=default_forecasts)

    # --- Handle "No forecasts available" (same as before) ---
    if "No forecasts available" in selected_forecasts:
        st.write("Please check the forecast data source.")
        return

    # --- Display plot and MAPE (same as before) ---
    plot_long_term_forecasts(actuals_peak, actuals_energy, forecast_series, forecast_series_energy, selected_forecasts, plot_type)
    st.header("MAPE (Mean Absolute Percentage Error)")
    calculate_mape_long_term(actuals_peak, actuals_energy, forecast_series, forecast_series_energy)
    mape_values = calculate_mape_long_term(actuals_peak, actuals_energy, forecast_series, forecast_series_energy)
    st.write(mape_values)




def main():
    """Main function to run the Streamlit app and for debugging."""
    print("Running main function...") # Debug start message

    config = load_config("config.json")
    print("Config loaded:", config) # Print config after loading

    if config: # Only proceed if config is loaded successfully
        render_long_term_tab() # Call the function to render the tab
    else:
        print("Config loading failed, exiting main function.") # Debug message if config fails

    print("Main function finished.") # Debug end message


if __name__ == "__main__":
    main()