
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Import our own modules (assuming these are in the same directory or accessible)
from iso_data_integration import ISO_CONFIG, load_all_iso_data, ensure_uniform_hourly_index
from metrics_calculation import compute_iso_metrics
from rendering import load_data, get_global_date_range, render_comparison_tab, render_iso_analysis_tab, render_weather_tab

# -- Set page configuration
def _parse_actuals_line(line):
    """Parses a single line of actuals data."""
    parts = re.split(r'\s+', line.strip())
    if len(parts) >= 2:
        try:
            year = int(parts[0])
            peak = int(parts[1].replace(',', ''))
            energy = int(parts[2].replace(',', '')) if len(parts) > 2 else None
            return year, peak, energy
        except ValueError:
            st.warning(f"Invalid actuals data line: {line}")
            return None, None, None
    return None, None, None

def _parse_forecasts_text(forecasts_text):
    """Parses the forecasts text, extracting peak and energy data."""
    forecast_series = {}
    forecast_series_energy = {}
    current_forecast_year = None

    for line in forecasts_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"Forecast (\d{4})", line, re.IGNORECASE)
        if match:
            current_forecast_year = int(match.group(1))
            forecast_series[current_forecast_year] = []
            forecast_series_energy[current_forecast_year] = []
            continue

        if current_forecast_year:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    year = int(parts[0])
                    # Handle potential commas and decimals in peak and energy
                    peak = int(float(parts[1].replace(",", "")))
                    energy = int(float(parts[2].replace(",", "")))

                    forecast_series[current_forecast_year].append((year, peak))
                    forecast_series_energy[current_forecast_year].append((year, energy))
                except ValueError:
                    st.warning(f"Skipping invalid forecast data: {line}")

    return forecast_series, forecast_series_energy



def load_data_long_term():
    """Loads and parses the long-term data (actuals and forecasts)."""
    actuals_text = """
2006	62,203		305
2007	62,115		307
2008	62,103		311
2009	63,407		307
2010	65,713		318
2011	68,318		334
2012	66,558		325
2013	67,253		332
2014	66,464		340
2015	69,620		347
2016	71,093		351
2017	69,496		357
2018	73,308		376
2019	74,666		385
2020	74,328		382
2021	73,651		393
2022	80038		419
2023	85464		444
"""

    forecasts_text = """
Forecast 2024:
Year	Summer Peak Demand (MW)	Energy (TWh)
2024	 86,017	497
2025	 90,472	547
2026	106,405	653
2027	121,140	791
2028	 137,319	932
2029	 140,872	1006
2030    147,977	1058
2031	 149,758	1079
2032	 151,510	1093
2033	 153,229	1101

forecast 2023 : Peak Demand and Energy Forecast Summary
Year Summer Peak
Demand (MW)
Energy (TWh)
2023 82,308 445
2024 84,325 465
2025 85,740 480
2026 87,131 494
2027 88,518 508
2028 89,090 516
2029 89,624 521
2030 90,120 527
2031 90,563 532
2032 90,978 539

Appendix A
forecast 2022 :Peak Demand and Energy Forecast Summary
Year Summer Peak
Demand (MW)
Energy (TWh)
2022 77,733 423
2023 79,329 440
2024 80,554 452
2025 81,581 460
2026 82,606 470
2027 83,398 477
2028 84,146 485
2029 84,878 490
2030 85,569 496
2031 86,233 502

forecast
Appendix A
Forecast 2021Peak Demand and Energy Forecast Summary
Year Summer Peak
Demand (MW)
Energy (TWh)
2021 77,244 406
2022 78,855 420
2023 80,280 434
2024 81,267 444
2025 82,058 451
2026 82,838 458
2027 83,616 465
2028 84,362 473
2029 85,095 479
2030 85,820 485

forecast 2020
Peak Demand and Energy Forecast Summary
Year Summer Peak
Demand (MW)
Energy (TWh)
2020 76,696 401
2021 78,299 411
2022 80,108 425
2023 81,593 438
2024 82,982 450
2025 84,193 458
2026 85,384 467
2027 86,546 476
2028 87,668 486
2029 88,751 493

forecast 2019:
Appendix A
Peak Demand and Energy Forecast Summary
Year
Summer
Peak
Demand
(MW)
Energy
(TWh)
2019 74,853 384
2020 76,845 401
2021 78,824 413
2022 80,455 426
2023 82,101 438
2024 83,716 450
2025 85,327 461
2026 86,940 473
2027 88,508 484
2028 90,021 496

Forecast 2018
Appendix A
Peak Demand and Energy Forecast Summary
Year
Summer
Peak
Demand
(MW)
Energy
(TWh)
2018 72,974 371
2019 74,639 383
2020 75,879 393
2021 77,125 401
2022 78,556 411
2023 79,959 420
2024 81,200 429
2025 82,376 437
2026 83,704 446
2027 85,035 455

Forecast 2017 Appendix A
Peak Demand and Energy Forecast Summary
Year
Summer
Peak
Demand
(MW)
Energy
(TWh)
2017 72,934 356
2018 74,149 362
2019 75,588 371
2020 76,510 376
2021 77,417 380
2022 78,377 385
2023 79,348 389
2024 80,315 393
2025 81,261 398
2026 82,286 417

Forecast 2016:
Appendix A
Peak Demand and Energy Forecast Summary
Year
Summer
Peak
Demand
(MW)
Energy
(TWh)
2016 70,588 350.6
2017 71,416 356.2
2018 72,277 362.3
2019 73,663 370.7
2020 74,288 376.1
2021 74,966 380.4
2022 75,660 384.6
2023 76,350 388.9
2024 77,036 393.4
2025 77,732 397.7

Forecast 2014
Appendix A
Peak Demand and Energy Forecast Summary
Year
Summer
Peak
Demand
(MW)
Energy
(TWh)
2014 68,096 336.3
2015 69,057 342.9
2016 70,014 349.4
2017 70,871 355.9
2018 71,806 362.3
2019 72,859 368.7
2020 73,784 375.0
2021 74,710 381.4
2022 75,631 387.7
2023 76,550 394.0
2024 77,471 400.2
"""
    actuals_peak = {}
    actuals_energy = {}
    for line in actuals_text.strip().splitlines():
        year, peak, energy = _parse_actuals_line(line)
        if year is not None:
            actuals_peak[year] = peak
            if energy is not None:
                actuals_energy[year] = energy

    forecast_series, forecast_series_energy = _parse_forecasts_text(forecasts_text)
    return actuals_peak, actuals_energy, forecast_series, forecast_series_energy


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
    """Calculates and displays MAPE for long-term forecasts."""

    horizon_errors_peak = defaultdict(list)
    horizon_errors_energy = defaultdict(list)

    for rel_year, forecasts in forecast_series.items():
        for target_year, forecast_peak in forecasts:
            if target_year in actuals_peak:
                horizon = target_year - rel_year
                if horizon > 5:
                    continue
                error = abs(forecast_peak - actuals_peak[target_year]) / actuals_peak[target_year] * 100
                horizon_errors_peak[horizon].append(error)

    for rel_year, forecasts in forecast_series_energy.items():
        for target_year, forecast_energy in forecasts:
            if target_year in actuals_energy:
                horizon = target_year - rel_year
                if horizon > 5:
                    continue
                error = abs(forecast_energy - actuals_energy[target_year]) / actuals_energy[target_year] * 100
                horizon_errors_energy[horizon].append(error)

    # Compute and display MAPE
    mape_data = []
    for h in range(1, 6):
        peak_mape = np.mean(horizon_errors_peak[h]) if h in horizon_errors_peak else np.nan
        energy_mape = np.mean(horizon_errors_energy[h]) if h in horizon_errors_energy else np.nan
        mape_data.append({"Forecast Horizon (years)": h, "MAPE Peak Demand (%)": peak_mape, "MAPE Energy (%)": energy_mape})

    mape_df = pd.DataFrame(mape_data)
    st.dataframe(mape_df)

    mape_peak_all = np.nanmean([item for sublist in horizon_errors_peak.values() for item in sublist])  #Calculate overall MAPE
    mape_energy_all = np.nanmean([item for sublist in horizon_errors_energy.values() for item in sublist])

    st.write(f"Average MAPE for Peak Demand (all horizons): {mape_peak_all:.2f}%" if not np.isnan(mape_peak_all) else "Average MAPE for Peak Demand: N/A")
    st.write(f"Average MAPE for Energy (all horizons): {mape_energy_all:.2f}%" if not np.isnan(mape_energy_all) else "Average MAPE for Energy: N/A")

def render_long_term_tab():
    """Renders the long-term forecast analysis tab."""
    st.title('ERCOT Long-Term Load Forecasting Analysis')

    # Load data
    actuals_peak, actuals_energy, forecast_series, forecast_series_energy = load_data_long_term()

    # --- Handle empty forecast_series ---
    if forecast_series:
        all_forecasts = sorted(forecast_series.keys())
        default_forecasts = all_forecasts  # Select all by default if available
    else:
        all_forecasts = ["No forecasts available"]  # Provide a placeholder
        default_forecasts = []  # Don't select anything by default
        st.warning("No forecast data could be loaded.")


    # --- Sidebar moved INSIDE the tab ---
    with st.expander('Plot Options'):  # Options within the tab's sidebar
        plot_type = st.selectbox('Select Plot Type', ['Peak Demand', 'Energy'])
        # Use the conditional default
        selected_forecasts = st.multiselect('Select Forecasts to Display', all_forecasts, default=default_forecasts)


    # --- Handle the "No forecasts available" selection ---
    if "No forecasts available" in selected_forecasts:
        st.write("Please check the forecast data source.")
        return  # Exit early, don't try to plot

    # Display plot and MAPE (only if valid forecasts are selected)
    plot_long_term_forecasts(actuals_peak, actuals_energy, forecast_series, forecast_series_energy, selected_forecasts, plot_type)
    st.header("MAPE (Mean Absolute Percentage Error)")
    calculate_mape_long_term(actuals_peak, actuals_energy, forecast_series, forecast_series_energy)