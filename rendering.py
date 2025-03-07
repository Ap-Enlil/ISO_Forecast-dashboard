import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.colors import LinearSegmentedColormap
from functions import  load_config, calculate_mape_long_term
from long_term_forecast_data import load_data_long_term_ercot
# Import your own modules
from iso_data_integration2 import ISO_CONFIG, load_all_iso_data, ensure_uniform_hourly_index
from metrics_calculation import compute_iso_metrics
from collections import defaultdict

# ------------------------------
# Helper: Column Color Mapping (if needed)
# ------------------------------
def apply_colormap(col):
    """Apply a column-wise background color gradient."""
    cmap = LinearSegmentedColormap.from_list("RedGreenBlue", ["white", "white", "white"])
    min_val, max_val = col.min(), col.max()
    if max_val == min_val:
        return ["background-color: rgba(255, 255, 255, 0.3); font-weight: bold"] * len(col)

    def color_mapping(x):
        normalized_value = (x - min_val) / (max_val - min_val)
        r, g, b, _ = cmap(normalized_value)
        return f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.3); font-weight: bold"

    return col.map(color_mapping)

# ------------------------------
# Cache the data load (24-hour TTL)
# ------------------------------
@st.cache_data(ttl=24 * 60 * 60)
def load_data():
    return load_all_iso_data()

# ------------------------------
# Helper: Get Global Date Range from ISO Data
# ------------------------------
def get_global_date_range(iso_data_dict):
    valid_dates = []
    for df in iso_data_dict.values():
        if df is not None and not df.empty:
            valid_dates.append(df.index.min())
            valid_dates.append(df.index.max())
    if not valid_dates:
        return None, None
    global_min = min(valid_dates).date()
    global_max = max(valid_dates).date()
    return global_min, global_max

# ------------------------------
# Tab 1: Comparison of All ISOs
# ------------------------------
# ------------------------------
# Tab 1: Comparison of All ISOs
# ------------------------------
def render_comparison_tab(iso_data_dict, start_date, end_date):
    """
    Renders a comparison table for all short-term ISOs with only the key performance metrics 
    (MAPE and Avg Error) plus configuration characteristics.
    """
    import numpy as np
    import pandas as pd
    import streamlit as st
    from functions import load_config
    from iso_data_integration2 import ensure_uniform_hourly_index
    from metrics_calculation import compute_iso_metrics

    st.subheader("ISO Comparison – Performance & Characteristics")
    st.info(f"Showing data from **{start_date}** to **{end_date}**.")

    # Load configuration and filter for short-term ISOs.
    ISO_CONFIG = load_config()
    if ISO_CONFIG is None:
        st.error("Could not load ISO configuration.")
        return

    # Filter only short-term ISOs.
    short_term_data = {
        iso: df for iso, df in iso_data_dict.items()
        if iso in ISO_CONFIG and ISO_CONFIG[iso].get("timeframe") != "long"
    }

    overall_metrics = {}

    for iso, df in short_term_data.items():
        if df is None or df.empty:
            overall_metrics[iso] = {"MAPE (%)": np.nan, "Avg Error (MW)": np.nan}
            continue

        # Filter by date range and standardize index.
        df_filt = df.loc[str(start_date):str(end_date)].copy()
        df_filt = ensure_uniform_hourly_index(df_filt, iso)

        # Compute overall metrics.
        metrics = compute_iso_metrics(df_filt)
        overall_metrics[iso] = metrics

    # Build a summary DataFrame with only MAPE and Avg Error.
    df_summary = pd.DataFrame(overall_metrics).T[['MAPE (%)', 'Avg Error (MW)']]

    # Add ISO configuration characteristics (from JSON) to each row.
    # Characteristics to add: forecast_method, timeframe, type_of_forecast, update_frequency, time_step, timezone
    for iso in df_summary.index:
        config = ISO_CONFIG.get(iso, {})
        df_summary.loc[iso, 'forecast_method'] = config.get("forecast_method")
        df_summary.loc[iso, 'timeframe'] = config.get("timeframe")
        df_summary.loc[iso, 'ISO'] = config.get("ISO")
        df_summary.loc[iso, 'type_of_forecast'] = config.get("type_of_forecast")
        df_summary.loc[iso, 'update_frequency'] = config.get("update_frequency")
        df_summary.loc[iso, 'time_step'] = config.get("time_step")
        df_summary.loc[iso, 'timezone'] = config.get("timezone")

    # (Optional) Allow the user to toggle these extra columns.
    show_chars = st.checkbox("Show ISO characteristics", value=True)
    if not show_chars:
        df_summary = df_summary[['MAPE (%)', 'Avg Error (MW)']]

    # Display the summary table.
    # If you want to allow inline editing, you can use st.data_editor (Streamlit version 1.18+)
    st.data_editor(df_summary, num_rows="dynamic", use_container_width=True)

    return df_summary


# ------------------------------
# Tab 2: Single ISO Detailed Analysis
# ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress  # For regression analysis

# Import your own modules
from iso_data_integration2 import ISO_CONFIG, load_all_iso_data, ensure_uniform_hourly_index, add_price_data_to_existing_df
from metrics_calculation import compute_iso_metrics

# ------------------------------
# Tab 2: Single ISO Detailed Analysis
# ------------------------------
# ------------------------------
# Tab 2: Single ISO Detailed Analysis
# ------------------------------
def render_iso_analysis_tab(iso_data_dict, start_date, end_date):
    st.subheader("Single ISO Detailed Analysis")
    st.info(f"Using global date range: **{start_date}** to **{end_date}**.")

    # Select the ISO from the configuration keys
    iso_list = list(ISO_CONFIG.keys())
    selected_iso = st.selectbox("Select ISO for Analysis", options=iso_list)

    df_selected = iso_data_dict.get(selected_iso)
    if df_selected is None or df_selected.empty:
        st.error(f"No data available for {selected_iso}.")
        return

    # Filter data for the selected date range and ensure a uniform index
    df_range = df_selected.loc[str(start_date):str(end_date)].copy()
    if df_range.empty:
        st.error(f"No data available for {selected_iso} in the selected date range.")
        return
    df = ensure_uniform_hourly_index(df_range, selected_iso)

    # Compute metrics for the selected ISO and date range
    iso_metrics = compute_iso_metrics(df)

    # Display metrics table at the top of Tab 2
    st.markdown("### Performance Metrics")
    metrics_df = pd.DataFrame([iso_metrics]).T.rename(columns={0: "Value"})
    st.dataframe(metrics_df)

    # Add price data to the DataFrame (if available)
    df = add_price_data_to_existing_df(df, selected_iso, target_column="LMP Difference (USD)")

    # Compute a moving average for the forecast error (30-day window, 24 hours each day)
    if 'Forecast Error (MW)' in df.columns:
        df['Error_MA_30D'] = df['Forecast Error (MW)'].rolling(window=24 * 30, min_periods=1).mean()

    # ------------------------------
    # Plot 1: Load vs Forecast and Forecast Error
    # ------------------------------
    fig1 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Load vs Forecast", "Forecast Error vs Time")
    )

    # Subplot 1: Actual Load vs. Forecast
    if 'TOTAL Actual Load (MW)' in df.columns and 'SystemTotal Forecast Load (MW)' in df.columns:
        y_min = min(df['TOTAL Actual Load (MW)'].min(), df['SystemTotal Forecast Load (MW)'].min())

        # Actual Load
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TOTAL Actual Load (MW)'],
                name='Actual Load',
                mode='lines',
                line=dict(color='rgba(0,100,80,0.8)'),
                connectgaps=True
            ),
            row=1, col=1
        )

        # Fill area under actual load
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=[y_min] * len(df),
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                connectgaps=True,
                showlegend=False
            ),
            row=1, col=1
        )

        # Forecast
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SystemTotal Forecast Load (MW)'],
                name='Forecast',
                mode='lines',
                line=dict(color='rgba(0,0,255,0.8)'),
                connectgaps=True
            ),
            row=1, col=1
        )
        fig1.update_yaxes(title_text="Load (MW)", range=[y_min, None], row=1, col=1)
    else:
        st.warning("Missing columns for Actual Load or Forecast. Skipping first plot.")

    # Subplot 2: Forecast Error vs. Time
    if 'Forecast Error (MW)' in df.columns:
        # Separate positive (Overforecast) and negative (Underforecast) errors
        df_overforecast = df['Forecast Error (MW)'].clip(lower=0)   # error > 0
        df_underforecast = df['Forecast Error (MW)'].clip(upper=0)  # error < 0

        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df_overforecast,
                fill='tozeroy',
                name='Model Overforecast (MW)',
                fillcolor='rgba(0,0,255,0.2)',
                line=dict(color='rgba(0,0,255,0)'),
                connectgaps=True
            ),
            row=2, col=1
        )
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df_underforecast,
                fill='tozeroy',
                name='Model Underforecast (MW)',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                connectgaps=True
            ),
            row=2, col=1
        )
        fig1.update_yaxes(title_text="Forecast Error (MW)", row=2, col=1)
    else:
        st.warning("No 'Forecast Error (MW)' column to plot error. Skipping second part of Plot 1.")

    fig1.update_xaxes(title_text="Date", row=2, col=1)
    fig1.update_layout(height=800, hovermode='x unified', showlegend=True)
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------------------
    # Plot 2: MAPE vs Time with a threshold line (4%) and a 30-day moving average
    # ------------------------------
    if 'TOTAL Actual Load (MW)' in df.columns and 'SystemTotal Forecast Load (MW)' in df.columns:
        df['APE'] = (
            np.abs(df['TOTAL Actual Load (MW)'] - df['SystemTotal Forecast Load (MW)']) 
            / df['TOTAL Actual Load (MW)'] * 100
        )
        df['APE_MA'] = df['APE'].rolling(window=24 * 30, min_periods=1).mean()

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df.index,
                y=df['APE'],
                mode='lines',
                name='APE (%)',
                line=dict(color='orange')
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=df.index,
                y=df['APE_MA'],
                mode='lines',
                name='30-Day MA APE',
                line=dict(color='grey', width=1)
            )
        )
        # Threshold line at 4%
        fig2.add_shape(
            type="line",
            xref="paper", yref="y",
            x0=0, x1=1, y0=4, y1=4,
            line=dict(color="red", width=2, dash="dash")
        )
        fig2.update_layout(
            title="MAPE vs Time (Threshold = 4%)",
            xaxis_title="Date",
            yaxis_title="APE (%)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Required columns for APE calculation not found.")

    # ------------------------------
    # Day-Type Filter for Bar Plots
    # ------------------------------
    day_filter = st.radio("Select Day Type for Bar Plots", options=["Weekdays", "Weekends", "Both"], index=2)
    if day_filter == "Weekdays":
        df_filtered = df[df.index.dayofweek < 5]
    elif day_filter == "Weekends":
        df_filtered = df[df.index.dayofweek >= 5]
    else:
        df_filtered = df.copy()

    # ------------------------------
    # Plot 5: Bar Plot – Total MW Over/Under Forecasted by Hour
    # ------------------------------
    if 'Forecast Error (MW)' in df_filtered.columns:
        # Overforecast (error > 0): store the actual positive value
        df_filtered['Overforecast_MW'] = np.where(
            df_filtered['Forecast Error (MW)'] > 0,
            df_filtered['Forecast Error (MW)'],
            0
        )
        # Underforecast (error < 0): store the positive magnitude of the negative error
        df_filtered['Underforecast_MW'] = np.where(
            df_filtered['Forecast Error (MW)'] < 0,
            -df_filtered['Forecast Error (MW)'],
            0
        )

        grouped_mw = df_filtered.groupby(df_filtered.index.hour).agg({
            'Overforecast_MW': 'sum',
            'Underforecast_MW': 'sum'
        }).reset_index()

        fig3 = go.Figure()
        hour_col = grouped_mw.columns[0]
        fig3.add_trace(go.Bar(
            x=grouped_mw[hour_col],
            y=grouped_mw['Overforecast_MW'],
            name='Model Overforecast (MW)',
            marker_color='blue'
        ))
        fig3.add_trace(go.Bar(
            x=grouped_mw[hour_col],
            y=grouped_mw['Underforecast_MW'],
            name='Model Underforecast (MW)',
            marker_color='red'
        ))
        fig3.update_layout(
            title="Total MW Over/Under Forecasted by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Total MW",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Forecast Error data not available for the MW bar plot.")

    # ------------------------------
    # Plot 6: Bar Plot – Count of Over/Under Forecast Occurrences by Hour
    # ------------------------------
    if 'Forecast Error (MW)' in df_filtered.columns:
        # Overforecast: error > 0
        df_filtered['Overforecast_Count'] = np.where(
            df_filtered['Forecast Error (MW)'] > 0, 
            1, 
            0
        )
        # Underforecast: error < 0
        df_filtered['Underforecast_Count'] = np.where(
            df_filtered['Forecast Error (MW)'] < 0, 
            1, 
            0
        )

        grouped_count = df_filtered.groupby(df_filtered.index.hour).agg({
            'Overforecast_Count': 'sum',
            'Underforecast_Count': 'sum'
        }).reset_index()

        fig4 = go.Figure()
        hour_col = grouped_count.columns[0]
        fig4.add_trace(go.Bar(
            x=grouped_count[hour_col],
            y=grouped_count['Overforecast_Count'],
            name='Model Overforecast Count',
            marker_color='blue'
        ))
        fig4.add_trace(go.Bar(
            x=grouped_count[hour_col],
            y=grouped_count['Underforecast_Count'],
            name='Model Underforecast Count',
            marker_color='red'
        ))
        fig4.update_layout(
            title="Count of Over/Under Forecast Occurrences by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Count",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Forecast Error data not available for the count bar plot.")

    # ------------------------------
    # Plot 3: Average Price Difference vs Binned % Error (Weekdays)
    # ------------------------------
    if "LMP Difference (USD)" in df_filtered.columns and "Forecast Error (MW)" in df_filtered.columns:
        # Filter further for weekdays only
        weekday_data = df_filtered[df_filtered.index.dayofweek < 5].copy()

        # Remove NaN values
        valid_data = weekday_data[['LMP Difference (USD)', 'Forecast Error (MW)', 'TOTAL Actual Load (MW)']].dropna()

        if not valid_data.empty:
            # Calculate the percentage error
            valid_data['% Error'] = (
                valid_data['Forecast Error (MW)'] / valid_data['TOTAL Actual Load (MW)'] * 100
            )

            # Define bins for the % Error
            bins = [-np.inf, -4, -2, -1, 0, 1, 2, 4, np.inf]
            bin_labels = ["<-4%", "-4% to -2%", "-2% to -1%", "-1% to 0%",
                          "0% to 1%", "1% to 2%", "2% to 4%", ">4%"]

            # Bin the % Error
            valid_data['Error Bin'] = pd.cut(
                valid_data['% Error'],
                bins=bins,
                labels=bin_labels,
                include_lowest=True,
                right=False
            )

            # Calculate the average price difference by bin
            avg_price_diff_by_bin = (
                valid_data.groupby('Error Bin')['LMP Difference (USD)']
                .mean()
                .reset_index()
            )

            # Bar plot: Average Price Difference vs. Binned % Error
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=avg_price_diff_by_bin['Error Bin'],
                y=avg_price_diff_by_bin['LMP Difference (USD)'],
                name='Average Price Difference',
                marker_color='green'
            ))
            fig5.update_layout(
                title="Average Price Difference vs Binned % Error (Weekdays)",
                xaxis_title="Binned % Error (Weekdays)",
                yaxis_title="Average Price Difference (USD)",
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("No valid weekday data for the price difference vs % error bar plot.")
    else:
        st.warning("Required data not available for the price difference vs % error bar plot.")
# ------------------------------
# Tab 3: Weather Data (Placeholder)
# ------------------------------
