import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.colors import LinearSegmentedColormap

# Import your own modules
from iso_data_integration2 import ISO_CONFIG, load_all_iso_data, ensure_uniform_hourly_index
from metrics_calculation import compute_iso_metrics

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
def render_comparison_tab(iso_data_dict, start_date, end_date):
    st.subheader("Comparison of All ISOs – Model Performance Overview")
    st.info(f"Showing data from **{start_date}** to **{end_date}**.")

    overall_metrics = {}
    improvement_metrics = {}
    mape_by_hour = {}

    for iso_key, df in iso_data_dict.items():
        if df is not None and not df.empty:
            # Filter data by date and ensure a uniform hourly index
            df_filtered = df.loc[str(start_date):str(end_date)].copy()
            df_filtered = ensure_uniform_hourly_index(df_filtered, iso_key)

            # Compute overall metrics
            metrics = compute_iso_metrics(df_filtered)
            overall_metrics[iso_key] = metrics

            # Split data into first and second halves for improvement analysis
            date_range = pd.date_range(start=str(start_date), end=str(end_date), freq='H')
            if len(date_range) < 2:
                improvement_metrics[iso_key] = {
                    "MAPE (First Half)": np.nan,
                    "First Half Period": "N/A",
                    "MAPE (Second Half)": np.nan,
                    "Second Half Period": "N/A",
                    "Delta MAPE": np.nan
                }
            else:
                mid_date = date_range[int(len(date_range) / 2)]
                df_first = df_filtered.loc[str(start_date):str(mid_date)].copy()
                df_second = df_filtered.loc[str(mid_date):str(end_date)].copy()
                metrics_first = compute_iso_metrics(df_first)
                metrics_second = compute_iso_metrics(df_second)
                delta = metrics_second.get("MAPE (%)", np.nan) - metrics_first.get("MAPE (%)", np.nan)
                improvement_metrics[iso_key] = {
                    "MAPE (First Half)": metrics_first.get("MAPE (%)", np.nan),
                    "First Half Period": f"{start_date} to {mid_date.date()}",
                    "MAPE (Second Half)": metrics_second.get("MAPE (%)", np.nan),
                    "Second Half Period": f"{mid_date.date()} to {end_date}",
                    "Delta MAPE": delta
                }

            # Compute MAPE by hour if the required columns exist
            if ('TOTAL Actual Load (MW)' in df_filtered.columns and 
                'SystemTotal Forecast Load (MW)' in df_filtered.columns):
                df_filtered['APE'] = (np.abs(df_filtered['TOTAL Actual Load (MW)'] -
                                             df_filtered['SystemTotal Forecast Load (MW)'])
                                      / df_filtered['TOTAL Actual Load (MW)'] * 100)
                hourly_mape = df_filtered.groupby(df_filtered.index.hour)['APE'].mean()
                mape_by_hour[iso_key] = hourly_mape
            else:
                mape_by_hour[iso_key] = pd.Series([np.nan] * 24, index=range(24))
        else:
            overall_metrics[iso_key] = {"MAPE (%)": np.nan}
            improvement_metrics[iso_key] = {
                "MAPE (First Half)": np.nan,
                "First Half Period": "N/A",
                "MAPE (Second Half)": np.nan,
                "Second Half Period": "N/A",
                "Delta MAPE": np.nan
            }
            mape_by_hour[iso_key] = pd.Series([np.nan] * 24, index=range(24))

    # Combine overall and improvement metrics
    df_overall = pd.DataFrame(overall_metrics).T
    if "Avg APE (%)" in df_overall.columns:
        df_overall = df_overall.drop(columns=["Avg APE (%)"])
    df_improvement = pd.DataFrame(improvement_metrics).T
    df_summary = pd.concat([df_overall, df_improvement], axis=1)

    # Add an indicator if performance improved (Delta MAPE negative)
    df_summary["Improving"] = df_summary["Delta MAPE"].apply(
        lambda x: "Yes" if pd.notnull(x) and x < 0 else ("No" if pd.notnull(x) else "N/A")
    )

    st.markdown("### Summary Metrics")
    numeric_cols = df_summary.select_dtypes(include=[np.number]).columns
    fmt = {col: "{:.2f}" for col in numeric_cols}
    st.dataframe(df_summary.style.format(fmt))

    # Plot: Model Improvement (Delta MAPE) per ISO
    fig_delta = go.Figure(data=go.Bar(
        x=df_summary.index,
        y=df_summary["Delta MAPE"],
        marker_color=['green' if x < 0 else 'red' for x in df_summary["Delta MAPE"]]
    ))
    fig_delta.update_layout(
        title="Model Improvement (Δ MAPE: Second Half - First Half)",
        xaxis_title="ISO",
        yaxis_title="Δ MAPE (%)"
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # Plot: Average MAPE by Hour of Day for each ISO
    fig_hour = go.Figure()
    for iso, series in mape_by_hour.items():
        fig_hour.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines+markers',
            name=iso
        ))
    fig_hour.update_layout(
        title="Average MAPE by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average MAPE (%)",
        xaxis=dict(dtick=1)
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # Optional: Overall MAPE by ISO as a bar chart
    overall_mape = df_overall["MAPE (%)"]
    fig_overall = go.Figure(data=go.Bar(
        x=overall_mape.index,
        y=overall_mape.values,
        marker_color='blue'
    ))
    fig_overall.update_layout(
        title="Overall MAPE (%) by ISO",
        xaxis_title="ISO",
        yaxis_title="MAPE (%)"
    )
    st.plotly_chart(fig_overall, use_container_width=True)

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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress

# Assuming ISO_CONFIG, ensure_uniform_hourly_index, and add_price_data_to_existing_df are defined elsewhere

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

    # Add price data to the DataFrame (if available)
    df = add_price_data_to_existing_df(df, selected_iso, target_column="LMP Difference (USD)")

    # Compute a moving average for the forecast error
    if 'Forecast Error (MW)' in df.columns:
        df['Error_MA_30D'] = df['Forecast Error (MW)'].rolling(window=24 * 30, min_periods=1).mean()

    # ------------------------------
    # Plot 1: Load vs Forecast and Forecast Error
    # ------------------------------
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Load vs Forecast", "Forecast Error vs Time")
    )

    if 'TOTAL Actual Load (MW)' in df.columns and 'SystemTotal Forecast Load (MW)' in df.columns:
        y_min = min(df['TOTAL Actual Load (MW)'].min(), df['SystemTotal Forecast Load (MW)'].min())
        fig1.add_trace(
            go.Scatter(
                x=df.index, y=df['TOTAL Actual Load (MW)'],
                name='Actual Load',
                mode='lines',
                line=dict(color='rgba(0,100,80,0.8)'),
                connectgaps=True
            ),
            row=1, col=1
        )
        fig1.add_trace(
            go.Scatter(
                x=df.index, y=[y_min] * len(df),
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                connectgaps=True,
                showlegend=False
            ),
            row=1, col=1
        )
        fig1.add_trace(
            go.Scatter(
                x=df.index, y=df['SystemTotal Forecast Load (MW)'],
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

    if 'Forecast Error (MW)' in df.columns:
        df_positive = df['Forecast Error (MW)'].clip(lower=0)
        df_negative = df['Forecast Error (MW)'].clip(upper=0)
        fig1.add_trace(
            go.Scatter(
                x=df.index, y=df_positive,
                fill='tozeroy', name='Over-forecast',
                fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(0,0,255,0)'),
                connectgaps=True
            ),
            row=2, col=1
        )
        fig1.add_trace(
            go.Scatter(
                x=df.index, y=df_negative,
                fill='tozeroy', name='Under-forecast',
                fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,0,0,0)'),
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
    # Plot 2: MAPE vs Time with a threshold line and a 30-day moving average
    # ------------------------------
    if 'TOTAL Actual Load (MW)' in df.columns and 'SystemTotal Forecast Load (MW)' in df.columns:
        df['APE'] = np.abs(df['TOTAL Actual Load (MW)'] - df['SystemTotal Forecast Load (MW)']) / df['TOTAL Actual Load (MW)'] * 100
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

    # Day-Type Filter for upcoming bar plots
    day_filter = st.radio("Select Day Type for Bar Plots", options=["Weekdays", "Weekends", "Both"], index=2)
    if day_filter == "Weekdays":
        df_filtered = df[df.index.dayofweek < 5]
    elif day_filter == "Weekends":
        df_filtered = df[df.index.dayofweek >= 5]
    else:
        df_filtered = df.copy()

    
    # Plot 5 (former Plot 4): Bar Plot – Total MW Over/Under Forecasted by Hour
    if 'Forecast Error (MW)' in df_filtered.columns:
        df_filtered['Overforecast_MW'] = np.where(df_filtered['Forecast Error (MW)'] < 0, -df_filtered['Forecast Error (MW)'], 0)
        df_filtered['Underforecast_MW'] = np.where(df_filtered['Forecast Error (MW)'] > 0, df_filtered['Forecast Error (MW)'], 0)

        grouped_mw = df_filtered.groupby(df_filtered.index.hour).agg({
            'Overforecast_MW': 'sum',
            'Underforecast_MW': 'sum'
        }).reset_index()

        fig3 = go.Figure()
        hour_col = grouped_mw.columns[0]
        fig3.add_trace(go.Bar(
            x=grouped_mw[hour_col],
            y=grouped_mw['Overforecast_MW'],
            name='Overforecast (MW)',
            marker_color='blue'
        ))
        fig3.add_trace(go.Bar(
            x=grouped_mw[hour_col],
            y=grouped_mw['Underforecast_MW'],
            name='Underforecast (MW)',
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

    # Plot 6 (former Plot 5): Bar Plot – Count of Over/Under Forecast Occurrences by Hour
    if 'Forecast Error (MW)' in df_filtered.columns:
        df_filtered['Overforecast_Count'] = np.where(df_filtered['Forecast Error (MW)'] < 0, 1, 0)
        df_filtered['Underforecast_Count'] = np.where(df_filtered['Forecast Error (MW)'] > 0, 1, 0)

        grouped_count = df_filtered.groupby(df_filtered.index.hour).agg({
            'Overforecast_Count': 'sum',
            'Underforecast_Count': 'sum'
        }).reset_index()

        fig4 = go.Figure()
        hour_col = grouped_count.columns[0]
        fig4.add_trace(go.Bar(
            x=grouped_count[hour_col],
            y=grouped_count['Overforecast_Count'],
            name='Overforecast Count',
            marker_color='blue'
        ))
        fig4.add_trace(go.Bar(
            x=grouped_count[hour_col],
            y=grouped_count['Underforecast_Count'],
            name='Underforecast Count',
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


    # Plot 3: Bar Plot – Average Price Difference vs Binned Forecast Error
    if "LMP Difference (USD)" in df_filtered.columns and "Forecast Error (MW)" in df_filtered.columns:
    # Filter for weekdays only
        weekday_data = df_filtered[df_filtered.index.dayofweek < 5].copy()

        # Remove NaN values
        valid_data = weekday_data[['LMP Difference (USD)', 'Forecast Error (MW)', 'TOTAL Actual Load (MW)']].dropna()

        if not valid_data.empty:
            # Calculate the percentage error
            valid_data['% Error'] = (valid_data['Forecast Error (MW)'] / valid_data['TOTAL Actual Load (MW)']) * 100

            # Define the bins for percentage error
            bins = [-np.inf, -4, -2, -1, 0, 1, 2, 4, np.inf]
            bin_labels = ["<-4%", "-4% to -2%",  "-2% to -1%", "-1% to 0%", "0% to 1%", "1% to 2%", "2% to 4%",  ">4%"]

            # Bin the percentage error
            valid_data['Error Bin'] = pd.cut(valid_data['% Error'], bins=bins, labels=bin_labels, include_lowest=True, right=False)

            # Calculate the average price difference for each bin
            avg_price_diff_by_bin = valid_data.groupby('Error Bin')['LMP Difference (USD)'].mean().reset_index()

            # Create the bar plot
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=avg_price_diff_by_bin['Error Bin'],
                y=avg_price_diff_by_bin['LMP Difference (USD)'],
                name='Average Price Difference',
                marker_color='green'
            ))
            fig5.update_layout(
                title="Average Price Difference vs Binned Average % Error (Weekdays)",
                xaxis_title="Average % Error (Weekday)",
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
def render_weather_tab():
    st.subheader("Weather Data Analysis (Coming Soon)")
    st.info("Weather data integration and related metrics will be added in a future update.")
