import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions import ISO_CONFIG, download_data  # Your data functions + ISO_CONFIG

# -----------------------
# Better Cache Management
# -----------------------
# st.cache_data will cache the results of a function *unless* its input arguments change.
# This way, data is only re-downloaded when you switch ISOs or after 24 hours (as defined by ttl).

@st.cache_data(ttl=24*60*60)  # Cache for 24 hours
def load_and_process_data(iso):
    """Download and preprocess data for the selected ISO."""
    config = ISO_CONFIG[iso]
    raw_data = download_data(config['filenames'])
    if raw_data is not None:
        df = config['processor'](raw_data)
        return df
    else:
        return None


def main():
    st.title("ISO Load Forecast Analysis")

    # --- Sidebar ---
    st.sidebar.header("Settings")
    iso = st.sidebar.selectbox("Select ISO", options=list(ISO_CONFIG.keys()))

    # Load and process data (cached)
    df = load_and_process_data(iso)

    if df is None:
        st.error(f"Failed to download or process data for {iso}.")
        st.stop()

    # --- Time Slider ---
    st.sidebar.subheader("Time Period")
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    # Default to last 30 days
    default_start = max_date - pd.Timedelta(days=30)

    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD"
    )

    # Filter DataFrame for the chosen date range
    df = df.loc[str(start_date):str(end_date)]

    # -----------------------------
    # 1) Load vs. Forecast Plot
    # -----------------------------
    st.subheader("Load vs Forecast")

    # Figure 1: Actual Load vs. Forecast
    fig1 = go.Figure()

    # Add Actual Load Trace
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=df['TOTAL Actual Load (MW)'],
        name='Actual Load',
        mode='lines',
        line=dict(color='rgba(0,100,80,0.8)'),
        fill='tozeroy',
        fillcolor='rgba(0,100,80,0.2)'  # Light green fill under actual load
    ))

    # Add Forecast Trace
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=df['SystemTotal Forecast Load (MW)'],
        name='Forecast',
        mode='lines',
        line=dict(color='rgba(0,0,255,0.8)')  # Blue line
    ))

    # Set Y-axis min to 10% lower than the overall min in the displayed data
    combined_min = min(df['TOTAL Actual Load (MW)'].min(), df['SystemTotal Forecast Load (MW)'].min())
    y_min = combined_min * 0.9

    fig1.update_layout(
        yaxis=dict(title="Load (MW)", range=[y_min, None]),
        hovermode='x unified',
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ----------------------------------------------------------
    # 2) Forecast Error Plot (filled above/below zero + 30D MA)
    # ----------------------------------------------------------
    st.subheader("Forecast Error vs Time")

    # Create 30-day moving average of the forecast error
    df['Forecast Error 30D (MW)'] = df['Forecast Error (MW)'].rolling('30D').mean()

    # Separate positive and negative errors for color filling
    df_positive = df['Forecast Error (MW)'].clip(lower=0)  # 0 where negative
    df_negative = df['Forecast Error (MW)'].clip(upper=0)  # 0 where positive

    fig2 = go.Figure()

    # Positive errors (over-forecast) = fill in blue
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df_positive,
        fill='tozeroy',
        name='Over-forecast',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(0, 0, 255, 0.0)'),  # Hide the line, keep fill
        hoverinfo='x+y'
    ))

    # Negative errors (under-forecast) = fill in red
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df_negative,
        fill='tozeroy',
        name='Under-forecast',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0.0)'),
        hoverinfo='x+y'
    ))

    # Add a line for the forecast error (so you can see the sign crossing zero)
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df['Forecast Error (MW)'],
        name='Error (MW)',
        line=dict(color='black'),
        hoverinfo='x+y'
    ))

    # Add the 30-day moving average line
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df['Forecast Error 30D (MW)'],
        name='30-Day MA',
        line=dict(color='orange', dash='dash'),
        hoverinfo='x+y'
    ))

    fig2.update_layout(
        yaxis_title="Forecast Error (MW)",
        hovermode='x unified'
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------
    # 3) Forecast Performance
    # -----------------------
    st.subheader("Forecast Performance Metrics")

    # Calculate some metrics
    mae = df['Forecast Error (MW)'].abs().mean()
    mape = df['MAPE (%)'].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="MAE (MW)",
            value=f"{mae:.0f}",
            help="Mean Absolute Error (MAE) measures the average magnitude of the errors "
                 "between forecasted and actual values, without considering their direction."
        )
    with col2:
        st.metric(
            label="MAPE (%)",
            value=f"{mape:.1f}",
            help="Mean Absolute Percentage Error (MAPE) is the average of the absolute percentage "
                 "errors. It indicates how large the forecast errors are in percentage terms."
        )

    # -----------------------------------------
    # 4) Optional: Heatmap by Hour and Day Type
    # -----------------------------------------
    st.subheader("Error Analysis by Hour and Day Type")
    error_type = st.radio("Select Error Metric", ["MW", "APE"])

    df_heat = df.copy()
    df_heat['hour'] = df_heat.index.hour
    df_heat['weekday'] = df_heat.index.dayofweek < 5  # True for weekdays (Mon-Fri), False for weekends (Sat-Sun)

    if error_type == "MW":
        metric_col = 'Forecast Error (MW)'
        color_scale = 'RdYlGn'  # Red-Yellow-Green
        zmid = 0  # Center scale at 0
    else:  # APE
        metric_col = 'MAPE (%)'
        color_scale = 'Viridis'
        zmid = None

    # Group by hour and weekday/weekend
    heat_data = df_heat.groupby(['hour', 'weekday'])[metric_col].mean().reset_index()
    heat_data['Day Type'] = heat_data['weekday'].map({True: 'Weekday', False: 'Weekend'})
    heat_pivot = heat_data.pivot(index='hour', columns='Day Type', values=metric_col)

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns,
        y=heat_pivot.index,
        colorscale=color_scale,
        zmid=zmid,
        ygap=1,
        xgap=1,
        colorbar=dict(title=error_type)
    ))

    fig_heat.update_layout(
        xaxis_title="Day Type",
        yaxis_title="Hour of Day",
        yaxis=dict(autorange="reversed"),  # So hour=0 is at the top
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig_heat, use_container_width=True)


if __name__ == "__main__":
    main()
