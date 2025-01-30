import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
# Add to your imports at the top
import pytz

# Suppose these come from your local modules:
from functions import ISO_CONFIG, download_data  # Your data functions + ISO_CONFIG

# ---------------------------------
# Better Cache Management + Helpers
# ---------------------------------
@st.cache_data(ttl=24*60*60)  # Cache for 24 hours
def load_and_process_data(iso_key):
    """
    Download and preprocess data for the selected ISO.
    The config for each ISO is in ISO_CONFIG[iso_key].
    """
    config = ISO_CONFIG[iso_key]
    raw_data = download_data(config['filenames'])
    if raw_data is not None:
        df = config['processor'](raw_data)
        return df
    else:
        return None
def ensure_uniform_hourly_index(df, iso_key):
    """
    Robust timezone handling with explicit DST ambiguity resolution
    """
    config = ISO_CONFIG[iso_key]
    
    # 1. Remove duplicates first
    df = df[~df.index.duplicated(keep='first')]
    
    # 2. Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 3. Handle timezone conversion with DST parameters
    try:
        if df.index.tz is None:
            # Localize with explicit DST handling
            df = df.tz_localize(
                config['timezone'],
                ambiguous='infer',  # Let pandas infer DST based on timestamp order
                nonexistent='shift_forward'  # Handle spring-forward transitions
            )
        else:
            df = df.tz_convert(config['timezone'])
    except pytz.exceptions.AmbiguousTimeError:
        # Fallback for ambiguous times: assume non-DST (standard time)
        df = df.tz_localize(
            config['timezone'],
            ambiguous=False,
            nonexistent='shift_forward'
        )
    
    # 4. Convert to UTC for uniform handling
    df = df.tz_convert('UTC')
    
    # 5. Create complete UTC index
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='H',
        tz='UTC'
    )
    
    # 6. Reindex and interpolate
    df = df.reindex(full_range)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    
    return df
def main():
    # 1) Page config and styling
    st.set_page_config(page_title="ISO Load Forecast Analysis", layout="wide")
    st.markdown(
        """
        <style>
            body { background-color: #FFFFFF; color: #000000; }
            .stApp { background-color: #FFFFFF; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("ISO Load Forecast Analysis")

    # 2) Sidebar for ISO selection
    st.sidebar.header("Settings")
    iso = st.sidebar.selectbox("Select ISO", options=list(ISO_CONFIG.keys()))

    # 3) Load and process data (cached)
    df_raw = load_and_process_data(iso)
    if df_raw is None:
        st.error(f"Failed to download or process data for {iso}.")
        st.stop()

    # 4) Date range slider
    st.sidebar.subheader("Time Period")
    min_date = df_raw.index.min().date()
    max_date = df_raw.index.max().date()
    default_start = max_date - pd.Timedelta(days=30)

    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD"
    )

    df_range = df_raw.loc[str(start_date):str(end_date)].copy()

    # 5) Ensure uniform hourly index & fill missing data
    df = ensure_uniform_hourly_index(df_range,iso)

    # 6) Compute 30-day rolling average for forecast error
    if 'Forecast Error (MW)' in df.columns:
        # Adjust "30 days" = 24*30 = 720 hours
        df['Error_MA_30D'] = df['Forecast Error (MW)'].rolling(window=24*30, min_periods=1).mean()
    else:
        st.warning("Data does not contain 'Forecast Error (MW)'. Rolling average not computed.")

    # -------------------------------------------------------------
    # Plot 1 & 2: (1) Actual vs. Forecast and (2) Forecast Error
    # -------------------------------------------------------------
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Load vs Forecast", "Forecast Error vs Time")
    )

    # (a) Row 1: Actual vs Forecast
    if 'TOTAL Actual Load (MW)' in df.columns and 'SystemTotal Forecast Load (MW)' in df.columns:
        y_min = min(df['TOTAL Actual Load (MW)'].min(),
                    df['SystemTotal Forecast Load (MW)'].min())

        # Actual Load
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TOTAL Actual Load (MW)'],
                name='Actual Load',
                mode='lines',
                connectgaps=True,
                line=dict(color='rgba(0,100,80,0.8)')
            ),
            row=1, col=1
        )

        # Fill baseline
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=[y_min]*len(df),
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
                connectgaps=True,
                line=dict(color='rgba(0,0,255,0.8)')
            ),
            row=1, col=1
        )

        fig1.update_yaxes(title_text="Load (MW)", range=[y_min, None], row=1, col=1)
    else:
        st.warning("Missing columns for Actual Load or Forecast. Skipping first plot.")

    # (b) Row 2: Forecast Error (pos fill vs neg fill) + 30-day MA
    if 'Forecast Error (MW)' in df.columns:
        df_positive = df['Forecast Error (MW)'].clip(lower=0)
        df_negative = df['Forecast Error (MW)'].clip(upper=0)

        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df_positive,
                fill='tozeroy',
                name='Over-forecast',
                connectgaps=True,
                fillcolor='rgba(0,0,255,0.2)',
                line=dict(color='rgba(0,0,255,0.0)')
            ),
            row=2, col=1
        )

        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df_negative,
                fill='tozeroy',
                name='Under-forecast',
                connectgaps=True,
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0.0)')
            ),
            row=2, col=1
        )

        # 30-day rolling average
        if 'Error_MA_30D' in df.columns:
            fig1.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Error_MA_30D'],
                    name='30-Day MA Error',
                    mode='lines',
                    connectgaps=True,
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )

        fig1.update_yaxes(title_text="Forecast Error (MW)", row=2, col=1)
    else:
        st.warning("No 'Forecast Error (MW)' column to plot error. Skipping second plot.")

    # X-axis
    fig1.update_xaxes(title_text="Date", row=2, col=1)

    # Layout
    fig1.update_layout(
        hovermode='x unified',
        showlegend=True,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#000000'),
        margin=dict(l=40, r=40, b=80, t=80),
        height=800
    )
    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------------------------
    # Heatmap: Average Forecast Error by DayType/Hour
    # -------------------------------------------------
    if 'Forecast Error (MW)' in df.columns:
        df['DayType'] = np.where(df.index.dayofweek < 5, 'Weekday', 'Weekend')
        df['Hour'] = df.index.hour

        df_heatmap = df.groupby(['DayType', 'Hour'])['Forecast Error (MW)'].mean().reset_index()
        pivot_table = df_heatmap.pivot(index='Hour', columns='DayType', values='Forecast Error (MW)')

        fig2 = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title='Avg Error (MW)',
                orientation='v',
                thickness=15,
                len=0.6,
                x=1.02,
                xanchor='left',
                yanchor='middle'
            )
        ))
        fig2.update_xaxes(title_text='Day Type')
        fig2.update_yaxes(title_text='Hour of Day')
        fig2.update_layout(
            title="Heatmap of Avg Error (Weekday vs Weekend, Hour)",
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#000000'),
            margin=dict(l=40, r=100, b=80, t=80),
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------------------------------------------
    # Seasonal Decomposition: Weekly and Daily Patterns
    # ----------------------------------------------------------------
    if 'Forecast Error (MW)' in df.columns:
        series = df['Forecast Error (MW)'].dropna()
        
        # Check data sufficiency
        can_do_weekly = len(series) >= 2 * 24 * 7  # At least 2 weeks
        can_do_daily = len(series) >= 2 * 24       # At least 2 days

        if can_do_weekly or can_do_daily:
            st.subheader("Seasonal Error Patterns")
            cols = st.columns(2)
            
            if can_do_weekly:
                with cols[0]:
                    # Weekly decomposition
                    weekly_decomp = seasonal_decompose(
                        series, period=24*7, model='additive', extrapolate_trend='freq'
                    )
                    # Extract day-of-week averages from seasonal component
                    weekly_seasonal = weekly_decomp.seasonal
                    weekly_df = pd.DataFrame({
                        'Error': weekly_seasonal,
                        'DayOfWeek': weekly_seasonal.index.day_name()
                    })
                    week_avg = weekly_df.groupby('DayOfWeek')['Error'].mean().reindex([
                        'Monday', 'Tuesday', 'Wednesday', 
                        'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ])
                    
                    # Plot
                    fig_weekly = go.Figure()
                    fig_weekly.add_trace(go.Bar(
                        x=week_avg.index,
                        y=week_avg.values,
                        marker_color='royalblue'
                    ))
                    fig_weekly.update_layout(
                        title="Weekly Pattern (Avg Error by Day of Week)",
                        xaxis_title="Day of Week",
                        yaxis_title="Seasonal Error Component (MW)",
                        height=400
                    )
                    st.plotly_chart(fig_weekly, use_container_width=True)

            if can_do_daily:
                with cols[1]:
                    # Daily decomposition
                    daily_decomp = seasonal_decompose(
                        series, period=24, model='additive', extrapolate_trend='freq'
                    )
                    # Extract hourly averages from seasonal component
                    daily_seasonal = daily_decomp.seasonal
                    hour_avg = daily_seasonal.groupby(daily_seasonal.index.hour).mean()
                    
                    # Plot
                    fig_daily = go.Figure()
                    fig_daily.add_trace(go.Bar(
                        x=hour_avg.index,
                        y=hour_avg.values,
                        marker_color='coral'
                    ))
                    fig_daily.update_layout(
                        title="Daily Pattern (Avg Error by Hour)",
                        xaxis_title="Hour of Day",
                        yaxis_title="Seasonal Error Component (MW)",
                        xaxis=dict(tickmode='linear', dtick=1),
                        height=400
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)

    # -----------------------------------------------------------
    # Optional: Trend Component Visualization (Checkbox)
    # -----------------------------------------------------------
    if st.checkbox("Show Long-Term Trend Component") and 'Forecast Error (MW)' in df.columns:
        series = df['Forecast Error (MW)'].dropna()
        if len(series) >= 2 * 24 * 30:  # At least 2 months
            trend_decomp = seasonal_decompose(
                series, period=24*30, model='additive', extrapolate_trend='freq'
            )
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_decomp.trend.index,
                y=trend_decomp.trend,
                line=dict(color='green', width=2)
            ))
            fig_trend.update_layout(
                title="Long-Term Trend Component (30-day Seasonality)",
                xaxis_title="Date",
                yaxis_title="Trend Component (MW)",
                height=400
            )
            st.plotly_chart(fig_trend, use_container_width=True)



if __name__ == "__main__":
    main()
