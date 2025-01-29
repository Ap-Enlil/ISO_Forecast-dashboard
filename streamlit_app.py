import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions import ISO_CONFIG, download_data  # Your data functions + ISO_CONFIG

# -----------------------
# Better Cache Management
# -----------------------
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

# Use a wide page layout and set a white background.
st.set_page_config(page_title="ISO Load Forecast Analysis", layout="wide")

# Override default styling to ensure a white background.
st.markdown(
    """
    <style>
        body { background-color: #FFFFFF; color: #000000; }
        .stApp { background-color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True
)

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

    # Create subplots for the first two figures: 2 rows, 1 column, shared X-axis
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Load vs Forecast",
            "Forecast Error vs Time"
        )
    )

    # --- First plot (row=1) ---
    # Calculate y_min from visible data
    y_min = min(df['TOTAL Actual Load (MW)'].min(), 
                df['SystemTotal Forecast Load (MW)'].min())

    # Actual Load with fill to y_min
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df['TOTAL Actual Load (MW)'],
            name='Actual Load',
            mode='lines',
            line=dict(color='rgba(0,100,80,0.8)')
        ),
        row=1, col=1
    )

    # Invisible trace for fill baseline
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=[y_min] * len(df),
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            showlegend=False
        ),
        row=1, col=1
    )

    # Forecast line
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SystemTotal Forecast Load (MW)'],
            name='Forecast',
            mode='lines',
            line=dict(color='rgba(0,0,255,0.8)')
        ),
        row=1, col=1
    )

    fig1.update_yaxes(
        title_text="Load (MW)",
        range=[y_min, None],
        row=1,
        col=1
    )

    # --- Second plot (row=2) ---
    df_positive = df['Forecast Error (MW)'].clip(lower=0)
    df_negative = df['Forecast Error (MW)'].clip(upper=0)

    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df_positive,
            fill='tozeroy',
            name='Over-forecast',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(0, 0, 255, 0.0)')
        ),
        row=2, col=1
    )

    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df_negative,
            fill='tozeroy',
            name='Under-forecast',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255, 0, 0, 0.0)')
        ),
        row=2, col=1
    )

    fig1.update_yaxes(
        title_text="Forecast Error (MW)",
        row=2,
        col=1
    )

    # Update x-axis for the first two plots
    fig1.update_xaxes(title_text="Date", row=2, col=1)

    # Global figure styling for the first two plots
    fig1.update_layout(
        hovermode='x unified',
        showlegend=True,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#000000'),
        margin=dict(l=40, r=40, b=80, t=80),
        height=800  # Adjust height as needed
    )

    # Render the first figure (Load vs Forecast and Forecast Error)
    st.plotly_chart(fig1, use_container_width=True)

    # --- Third plot: Heatmap (separate figure) ---
    df['DayType'] = np.where(df.index.dayofweek < 5, 'Weekday', 'Weekend')
    df['Hour'] = df.index.hour

    # Group by DayType and Hour to compute average Forecast Error
    df_heatmap = df.groupby(['DayType','Hour'])['Forecast Error (MW)'].mean().reset_index()
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
            x=1.02,          # Position colorbar to the right of heatmap
            xanchor='left',  # Anchor to left side of colorbar
            yanchor='middle'
        )
    ))

    # Update third subplot axes
    fig2.update_xaxes(title_text='Day Type')
    fig2.update_yaxes(title_text='Hour of Day')

    # Global figure styling for the heatmap
    fig2.update_layout(
        title="Heatmap of Avg Error (Weekday vs Weekend, Hour)",
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#000000'),
        margin=dict(l=40, r=100, b=80, t=80),  # Increased right margin for colorbar
        height=600  # Adjust height as needed
    )

    # Render the heatmap
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()