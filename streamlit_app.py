import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from iso_data_integration import ISO_CONFIG, load_all_iso_data, ensure_uniform_hourly_index
from metrics_calculation import compute_iso_metrics

# ---------------------------------
# Better Cache Management + Helpers
# ---------------------------------
@st.cache_data(ttl=24*60*60)  # Cache for 24 hours
def load_and_process_data(iso_key):
    """
    Download and preprocess data for the selected ISO.
    The config for each ISO is in ISO_CONFIG[iso_key].
    """
    iso_data = load_all_iso_data()
    return iso_data.get(iso_key)

###############################
# 4) Streamlit App
###############################
def main():
    st.set_page_config(page_title="ISO Forecast Comparison", layout="wide")
    st.title("ISO Load Forecast - Comparison & Analysis")

    # 4.1) Load *all* data
    iso_data_dict = load_all_iso_data()  # {ISO_KEY -> df}

    # Create two tabs
    tab_comparison, tab_analysis = st.tabs(["Comparison (Mega Table)", "Single ISO Analysis"])

    # ============================
    # TAB 1: Comparison / Mega Table
    # ============================
    with tab_comparison:
        st.subheader("Comparison of All ISOs")

        # Optionally let user select a date range
        st.write("**Select Date Range for Metrics**")
        # We'll find a global min/max date across all loaded ISOs
        all_indices = []
        for df_ in iso_data_dict.values():
            if df_ is not None:
                all_indices.append(df_.index)
        if not all_indices:
            st.error("No data available for any ISO.")
            return

        # Global min/max
        global_min = min(idx.min() for idx in all_indices)
        global_max = max(idx.max() for idx in all_indices)
        default_start = global_max - pd.Timedelta(days=30)  # last 30 days default

        start_date, end_date = st.slider(
            label="Date Range",
            min_value=global_min.to_pydatetime().date(),
            max_value=global_max.to_pydatetime().date(),
            value=(default_start.to_pydatetime().date(), global_max.to_pydatetime().date()),
            format="YYYY-MM-DD"
        )

        # Now compute metrics for each ISO in that date window
        metrics_dict = {}
        for iso_key, df_ in iso_data_dict.items():
            if df_ is not None and not df_.empty:
                # Filter by date
                df_filtered = df_.loc[str(start_date):str(end_date)].copy()
                # Optionally ensure uniform index (if you want to do so for metrics):
                df_filtered = ensure_uniform_hourly_index(df_filtered, iso_key)

                # Compute metrics
                metrics = compute_iso_metrics(df_filtered)
                metrics_dict[iso_key] = metrics
            else:
                # If we have no data for that ISO, fill with NaN
                metrics_dict[iso_key] = {
                    'Avg APE (%)': np.nan,
                    'Avg Error (MW)': np.nan,
                    'MAPE (%)': np.nan,
                    'Avg % Error (Morning)': np.nan,
                    'Avg % Error (Afternoon)': np.nan,
                    'Avg % Error (Evening)': np.nan,
                    'Avg % Error (Night)': np.nan,
                    'Avg % Error (Weekday)': np.nan,
                    'Avg % Error (Weekend)': np.nan
                }

        # Create a DataFrame
        df_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index')
        # Sort the index so it's always in a consistent order (SPP, MISO, ERCOT, CAISO, PJM)
        iso_order = list(ISO_CONFIG.keys())  # e.g. ["SPP", "MISO", "ERCOT", "CAISO", "PJM"]
        df_metrics = df_metrics.reindex(index=iso_order)

        # Round for display
        df_metrics = df_metrics.round(2)

        # Let user pick a metric to "rank" by
        rank_by_col = st.selectbox(
            "Rank by which metric?",
            options=df_metrics.columns.tolist(),
            index=0
        )

        # Sort by that metric ascending
        df_metrics_sorted = df_metrics.sort_values(by=rank_by_col, ascending=True)

        st.write("**Comparison Table (Ranked by:** {}**)**".format(rank_by_col))
        st.dataframe(df_metrics_sorted)

        # Add metric explanations
        st.markdown("""
        **Metric Definitions:**

        *   **Avg APE (%)**: Average Absolute Percentage Error. The average of the absolute values of the percentage errors. It measures the magnitude of errors regardless of direction.
            *   `APE = |(Actual - Forecast) / Actual| * 100`
            *   `Avg APE = (1/n) * Σ APE`
        *   **Avg Error (MW)**: The average forecast error in megawatts (MW). A positive value indicates under-forecasting, and a negative value indicates over-forecasting.
            *   `Avg Error = (1/n) * Σ (Actual - Forecast)`
        *   **MAPE (%)**: Mean Absolute Percentage Error. The average of the absolute percentage errors, providing a measure of the overall forecast accuracy relative to the actual load.
            *   `MAPE = (1/n) * Σ |(Actual - Forecast) / Actual| * 100`
        *   **Avg % Error (Morning/Afternoon/Evening/Night)**: The average percentage error during specific time periods. This helps identify if the forecast performs better or worse during certain times of the day.
            *   `% Error = (Actual - Forecast) / Actual * 100`
            *   `Avg % Error = (1/n) * Σ % Error` (for the given time period)
        *   **Avg % Error (Weekday/Weekend)**: The average percentage error on weekdays versus weekends. This helps determine if there are systematic differences in forecast performance based on the day of the week.
            *   `Avg % Error = (1/n) * Σ % Error` (for weekdays or weekends)
        """)

    # ============================
    # TAB 2: Single ISO Analysis
    # ============================
    with tab_analysis:
        st.subheader("Single ISO Detailed Analysis")

        # Let user pick an ISO
        iso_list = list(ISO_CONFIG.keys())
        selected_iso = st.selectbox("Select ISO for Analysis", options=iso_list)

        df_selected = iso_data_dict[selected_iso]

        if df_selected is None or df_selected.empty:
            st.error(f"No data available for {selected_iso}.")
            return

        # Let user pick date range
        min_date = df_selected.index.min().date()
        max_date = df_selected.index.max().date()
        default_start = max_date - pd.Timedelta(days=30)

        start_date_iso, end_date_iso = st.slider(
            f"Select Date Range for {selected_iso}",
            min_value=min_date,
            max_value=max_date,
            value=(default_start, max_date),
            format="YYYY-MM-DD"
        )

        # Filter data
        df_range = df_selected.loc[str(start_date_iso):str(end_date_iso)].copy()

        # Ensure uniform hourly index
        df = ensure_uniform_hourly_index(df_range, selected_iso)

        # (Optional) 30-day rolling average
        if 'Forecast Error (MW)' in df.columns:
            df['Error_MA_30D'] = df['Forecast Error (MW)'].rolling(window=24*30, min_periods=1).mean()

        # ============= PLOTS (Same as your old code) =============
        # (1) Actual vs Forecast and (2) Forecast Error
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

        # (b) Row 2: Forecast Error
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
        fig1.update_layout(height=800, hovermode='x unified', showlegend=True)

        st.plotly_chart(fig1, use_container_width=True)

        # ============= Heatmap =============
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
                height=600
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ============= Seasonal Decomposition =============
        if 'Forecast Error (MW)' in df.columns:
            series = df['Forecast Error (MW)'].dropna()

            # weekly/daily decomposition
            can_do_weekly = len(series) >= 2 * 24 * 7
            can_do_daily = len(series) >= 2 * 24

            if can_do_weekly or can_do_daily:
                st.subheader("Seasonal Error Patterns")
                cols = st.columns(2)

                if can_do_weekly:
                    with cols[0]:
                        weekly_decomp = seasonal_decompose(
                            series, period=24*7, model='additive', extrapolate_trend='freq'
                        )
                        weekly_seasonal = weekly_decomp.seasonal
                        wdf = pd.DataFrame({
                            'Error': weekly_seasonal,
                            'DayOfWeek': weekly_seasonal.index.day_name()
                        })
                        week_avg = wdf.groupby('DayOfWeek')['Error'].mean().reindex([
                            'Monday', 'Tuesday', 'Wednesday',
                            'Thursday', 'Friday', 'Saturday', 'Sunday'
                        ])

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
                        daily_decomp = seasonal_decompose(
                            series, period=24, model='additive', extrapolate_trend='freq'
                        )
                        daily_seasonal = daily_decomp.seasonal
                        hour_avg = daily_seasonal.groupby(daily_seasonal.index.hour).mean()

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

        # ============= Long-Term Trend Checkbox =============
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
