import numpy as np

def compute_iso_metrics(df):
    """
    Compute the updated metrics:
     - Average APE
     - Average Error (MW)
     - MAPE
     - Average Percentage Error (Morning, Afternoon, Evening, Night, Weekday, Weekend)
    Return a dictionary of metrics.
    """
    if df is None or 'TOTAL Actual Load (MW)' not in df.columns:
        return {
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

    df = df.dropna(subset=[
        'TOTAL Actual Load (MW)',
        'Forecast Error (MW)',
        'APE (%)',
        'Percentage Error (%)'
    ], how='any')

    if len(df) == 0:
        return {
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

    # Calculate basic metrics
    avg_ape = df['APE (%)'].mean()
    avg_error = df['Forecast Error (MW)'].mean()
    mape = abs(df['Forecast Error (MW)']).sum() / abs(df['TOTAL Actual Load (MW)']).sum() * 100

    # Time-of-day and day-of-week metrics
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek  # Monday=0, Sunday=6

    morning_mask = (df['Hour'] >= 6) & (df['Hour'] < 12)
    afternoon_mask = (df['Hour'] >= 12) & (df['Hour'] < 18)
    evening_mask = (df['Hour'] >= 18) & (df['Hour'] < 24)
    night_mask = (df['Hour'] >= 0) & (df['Hour'] < 6)
    weekday_mask = df['DayOfWeek'] < 5
    weekend_mask = df['DayOfWeek'] >= 5

    avg_pct_error_morning = df.loc[morning_mask, 'Percentage Error (%)'].mean()
    avg_pct_error_afternoon = df.loc[afternoon_mask, 'Percentage Error (%)'].mean()
    avg_pct_error_evening = df.loc[evening_mask, 'Percentage Error (%)'].mean()
    avg_pct_error_night = df.loc[night_mask, 'Percentage Error (%)'].mean()
    avg_pct_error_weekday = df.loc[weekday_mask, 'Percentage Error (%)'].mean()
    avg_pct_error_weekend = df.loc[weekend_mask, 'Percentage Error (%)'].mean()

    return {
        'Avg APE (%)': avg_ape,
        'Avg Error (MW)': avg_error,
        'MAPE (%)': mape,
        'Avg % Error (Morning)': avg_pct_error_morning,
        'Avg % Error (Afternoon)': avg_pct_error_afternoon,
        'Avg % Error (Evening)': avg_pct_error_evening,
        'Avg % Error (Night)': avg_pct_error_night,
        'Avg % Error (Weekday)': avg_pct_error_weekday,
        'Avg % Error (Weekend)': avg_pct_error_weekend
    }