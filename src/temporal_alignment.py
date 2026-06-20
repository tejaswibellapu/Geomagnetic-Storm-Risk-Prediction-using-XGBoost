import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "temporal_alignment.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TemporalAlignment")

def align_events_optimized(flares_path, hourly_path, output_path):
    """Aligns solar flare events with geomagnetic responses using fast vectorized reversed rolling windows."""
    logger.info("Starting optimized temporal alignment...")
    
    if not os.path.exists(flares_path):
        raise FileNotFoundError(f"Flares file not found: {flares_path}")
    if not os.path.exists(hourly_path):
        raise FileNotFoundError(f"Hourly path not found: {hourly_path}")
        
    df_flares = pd.read_csv(flares_path)
    df_hourly = pd.read_csv(hourly_path)
    
    df_flares['time'] = pd.to_datetime(df_flares['time'])
    df_flares['start_time'] = pd.to_datetime(df_flares['start_time'])
    df_flares['end_time'] = pd.to_datetime(df_flares['end_time'])
    
    df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
    
    # Sort data chronologically
    df_flares = df_flares.sort_values('time').reset_index(drop=True)
    df_hourly = df_hourly.sort_values('datetime').reset_index(drop=True)
    
    logger.info("Computing future rolling indices on hourly timeline...")
    
    # Calculate flare duration and intensity for the flares dataframe
    from src.data_collection import flare_class_to_intensity
    df_flares['flare_intensity'] = df_flares['flare_class'].apply(flare_class_to_intensity)
    
    duration = (df_flares['end_time'] - df_flares['start_time']).dt.total_seconds() / 60.0
    duration = duration.fillna(20.0)
    duration = np.where(duration < 0, 20.0, duration)
    df_flares['duration_min'] = duration
    
    # For each lag window, compute future max Kp, ap, Ap and future min Dst
    lag_windows = [3, 6, 12, 24, 48, 72]
    
    # Copy hourly dataframe for lookup features
    lookup_df = df_hourly[['datetime', 'Kp', 'Dst', 'ap', 'Ap', 'TEC']].copy()
    
    for lag in lag_windows:
        logger.info(f"Computing look-ahead features for lag window: {lag} hours...")
        
        # In a 1-hour resolution dataset, a lag of L hours is exactly L+1 steps (current hour + L hours ahead)
        steps = lag + 1
        
        # Future max Kp
        lookup_df[f'Kp_max_{lag}h'] = lookup_df['Kp'].iloc[::-1].rolling(window=steps, min_periods=1).max().iloc[::-1]
        # Future min Dst
        lookup_df[f'Dst_min_{lag}h'] = lookup_df['Dst'].iloc[::-1].rolling(window=steps, min_periods=1).min().iloc[::-1]
        # Future max ap
        lookup_df[f'ap_max_{lag}h'] = lookup_df['ap'].iloc[::-1].rolling(window=steps, min_periods=1).max().iloc[::-1]
        # Future max Ap
        lookup_df[f'Ap_max_{lag}h'] = lookup_df['Ap'].iloc[::-1].rolling(window=steps, min_periods=1).max().iloc[::-1]
        # Future max TEC
        lookup_df[f'TEC_max_{lag}h'] = lookup_df['TEC'].iloc[::-1].rolling(window=steps, min_periods=1).max().iloc[::-1]
        
    logger.info("Aligning flare events to hourly future indices...")
    
    # We map each flare event to the nearest hourly datetime block
    df_flares['datetime_hour'] = df_flares['time'].dt.floor('h')
    
    # Merge flares with look-ahead features
    feature_cols = ['datetime'] + [f'Kp_max_{lag}h' for lag in lag_windows] + \
                                  [f'Dst_min_{lag}h' for lag in lag_windows] + \
                                  [f'ap_max_{lag}h' for lag in lag_windows] + \
                                  [f'Ap_max_{lag}h' for lag in lag_windows] + \
                                  [f'TEC_max_{lag}h' for lag in lag_windows]
                                  
    df_aligned = pd.merge(
        df_flares,
        lookup_df[feature_cols],
        left_on='datetime_hour',
        right_on='datetime',
        how='left'
    )
    
    # Drop intermediate columns
    df_aligned.drop(columns=['datetime_hour', 'datetime'], inplace=True)
    
    # Handle any NaN entries from edge overlaps (fill with median)
    for col in df_aligned.columns:
        if df_aligned[col].isnull().sum() > 0:
            if col.startswith('Dst_min'):
                df_aligned[col] = df_aligned[col].fillna(df_aligned[col].median())
            elif col.startswith('Kp_max') or col.startswith('ap_max') or col.startswith('Ap_max') or col.startswith('TEC_max'):
                df_aligned[col] = df_aligned[col].fillna(df_aligned[col].median())
                
    # Save the output
    df_aligned.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(df_aligned)} event-aligned records to: {output_path}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    flares_path = os.path.join(data_dir, "solar_flares_raw.csv")
    hourly_path = os.path.join(data_dir, "space_weather_processed.csv")
    output_path = os.path.join(data_dir, "event_aligned_data.csv")
    
    align_events_optimized(flares_path, hourly_path, output_path)

if __name__ == "__main__":
    main()
