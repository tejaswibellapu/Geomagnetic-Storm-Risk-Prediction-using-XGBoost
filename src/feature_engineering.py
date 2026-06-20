import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "feature_engineering.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FeatureEngineering")

def engineer_features(input_path, output_path):
    """Generates features from hourly space weather data."""
    logger.info(f"Loading preprocessed data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    logger.info("Engineering solar and geomagnetic features...")
    
    # 1. Solar Features (Daily and Monthly counts)
    # 24 hours = 1 day
    df['daily_flare_count'] = df['flare_count'].rolling(window=24, min_periods=1).sum().astype(int)
    # 720 hours = 30 days
    df['monthly_flare_count'] = df['flare_count'].rolling(window=720, min_periods=1).sum().astype(int)
    
    # Event frequency (e.g. rolling mean of daily flare count)
    df['flare_event_frequency'] = df['flare_count'].rolling(window=168, min_periods=1).mean() # weekly frequency
    
    # 2. Derived Features: Rolling Means and Std Devs for geomagnetic variables
    windows = [6, 12, 24, 48, 72]
    
    for w in windows:
        logger.info(f"Computing rolling features for window size: {w} hours...")
        
        # Kp Index rolling features
        df[f'Kp_rolling_mean_{w}h'] = df['Kp'].rolling(window=w, min_periods=1).mean()
        df[f'Kp_rolling_std_{w}h'] = df['Kp'].rolling(window=w, min_periods=1).std().fillna(0.0)
        
        # Dst Index rolling features
        df[f'Dst_rolling_mean_{w}h'] = df['Dst'].rolling(window=w, min_periods=1).mean()
        df[f'Dst_rolling_std_{w}h'] = df['Dst'].rolling(window=w, min_periods=1).std().fillna(0.0)
        
        # ap Index rolling features
        df[f'ap_rolling_mean_{w}h'] = df['ap'].rolling(window=w, min_periods=1).mean()
        df[f'ap_rolling_std_{w}h'] = df['ap'].rolling(window=w, min_periods=1).std().fillna(0.0)
        
        # TEC rolling features
        df[f'TEC_rolling_mean_{w}h'] = df['TEC'].rolling(window=w, min_periods=1).mean()
        df[f'TEC_rolling_std_{w}h'] = df['TEC'].rolling(window=w, min_periods=1).std().fillna(0.0)
        
        # Solar flare intensity rolling features
        df[f'flare_intensity_rolling_mean_{w}h'] = df['max_flare_intensity'].rolling(window=w, min_periods=1).mean()
        df[f'flare_intensity_rolling_max_{w}h'] = df['max_flare_intensity'].rolling(window=w, min_periods=1).max()
        
    # 3. Lag Features
    lags = [1, 2, 3, 6, 12, 24, 48, 72]
    for lag in lags:
        logger.info(f"Computing lag features for lag: {lag} hours...")
        df[f'Kp_lag_{lag}h'] = df['Kp'].shift(lag)
        df[f'Dst_lag_{lag}h'] = df['Dst'].shift(lag)
        df[f'ap_lag_{lag}h'] = df['ap'].shift(lag)
        df[f'TEC_lag_{lag}h'] = df['TEC'].shift(lag)
        df[f'flare_count_lag_{lag}h'] = df['flare_count'].shift(lag)
        df[f'flare_intensity_lag_{lag}h'] = df['max_flare_intensity'].shift(lag)
        
    # 4. Trend Indicators
    logger.info("Computing trend indicators...")
    # Kp short-term vs long-term trend
    df['Kp_trend_6h_vs_24h'] = df['Kp_rolling_mean_6h'] - df['Kp_rolling_mean_24h']
    df['Dst_trend_6h_vs_24h'] = df['Dst_rolling_mean_6h'] - df['Dst_rolling_mean_24h']
    df['TEC_trend_6h_vs_24h'] = df['TEC_rolling_mean_6h'] - df['TEC_rolling_mean_24h']
    
    # 1-hour change rates
    df['Kp_diff_1h'] = df['Kp'].diff().fillna(0.0)
    df['Dst_diff_1h'] = df['Dst'].diff().fillna(0.0)
    df['TEC_diff_1h'] = df['TEC'].diff().fillna(0.0)
    
    # Handle NaNs from shifting operations by backfilling
    logger.info("Handling missing values from shifts...")
    df = df.bfill()
    
    # Save the output
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved feature engineered dataset to: {output_path}")
    logger.info(f"Dataset shape: {df.shape}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    input_path = os.path.join(data_dir, "space_weather_processed.csv")
    output_path = os.path.join(data_dir, "space_weather_features.csv")
    
    engineer_features(input_path, output_path)

if __name__ == "__main__":
    main()
