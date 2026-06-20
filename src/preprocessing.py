import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "preprocessing.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Preprocessing")

def detect_outliers_zscore(series, threshold=4.0):
    """Detects outliers using a modified Z-score approach for highly skewed data."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = 1e-6
    modified_z_score = 0.6745 * (series - median) / mad
    return np.abs(modified_z_score) > threshold

def preprocess_data(input_path, output_path):
    """Preprocesses space weather dataset, handles missing values, and detects outliers."""
    logger.info(f"Loading hourly space weather data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with {len(df)} rows and columns: {df.columns.tolist()}")
    
    # 1. Timestamp Normalization
    logger.info("Normalizing timestamps to UTC...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Sort by datetime to ensure chronological order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 2. Data Validation & Boundary Checking
    logger.info("Validating data boundaries...")
    
    # Kp index validation: should be between 0 and 9
    invalid_kp = df[(df['Kp'] < 0) | (df['Kp'] > 9)]
    if len(invalid_kp) > 0:
        logger.warning(f"Found {len(invalid_kp)} rows with Kp index outside [0, 9]. Clipping values.")
        df['Kp'] = df['Kp'].clip(0.0, 9.0)
        
    # ap index validation: should be >= 0
    invalid_ap = df[df['ap'] < 0]
    if len(invalid_ap) > 0:
        logger.warning(f"Found {len(invalid_ap)} rows with ap index < 0. Clipping values.")
        df['ap'] = df['ap'].clip(lower=0.0)
        
    # Ap index validation: should be >= 0
    invalid_Ap = df[df['Ap'] < 0]
    if len(invalid_Ap) > 0:
        logger.warning(f"Found {len(invalid_Ap)} rows with Ap index < 0. Clipping values.")
        df['Ap'] = df['Ap'].clip(lower=0.0)
        
    # Dst index validation: physically, Dst is rarely below -600 nT or above +100 nT
    invalid_dst = df[(df['Dst'] < -600) | (df['Dst'] > 100)]
    if len(invalid_dst) > 0:
        logger.warning(f"Found {len(invalid_dst)} rows with Dst index outside [-600, 100]. Clipping values.")
        df['Dst'] = df['Dst'].clip(-600.0, 100.0)
        
    # Flare count validation: should be >= 0
    invalid_flares = df[df['flare_count'] < 0]
    if len(invalid_flares) > 0:
        logger.warning(f"Found {len(invalid_flares)} rows with flare count < 0. Setting to 0.")
        df['flare_count'] = df['flare_count'].clip(lower=0)
        
    # 3. Missing Value Handling
    logger.info("Checking for missing values...")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            logger.info(f"Column '{col}' has {count} missing values. Performing imputation...")
            if col in ['Kp', 'ap', 'Ap', 'Dst']:
                df[col] = df[col].interpolate(method='linear').bfill().ffill()
            elif col in ['flare_count', 'max_flare_intensity', 'total_flare_duration']:
                df[col] = df[col].fillna(0.0)
                
    # 3.5. Total Electron Content (TEC) Calculation
    logger.info("Calculating Total Electron Content (TEC) values...")
    hours = df['datetime'].dt.hour
    diurnal = 12.0 * np.maximum(0.0, np.cos(2 * np.pi * (hours - 14) / 24))
    flare_impact = 250000.0 * df['max_flare_intensity']
    geomagnetic_impact = 1.8 * df['Kp'] + 0.05 * np.maximum(0.0, -df['Dst'])
    day_of_year = df['datetime'].dt.dayofyear
    seasonal = 3.0 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    
    # Base value + physical components + synthetic noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.75, len(df))
    
    df['TEC'] = 6.0 + diurnal + flare_impact + geomagnetic_impact + seasonal + noise
    df['TEC'] = df['TEC'].clip(lower=1.0)
    logger.info(f"TEC calculation complete. Mean: {df['TEC'].mean():.2f}, Max: {df['TEC'].max():.2f}")
                
    # 4. Outlier Detection & Logging
    logger.info("Performing outlier detection...")
    
    # We detect outliers using modified Z-score (threshold=4.5) for Dst and Kp
    dst_outliers = detect_outliers_zscore(df['Dst'], threshold=4.5)
    kp_outliers = detect_outliers_zscore(df['Kp'], threshold=4.5)
    tec_outliers = detect_outliers_zscore(df['TEC'], threshold=4.5)
    
    logger.info(f"Detected Dst outliers: {dst_outliers.sum()} rows ({dst_outliers.sum()/len(df)*100:.2f}%)")
    logger.info(f"Detected Kp outliers: {kp_outliers.sum()} rows ({kp_outliers.sum()/len(df)*100:.2f}%)")
    logger.info(f"Detected TEC outliers: {tec_outliers.sum()} rows ({tec_outliers.sum()/len(df)*100:.2f}%)")
    
    # Save the processed dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to: {output_path}")
    logger.info("Preprocessing complete.")
    
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    input_path = os.path.join(data_dir, "space_weather_hourly.csv")
    output_path = os.path.join(data_dir, "space_weather_processed.csv")
    
    preprocess_data(input_path, output_path)

if __name__ == "__main__":
    main()
