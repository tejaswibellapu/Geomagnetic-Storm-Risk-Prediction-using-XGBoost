import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "event_detector.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EventDetector")

def get_goes_class_level(flare_class):
    """Returns classification level (e.g. X, M, C, B, A)."""
    if isinstance(flare_class, str) and len(flare_class) > 0:
        return flare_class[0].upper()
    return 'U'

def get_geomagnetic_storm_level(kp):
    """Maps Kp index to NOAA G-scale storm level."""
    if kp < 5.0:
        return "No Storm"
    elif kp < 6.0:
        return "G1 (Minor Storm)"
    elif kp < 7.0:
        return "G2 (Moderate Storm)"
    elif kp < 8.0:
        return "G3 (Strong Storm)"
    elif kp < 9.0:
        return "G4 (Severe Storm)"
    else:
        return "G5 (Extreme Storm)"

def detect_significant_events(event_aligned_path, output_path):
    """Scans aligned data to identify major solar-geomagnetic event pairs."""
    logger.info("Starting significant event detection...")
    
    if not os.path.exists(event_aligned_path):
        raise FileNotFoundError(f"Event-aligned data not found: {event_aligned_path}")
        
    df = pd.read_csv(event_aligned_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # 1. Filter for major solar flares: M or X class
    df['class_level'] = df['flare_class'].apply(get_goes_class_level)
    major_flares = df[df['class_level'].isin(['M', 'X'])].copy()
    
    logger.info(f"Found {len(major_flares)} major (M/X class) solar flare events.")
    
    significant_events = []
    
    for idx, row in major_flares.iterrows():
        kp_max = row['Kp_max_48h']
        dst_min = row['Dst_min_48h']
        
        # Determine storm level based on 48h subsequent response
        storm_level = get_geomagnetic_storm_level(kp_max)
        
        # Categorize correlation
        if kp_max >= 7.0 or dst_min <= -100:
            correlation_status = "Strong Correlation"
            severity = "Severe"
        elif kp_max >= 5.0 or dst_min <= -50:
            correlation_status = "Moderate Correlation"
            severity = "Moderate"
        else:
            correlation_status = "No Significant Response"
            severity = "Low"
            
        event_name = f"{row['flare_class']} Solar Flare"
        if pd.notnull(row['active_region']) and row['active_region'] > 0:
            event_name += f" (AR {int(row['active_region'])})"
            
        geomagnetic_response = f"{storm_level} (Kp max: {kp_max:.1f}, Dst min: {dst_min:.1f} nT)"
        
        significant_events.append({
            'flare_id': row['flare_id'],
            'flare_class': row['flare_class'],
            'flare_time': row['time'],
            'flare_intensity': row['flare_intensity'],
            'duration_min': row['duration_min'],
            'active_region': row['active_region'],
            'Kp_max_48h': kp_max,
            'Dst_min_48h': dst_min,
            'storm_level': storm_level,
            'severity': severity,
            'correlation_status': correlation_status,
            'event_name': event_name,
            'geomagnetic_response': geomagnetic_response
        })
        
    df_events = pd.DataFrame(significant_events)
    # Sort chronologically
    df_events = df_events.sort_values('flare_time').reset_index(drop=True)
    
    # Save to CSV
    df_events.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df_events)} detected significant events to: {output_path}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    event_aligned_path = os.path.join(data_dir, "event_aligned_data.csv")
    output_path = os.path.join(data_dir, "significant_events.csv")
    
    detect_significant_events(event_aligned_path, output_path)

if __name__ == "__main__":
    main()
