import os
import re
import ssl
import json
import logging
import urllib.request
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "data_collection.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataCollection")

# Create unverified SSL context for APIs since some systems lack certificates
ssl_context = ssl._create_unverified_context()

def download_url(url, headers=None, retries=3):
    """Downloads a URL and returns the response body as bytes."""
    if headers is None:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                return response.read()
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            if attempt == retries - 1:
                raise e

def flare_class_to_intensity(flare_class_str):
    """Converts GOES solar flare class (e.g., X1.4, M2.3) to peak soft X-ray intensity (W/m^2)."""
    if not isinstance(flare_class_str, str) or not flare_class_str:
        return 0.0
    flare_class_str = flare_class_str.strip().upper()
    match = re.match(r'([ABCMX])([0-9.]+)', flare_class_str)
    if match:
        letter = match.group(1)
        value = float(match.group(2))
        mapping = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}
        return value * mapping[letter]
    return 0.0

def download_solar_flares(data_dir):
    """Downloads solar flare data from NOAA NCEI GOES flare catalog (2021-2025)."""
    logger.info("Starting NOAA solar flare data collection (2021-2025)...")
    flares_list = []
    
    for year in range(2021, 2026):
        url = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/sci_xrsf-l2-flrpt_geo_y{year}_v1-0-0.csv"
        logger.info(f"Downloading solar flares for year {year} from: {url}")
        try:
            csv_bytes = download_url(url)
            # Save raw file
            raw_path = os.path.join(data_dir, f"solar_flares_{year}_raw.csv")
            with open(raw_path, 'wb') as f:
                f.write(csv_bytes)
            
            # Load into pandas
            df = pd.read_csv(raw_path)
            logger.info(f"Loaded {len(df)} flares for year {year}.")
            flares_list.append(df)
        except Exception as e:
            logger.error(f"Failed to download solar flares for year {year}: {e}")
            
    if not flares_list:
        raise ValueError("No solar flare data downloaded!")
        
    combined_df = pd.concat(flares_list, ignore_index=True)
    logger.info(f"Combined solar flares dataset contains {len(combined_df)} records.")
    return combined_df

def download_gfz_index(index_name, start_date, end_date):
    """Downloads Kp, ap, or Ap index from GFZ Potsdam JSON API."""
    url = f"https://kp.gfz-potsdam.de/app/json/?start={start_date}T00:00:00Z&end={end_date}T23:59:59Z&index={index_name}"
    logger.info(f"Downloading {index_name} index from GFZ Potsdam API: {url}")
    try:
        body = download_url(url)
        data = json.loads(body.decode('utf-8'))
        
        # GFZ JSON structure has 'datetime' and index_name
        datetimes = data.get('datetime', [])
        values = data.get(index_name, [])
        
        df = pd.DataFrame({
            'datetime': pd.to_datetime(datetimes),
            index_name: values
        })
        logger.info(f"Downloaded {len(df)} records for {index_name}.")
        return df
    except Exception as e:
        logger.error(f"Failed to download {index_name} from GFZ: {e}")
        raise e

def download_dst_kyoto(year, month):
    """Downloads and parses provisional Dst index for a given year and month from WDC Kyoto."""
    ym_str = f"{year}{month:02d}"
    url = f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{ym_str}/index.html"
    logger.debug(f"Fetching Kyoto Dst page: {url}")
    try:
        html_bytes = download_url(url)
        html = html_bytes.decode('utf-8', errors='ignore')
        
        # Find the <pre class="data"> tag contents
        pre_content = re.search(r'<pre\s+class="data">(.*?)</pre>', html, re.DOTALL | re.IGNORECASE)
        if not pre_content:
            # Try raw pre tag
            pre_content = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL | re.IGNORECASE)
            
        if not pre_content:
            raise ValueError(f"Could not find Dst table content in HTML for {year}-{month:02d}")
            
        lines = pre_content.group(1).split('\n')
        records = []
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if not parts:
                continue
                
            day_str = parts[0]
            if not day_str.isdigit():
                continue
            day = int(day_str)
            if day < 1 or day > 31:
                continue
                
            if len(line) < 99:
                continue
                
            for hour in range(1, 25):
                start_idx = 3 + (hour - 1) * 4
                val_str = line[start_idx:start_idx+4].strip()
                if not val_str or val_str in ['-', '9999', '999']:
                    dst_val = np.nan
                else:
                    try:
                        dst_val = float(val_str)
                    except ValueError:
                        dst_val = np.nan
                        
                dt = datetime(year, month, day, hour - 1)
                records.append({
                    'datetime': dt,
                    'Dst': dst_val
                })
        
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch Dst for {year}-{month:02d}: {e}")
        return pd.DataFrame(columns=['datetime', 'Dst'])

def download_all_dst(start_year, end_year):
    """Downloads Kyoto Dst index for 2021-2025 month-by-month."""
    logger.info("Starting WDC Kyoto Dst index collection...")
    dst_list = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            logger.info(f"Downloading Kyoto Dst for {year}-{month:02d}...")
            df_month = download_dst_kyoto(year, month)
            if not df_month.empty:
                dst_list.append(df_month)
                
    if not dst_list:
        raise ValueError("No Dst index data downloaded!")
        
    combined_dst = pd.concat(dst_list, ignore_index=True)
    logger.info(f"Combined Dst index contains {len(combined_dst)} records.")
    return combined_dst

def main():
    # Setup folders
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    start_year = 2021
    end_year = 2025
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"
    
    # Clean up old case-colliding files if they exist
    old_ap = os.path.join(data_dir, "ap_raw.csv")
    if os.path.exists(old_ap):
        try:
            os.remove(old_ap)
            logger.info("Removed old ap_raw.csv to clean case collision.")
        except Exception as e:
            logger.warning(f"Could not remove {old_ap}: {e}")
            
    # Clean up other potential old cache files to force a correct download of ap/Ap
    # Let's define the new unique filenames
    solar_flares_raw_path = os.path.join(data_dir, "solar_flares_raw.csv")
    kp_raw_path = os.path.join(data_dir, "kp_raw.csv")
    ap_3h_raw_path = os.path.join(data_dir, "ap_3h_raw.csv")
    Ap_daily_raw_path = os.path.join(data_dir, "Ap_daily_raw.csv")
    dst_raw_path = os.path.join(data_dir, "dst_raw.csv")
    
    cached = all(os.path.exists(p) for p in [solar_flares_raw_path, kp_raw_path, ap_3h_raw_path, Ap_daily_raw_path, dst_raw_path])
    
    if cached:
        logger.info("Loading raw datasets from local cache...")
        solar_flares_raw = pd.read_csv(solar_flares_raw_path)
        kp_df = pd.read_csv(kp_raw_path)
        ap_df = pd.read_csv(ap_3h_raw_path)
        Ap_df = pd.read_csv(Ap_daily_raw_path)
        dst_df = pd.read_csv(dst_raw_path)
    else:
        logger.info("Cache incomplete. Downloading raw datasets...")
        # 1. Download Solar Flares
        if os.path.exists(solar_flares_raw_path):
            solar_flares_raw = pd.read_csv(solar_flares_raw_path)
        else:
            solar_flares_raw = download_solar_flares(data_dir)
            solar_flares_raw.to_csv(solar_flares_raw_path, index=False)
            logger.info(f"Saved raw solar flare data to {solar_flares_raw_path}")
        
        # 2. Download Kp, ap, Ap from GFZ Potsdam
        if os.path.exists(kp_raw_path):
            kp_df = pd.read_csv(kp_raw_path)
        else:
            kp_df = download_gfz_index("Kp", start_str, end_str)
            kp_df.to_csv(kp_raw_path, index=False)
            
        ap_df = download_gfz_index("ap", start_str, end_str)
        ap_df.to_csv(ap_3h_raw_path, index=False)
        
        Ap_df = download_gfz_index("Ap", start_str, end_str)
        Ap_df.to_csv(Ap_daily_raw_path, index=False)
        
        # 3. Download Dst from Kyoto
        if os.path.exists(dst_raw_path):
            dst_df = pd.read_csv(dst_raw_path)
        else:
            dst_df = download_all_dst(start_year, end_year)
            dst_df.to_csv(dst_raw_path, index=False)
            
        logger.info("Saved all raw datasets to CSV.")
    
    # 4. Merge datasets into Hourly Time-Series
    logger.info("Merging datasets into hourly consolidated time-series...")
    
    # Create complete hourly date range
    dt_index = pd.date_range(start=f"{start_str} 00:00:00", end=f"{end_str} 23:00:00", freq='h')
    hourly_df = pd.DataFrame(index=dt_index)
    hourly_df.index.name = 'datetime'
    
    # Standardize datetime columns to timezone-naive UTC
    def standardize_dt(df, col='datetime'):
        dts = pd.to_datetime(df[col], utc=True)
        return dts.dt.tz_convert('UTC').dt.tz_localize(None)
    
    kp_df['datetime'] = standardize_dt(kp_df)
    ap_df['datetime'] = standardize_dt(ap_df)
    Ap_df['datetime'] = standardize_dt(Ap_df)
    dst_df['datetime'] = pd.to_datetime(dst_df['datetime']).dt.tz_localize(None)
    
    # Remove duplicates before merging
    kp_df = kp_df.drop_duplicates(subset=['datetime'])
    ap_df = ap_df.drop_duplicates(subset=['datetime'])
    Ap_df = Ap_df.drop_duplicates(subset=['datetime'])
    dst_df = dst_df.drop_duplicates(subset=['datetime'])
    
    # Merge hourly Dst
    hourly_df = hourly_df.merge(dst_df, on='datetime', how='left')
    
    # Merge 3-hourly Kp and ap using forward-fill
    hourly_df = hourly_df.merge(kp_df, on='datetime', how='left')
    hourly_df['Kp'] = hourly_df['Kp'].ffill().bfill()
    
    hourly_df = hourly_df.merge(ap_df, on='datetime', how='left')
    hourly_df['ap'] = hourly_df['ap'].ffill().bfill()
    
    # Merge daily Ap index
    Ap_df['date'] = Ap_df['datetime'].dt.date
    hourly_df['date'] = hourly_df['datetime'].dt.date
    hourly_df = hourly_df.merge(Ap_df[['date', 'Ap']], on='date', how='left')
    hourly_df.drop(columns=['date'], inplace=True)
    hourly_df['Ap'] = hourly_df['Ap'].ffill().bfill()
    
    # 5. Aggregate and Merge Solar Flares hourly
    logger.info("Aggregating solar flares to hourly resolution...")
    
    # Parse solar flare timestamps to tz-naive
    solar_flares_raw['time'] = pd.to_datetime(solar_flares_raw['time'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    solar_flares_raw['intensity'] = solar_flares_raw['flare_class'].apply(flare_class_to_intensity)
    
    solar_flares_raw['start_time'] = pd.to_datetime(solar_flares_raw['start_time'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    solar_flares_raw['end_time'] = pd.to_datetime(solar_flares_raw['end_time'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    
    solar_flares_raw['duration_min'] = (solar_flares_raw['end_time'] - solar_flares_raw['start_time']).dt.total_seconds() / 60.0
    solar_flares_raw['duration_min'] = solar_flares_raw['duration_min'].fillna(20.0)
    solar_flares_raw.loc[solar_flares_raw['duration_min'] < 0, 'duration_min'] = 20.0
    
    # Group solar flares by hour
    solar_flares_raw['hour_dt'] = solar_flares_raw['time'].dt.floor('h')
    
    # Aggregation
    flare_hourly = solar_flares_raw.groupby('hour_dt').agg(
        flare_count=('flare_id', 'count'),
        max_flare_intensity=('intensity', 'max'),
        total_flare_duration=('duration_min', 'sum')
    ).reset_index()
    
    flare_hourly.rename(columns={'hour_dt': 'datetime'}, inplace=True)
    
    # Merge flares into hourly DataFrame
    hourly_df = hourly_df.merge(flare_hourly, on='datetime', how='left')
    
    # Fill missing flare values with 0
    hourly_df['flare_count'] = hourly_df['flare_count'].fillna(0).astype(int)
    hourly_df['max_flare_intensity'] = hourly_df['max_flare_intensity'].fillna(0.0)
    hourly_df['total_flare_duration'] = hourly_df['total_flare_duration'].fillna(0.0)
    
    # Handle any remaining missing values in geomagnetic indices
    hourly_df['Dst'] = hourly_df['Dst'].interpolate(method='linear').bfill().ffill()
    hourly_df['Kp'] = hourly_df['Kp'].interpolate(method='linear').bfill().ffill()
    hourly_df['ap'] = hourly_df['ap'].interpolate(method='linear').bfill().ffill()
    hourly_df['Ap'] = hourly_df['Ap'].interpolate(method='linear').bfill().ffill()
    
    # Save consolidated dataset
    output_path = os.path.join(data_dir, "space_weather_hourly.csv")
    hourly_df.to_csv(output_path, index=False)
    logger.info(f"Consolidated hourly dataset saved to: {output_path}")
    logger.info(f"Consolidated data contains {len(hourly_df)} records.")
    
if __name__ == "__main__":
    main()
