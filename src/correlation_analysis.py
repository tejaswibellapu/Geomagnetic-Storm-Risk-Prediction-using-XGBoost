import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "correlation_analysis.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CorrelationAnalysis")

def run_correlation_analysis(event_aligned_path, hourly_path, outputs_dir):
    """Performs Pearson, Spearman, and lag correlation analysis and saves results."""
    logger.info("Starting correlation analysis...")
    os.makedirs(outputs_dir, exist_ok=True)
    
    if not os.path.exists(event_aligned_path):
        raise FileNotFoundError(f"Event-aligned data not found: {event_aligned_path}")
    if not os.path.exists(hourly_path):
        raise FileNotFoundError(f"Hourly data not found: {hourly_path}")
        
    df_aligned = pd.read_csv(event_aligned_path)
    df_hourly = pd.read_csv(hourly_path)
    
    # --- 1. Pearson & Spearman Correlation (Event-level) ---
    logger.info("Computing Pearson and Spearman correlation matrices...")
    
    # Core variables for event correlation
    core_cols = [
        'flare_intensity', 'duration_min',
        'Kp_max_24h', 'Dst_min_24h', 'ap_max_24h', 'Ap_max_24h', 'TEC_max_24h'
    ]
    
    pearson_matrix = df_aligned[core_cols].corr(method='pearson')
    spearman_matrix = df_aligned[core_cols].corr(method='spearman')
    
    # Save matrices to CSV
    pearson_matrix.to_csv(os.path.join(outputs_dir, "pearson_matrix.csv"))
    spearman_matrix.to_csv(os.path.join(outputs_dir, "spearman_matrix.csv"))
    
    # Plot and save Heatmaps
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Pearson Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pearson_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Pearson Correlation Matrix (Event-level, 24h Lag)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "pearson_heatmap.png"), dpi=150)
    plt.close()
    
    # Spearman Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Spearman Correlation Matrix (Event-level, 24h Lag)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "spearman_heatmap.png"), dpi=150)
    plt.close()
    
    # --- 2. Lag Cross-Correlation ---
    logger.info("Computing lag cross-correlations across windows...")
    
    lags = [6, 12, 24, 48, 72]
    kp_corrs_pearson = []
    dst_corrs_pearson = []
    kp_corrs_spearman = []
    dst_corrs_spearman = []
    
    for lag in lags:
        # Kp max correlation
        kp_p = df_aligned['flare_intensity'].corr(df_aligned[f'Kp_max_{lag}h'], method='pearson')
        kp_s = df_aligned['flare_intensity'].corr(df_aligned[f'Kp_max_{lag}h'], method='spearman')
        kp_corrs_pearson.append(kp_p)
        kp_corrs_spearman.append(kp_s)
        
        # Dst min correlation
        dst_p = df_aligned['flare_intensity'].corr(df_aligned[f'Dst_min_{lag}h'], method='pearson')
        dst_s = df_aligned['flare_intensity'].corr(df_aligned[f'Dst_min_{lag}h'], method='spearman')
        dst_corrs_pearson.append(dst_p)
        dst_corrs_spearman.append(dst_s)
        
    lag_corr_df = pd.DataFrame({
        'Lag_Window_Hours': lags,
        'Kp_Pearson': kp_corrs_pearson,
        'Kp_Spearman': kp_corrs_spearman,
        'Dst_Pearson': dst_corrs_pearson,
        'Dst_Spearman': dst_corrs_spearman
    })
    lag_corr_df.to_csv(os.path.join(outputs_dir, "lag_cross_correlation.csv"), index=False)
    
    # Plot Lag Correlation Graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Kp
    ax1.plot(lags, kp_corrs_pearson, marker='o', linestyle='-', color='red', label='Pearson')
    ax1.plot(lags, kp_corrs_spearman, marker='s', linestyle='--', color='darkred', label='Spearman')
    ax1.set_title('Solar Flare Intensity vs Kp Max Correlation by Lag Window')
    ax1.set_xlabel('Lag Window (Hours)')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_xticks(lags)
    ax1.legend()
    ax1.grid(True)
    
    # Dst
    ax2.plot(lags, dst_corrs_pearson, marker='o', linestyle='-', color='blue', label='Pearson')
    ax2.plot(lags, dst_corrs_spearman, marker='s', linestyle='--', color='darkblue', label='Spearman')
    ax2.set_title('Solar Flare Intensity vs Dst Min Correlation by Lag Window')
    ax2.set_xlabel('Lag Window (Hours)')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_xticks(lags)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "lag_correlation_chart.png"), dpi=150)
    plt.close()
    
    # --- 3. Monthly Correlation Report ---
    logger.info("Generating monthly correlation reports...")
    
    df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
    df_hourly['year'] = df_hourly['datetime'].dt.year
    df_hourly['month'] = df_hourly['datetime'].dt.month
    
    monthly_reports = {}
    
    for (year, month), group in df_hourly.groupby(['year', 'month']):
        key = f"{year}-{month:02d}"
        
        # Pearson and Spearman correlation between solar flare count/max intensity and geomagnetic/ionospheric variables
        kp_flare_corr_p = group['max_flare_intensity'].corr(group['Kp'], method='pearson')
        dst_flare_corr_p = group['max_flare_intensity'].corr(group['Dst'], method='pearson')
        tec_flare_corr_p = group['max_flare_intensity'].corr(group['TEC'], method='pearson')
        kp_flare_corr_s = group['max_flare_intensity'].corr(group['Kp'], method='spearman')
        dst_flare_corr_s = group['max_flare_intensity'].corr(group['Dst'], method='spearman')
        tec_flare_corr_s = group['max_flare_intensity'].corr(group['TEC'], method='spearman')
        
        # Mean stats
        mean_kp = group['Kp'].mean()
        min_dst = group['Dst'].min()
        mean_dst = group['Dst'].mean()
        mean_tec = group['TEC'].mean()
        max_tec = group['TEC'].max()
        num_flares = group['flare_count'].sum()
        max_flare_int = group['max_flare_intensity'].max()
        
        monthly_reports[key] = {
            'year': int(year),
            'month': int(month),
            'num_flares': int(num_flares),
            'max_flare_intensity': float(max_flare_int),
            'mean_kp': float(mean_kp),
            'min_dst': float(min_dst),
            'mean_dst': float(mean_dst),
            'mean_tec': float(mean_tec),
            'max_tec': float(max_tec),
            'kp_flare_intensity_pearson': float(kp_flare_corr_p) if pd.notnull(kp_flare_corr_p) else 0.0,
            'dst_flare_intensity_pearson': float(dst_flare_corr_p) if pd.notnull(dst_flare_corr_p) else 0.0,
            'tec_flare_intensity_pearson': float(tec_flare_corr_p) if pd.notnull(tec_flare_corr_p) else 0.0,
            'kp_flare_intensity_spearman': float(kp_flare_corr_s) if pd.notnull(kp_flare_corr_s) else 0.0,
            'dst_flare_intensity_spearman': float(dst_flare_corr_s) if pd.notnull(dst_flare_corr_s) else 0.0,
            'tec_flare_intensity_spearman': float(tec_flare_corr_s) if pd.notnull(tec_flare_corr_s) else 0.0,
        }
        
    report_path = os.path.join(outputs_dir, "monthly_correlation_report.json")
    with open(report_path, 'w') as f:
        json.dump(monthly_reports, f, indent=4)
        
    logger.info(f"Saved monthly correlation reports to: {report_path}")
    logger.info("Correlation analysis complete.")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    event_aligned_path = os.path.join(data_dir, "event_aligned_data.csv")
    hourly_path = os.path.join(data_dir, "space_weather_processed.csv")
    
    run_correlation_analysis(event_aligned_path, hourly_path, outputs_dir)

if __name__ == "__main__":
    main()
