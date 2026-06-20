import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from src.risk_engine import RiskEngine




# Page config
st.set_page_config(
    page_title="Solar Activity and Geomagnetic Risk Score Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium White Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Force Light Theme Backgrounds */
    .stApp {
        background-color: #FAFAFA !important;
        color: #1F2937 !important;
        font-family: 'Outfit', sans-serif;
        font-size: 18px !important;
    }
    
    /* Remove default Streamlit header and padding */
    header {
        visibility: hidden !important;
        height: 0px !important;
        display: none !important;
    }
    
    .stApp > header {
        display: none !important;
    }
    
    div[data-testid="stDecoration"] {
        display: none !important;
        height: 0px !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
    }
    
    /* Global font size increases */
    html, body, p, [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] label, div[data-baseweb="select"] {
        font-size: 18px !important;
    }
    
    /* Increase headings by +2px approx */
    h1 { font-size: 2.75rem !important; }
    h2 { font-size: 2.25rem !important; }
    h3 { font-size: 1.75rem !important; }
    h4 { font-size: 1.5rem !important; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F3F4F6 !important;
        border-right: 1px solid #E5E7EB !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
    }

    
    /* Sidebar text/labels */
    [data-testid="stSidebar"] label {
        color: #374151 !important;
        font-weight: 600;
    }

    /* Selectbox style overrides */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] input {
        color: #1F2937 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #FFFFFF !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: #1F2937 !important;
    }
    
    /* Card design */
    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .metric-val-flares {
        font-size: 2.75rem;
        font-weight: 800;
        color: #D97706; /* High contrast amber on white */
    }
    
    .metric-val-kp {
        font-size: 2.75rem;
        font-weight: 800;
        color: #2563EB; /* Deep Royal Blue on white */
    }
    
    .metric-val-tec {
        font-size: 2.75rem;
        font-weight: 800;
        color: #7C3AED; /* Violet on white */
    }
    
    .metric-lbl {
        font-size: 0.975rem;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.25rem;
    }

    /* Titles */
    h2, h3, h4 {
        color: #1F2937 !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "xgboost_risk_model.pkl")
hourly_path = os.path.join(base_dir, "data", "space_weather_features.csv")
flares_path = os.path.join(base_dir, "data", "solar_flares_raw.csv")

# Initialize Engine
@st.cache_resource
def init_modules():
    return RiskEngine(model_path, hourly_path, flares_path)

risk_engine = init_modules()

# Helper function to generate subpoint and auroral zone locations
def generate_impact_map(year, month):
    map_data = []
    
    # 1. Load flares and calculate solar subpoint (dayside)
    if os.path.exists(flares_path):
        df_flares = pd.read_csv(flares_path)
        df_flares['time'] = pd.to_datetime(df_flares['time'])
        month_flares = df_flares[
            (df_flares['time'].dt.year == year) & 
            (df_flares['time'].dt.month == month)
        ].copy()
        
        for _, row in month_flares.iterrows():
            dt = row['time']
            # Day of year declination
            day_of_year = dt.timetuple().tm_yday
            dec = 23.44 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            # Solar subpoint longitude
            utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            lon = -(utc_hour - 12.0) * 15.0
            lon = (lon + 180) % 360 - 180
            
            f_class = str(row['flare_class'])
            intensity = 2.0
            if f_class.startswith('X'):
                intensity = 8.0
            elif f_class.startswith('M'):
                intensity = 5.0
            elif f_class.startswith('C'):
                intensity = 3.0
                
            map_data.append({
                'Latitude': dec,
                'Longitude': lon,
                'Impact Type': 'Solar Flare Radio Blackout (Dayside)',
                'Intensity': intensity,
                'Details': f"Flare Class: {f_class} at {dt.strftime('%d %b %H:%M UTC')}"
            })
            
    # 2. Load hourly data for geomagnetic storms (Kp >= 5) (nightside magnetic midnight)
    if os.path.exists(hourly_path):
        df_hourly = pd.read_csv(hourly_path)
        df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
        month_hourly = df_hourly[
            (df_hourly['datetime'].dt.year == year) & 
            (df_hourly['datetime'].dt.month == month) &
            (df_hourly['Kp'] >= 5.0)
        ].copy()
        
        for _, row in month_hourly.iterrows():
            dt = row['datetime']
            utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            lon_sun = -(utc_hour - 12.0) * 15.0
            # Magnetic midnight (opposite side of Earth)
            lon_aurora = lon_sun + 180.0
            lon_aurora = (lon_aurora + 180) % 360 - 180
            
            kp_val = row['Kp']
            intensity = (kp_val - 4.0) * 3.0
            
            # Add Northern Auroral Oval Point
            map_data.append({
                'Latitude': 65.0,
                'Longitude': lon_aurora,
                'Impact Type': 'Geomagnetic Storm & Aurora (Nightside)',
                'Intensity': intensity,
                'Details': f"Kp Index: {kp_val} at {dt.strftime('%d %b %H:%M UTC')}"
            })
            # Add Southern Auroral Oval Point
            map_data.append({
                'Latitude': -65.0,
                'Longitude': lon_aurora,
                'Impact Type': 'Geomagnetic Storm & Aurora (Nightside)',
                'Intensity': intensity,
                'Details': f"Kp Index: {kp_val} at {dt.strftime('%d %b %H:%M UTC')}"
            })
            
    if map_data:
        return pd.DataFrame(map_data)
    else:
        return pd.DataFrame(columns=['Latitude', 'Longitude', 'Impact Type', 'Intensity', 'Details'])

# ----------------- SIDEBAR (Control Panel) -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #1F2937; font-weight: 800; margin-bottom: 0.75rem;'>Enter Year and Month</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='height: 1px; background: rgba(0,0,0,0.1); margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

# Month/Year Selection
year = st.sidebar.selectbox("Select Year", options=[2021, 2022, 2023, 2024, 2025], index=3) # Default 2024
month = st.sidebar.selectbox("Select Month", options=list(range(1, 13)), format_func=lambda m: pd.to_datetime(f"2024-{m:02d}-01").strftime('%B'), index=2) # Default March

# Fetch monthly stats
risk_report = risk_engine.evaluate_month(year, month)



# ----------------- MAIN VIEW -----------------
st.markdown("<h2 style='margin-top: 0px; margin-bottom: 1.25rem; color: #1F2937;'> Solar Activity and Geomagnetic Risk Score Analysis</h2>", unsafe_allow_html=True)

# 1. Two side-by-side Gauges (Geomagnetic Risk vs Solar Flare Activity)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='text-align: center; margin-bottom: 0.5rem;'>Geomagnetic Risk Index</h4>", unsafe_allow_html=True)
    
    geo_level = risk_report['observed_category']
    geo_impact = "Quiet magnetosphere."
    if geo_level == "Moderate":
        geo_impact = "Minor grid and satellite orbit effects."
    elif geo_level == "High":
        geo_impact = "Elevated grid alarms and GPS fluctuations."
    elif geo_level == "Extreme":
        geo_impact = "Severe grid fluctuations and aurora expansions."

    # Observed Gauge Plotly
    fig_obs = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_report['observed_score'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<span style='font-size:1.25em; color:#2563EB; font-weight:bold;'>Level: {geo_level}</span><br><span style='font-size:0.975em;color:#4B5563'>Impact: {geo_impact}</span>", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#4B5563"},
            'bar': {'color': "#2563EB"}, # Deep Royal Blue
            'bgcolor': "rgba(0,0,0,0.03)",
            'borderwidth': 1,
            'bordercolor': "#D1D5DB",
            'steps': [
                {'range': [0, 3], 'color': 'rgba(16, 185, 129, 0.1)'},
                {'range': [3, 6], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [6, 9], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [9, 10], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3},
                'thickness': 0.75,
                'value': risk_report['observed_score']
            }
        }
    ))
    fig_obs.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font={'color': "#1F2937", 'family': "Outfit"}, 
        height=320, 
        margin=dict(t=80, b=40, l=40, r=40)
    )
    st.plotly_chart(fig_obs, use_container_width=True)
    
with col2:
    st.markdown("<h4 style='text-align: center; margin-bottom: 0.5rem;'>Solar Flare Activity</h4>", unsafe_allow_html=True)
    
    flare_level = risk_report['predicted_category']
    flare_impact = "No major eruptions."
    if flare_level == "Moderate":
        flare_impact = "Minor solar flare activity."
    elif flare_level == "High":
        flare_impact = "Strong M or X class solar flares."
    elif flare_level == "Extreme":
        flare_impact = "High energy solar flares, CMEs predicted."

    # Predicted Gauge Plotly
    fig_pred = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_report['predicted_score'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<span style='font-size:1.25em; color:#D97706; font-weight:bold;'>Level: {flare_level}</span><br><span style='font-size:0.975em;color:#4B5563'>Impact: {flare_impact}</span>", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#4B5563"},
            'bar': {'color': "#D97706"}, # Amber/Orange
            'bgcolor': "rgba(0,0,0,0.03)",
            'borderwidth': 1,
            'bordercolor': "#D1D5DB",
            'steps': [
                {'range': [0, 3], 'color': 'rgba(16, 185, 129, 0.1)'},
                {'range': [3, 6], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [6, 9], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [9, 10], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3},
                'thickness': 0.75,
                'value': risk_report['predicted_score']
            }
        }
    ))
    fig_pred.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font={'color': "#1F2937", 'family': "Outfit"}, 
        height=320, 
        margin=dict(t=80, b=40, l=40, r=40)
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
# 2. Main Metrics Block (Total Flares, Max Kp, Observed TEC, Predicted TEC)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-lbl'>Total Flares Detected</div>
        <div class='metric-val-flares'>{risk_report['summary_stats'].get('flare_count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-lbl'>Max Kp Index</div>
        <div class='metric-val-kp'>{risk_report['summary_stats'].get('max_kp', 0.0):.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-lbl'>Observed Max TEC</div>
        <div class='metric-val-tec'>{risk_report.get('observed_tec', 0.0):.1f} <span style="font-size: 1.125rem; color: #6B7280; font-weight: 600;">TECU</span></div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-lbl'>Predicted Max TEC (3h)</div>
        <div class='metric-val-tec' style='color: #059669;'>{risk_report.get('predicted_tec', 0.0):.1f} <span style="font-size: 1.125rem; color: #6B7280; font-weight: 600;">TECU</span></div>
    </div>
    """, unsafe_allow_html=True)

# 3. Global Impact Map Section
st.markdown("<h2 style='margin-top: 3rem; color: #1F2937; font-size: 1.625rem !important;'>  Impact Map</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.025rem; color:#4B5563; margin-bottom:1rem;'>Visualizing solar flare radio blackouts (dayside subpoint) and geomagnetic storm auroral zones (nightside magnetic midnight) for the selected month.</p>", unsafe_allow_html=True)

df_map = generate_impact_map(year, month)
if not df_map.empty:
    fig_map = px.scatter_geo(
        df_map,
        lat='Latitude',
        lon='Longitude',
        color='Impact Type',
        size='Intensity',
        hover_name='Details',
        projection='natural earth',
        color_discrete_map={
            'Solar Flare Radio Blackout (Dayside)': '#EA580C', # Deep Orange
            'Geomagnetic Storm & Aurora (Nightside)': '#2563EB' # Royal Blue
        },
        size_max=15
    )
    fig_map.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1, 
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(color="#1F2937", family="Outfit", size=14),
            title=dict(font=dict(color="#1F2937", family="Outfit", size=14))
        ),
        height=450,
        font={'color': "#1F2937", 'family': "Outfit", 'size': 14}
    )
    fig_map.update_geos(
        showland=True, landcolor='#E5E7EB',
        showocean=True, oceancolor='#EFF6FF',
        showlakes=True, lakecolor='#EFF6FF',
        showcountries=True, countrycolor='#9CA3AF',
        coastlinecolor='#9CA3AF',
        resolution=110
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No major solar flare or geomagnetic storm events recorded for this month.")