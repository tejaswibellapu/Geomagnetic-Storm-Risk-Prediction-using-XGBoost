import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.risk_engine import RiskEngine
from src.insight_generator import InsightGenerator

app = FastAPI(
    title="Solar Activity and Geomag Risk Score Analysis API",
    description="API exposing solar activity correlations, geomagnetic risk scores, and ionospheric insights.",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "xgboost_risk_model.pkl")
hourly_path = os.path.join(base_dir, "data", "space_weather_features.csv")
flares_path = os.path.join(base_dir, "data", "solar_flares_raw.csv")
events_path = os.path.join(base_dir, "data", "significant_events.csv")

# Initialize modules
risk_engine = RiskEngine(model_path, hourly_path, flares_path)
insight_gen = InsightGenerator(model_path, hourly_path, flares_path, events_path)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "description": "Solar Activity and Geomag Risk Score Analysis API",
        "supported_range": "2021-2025"
    }

@app.get("/api/risk")
def get_monthly_risk(
    year: int = Query(..., ge=2021, le=2025, description="Year to query (2021-2025)"),
    month: int = Query(..., ge=1, le=12, description="Month to query (1-12)")
):
    """Returns observed and predicted geomagnetic risk scores for the selected month."""
    try:
        result = risk_engine.evaluate_month(year, month)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights")
def get_monthly_insights(
    year: int = Query(..., ge=2021, le=2025, description="Year to query (2021-2025)"),
    month: int = Query(..., ge=1, le=12, description="Month to query (1-12)")
):
    """Returns the markdown executive summary report for the selected month."""
    try:
        report = insight_gen.generate_report(year, month)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
