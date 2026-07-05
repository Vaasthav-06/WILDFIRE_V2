"""
src/api/routes/permits.py

Endpoints for processing fire permit applications and checking zone status.
Fixes the bug where `app.state.engine` was incorrectly used instead of `_eng`.
"""

import time
import pandas as pd
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from src.api.dependencies import require_engine

router = APIRouter()


class PermitApplication(BaseModel):
    officer_name: str
    officer_id: str
    zone_name: str
    latitude: float
    longitude: float
    activity_type: str
    start_date: str
    end_date: str
    area_hectares: float
    has_fire_extinguisher: bool
    has_insurance: bool
    officer_trained: bool


@router.get("/api/permits/check-zone", tags=["Permits"])
async def check_permit_zone(
    lat: float = Query(...),
    lon: float = Query(...),
):
    """
    Check the current wildfire risk for a specific zone before approving a permit.
    Returns the risk assessment.
    """
    require_engine()
    from src.api.dependencies import engine as _eng
    
    t0 = time.time()
    
    # 1. Fetch risk from the engine
    grid_df = pd.DataFrame([{"latitude": lat, "longitude": lon}])
    scored = _eng.predict_now(grid_df=grid_df)
    
    if len(scored) == 0:
        raise HTTPException(status_code=404, detail="Zone data unavailable.")
        
    fire_prob = float(scored["fire_prob"].iloc[0])
    risk_tier = str(scored["risk_tier"].iloc[0])
    temp = float(scored["temp"].iloc[0]) if "temp" in scored.columns else None
    wind = float(scored["wind"].iloc[0]) if "wind" in scored.columns else None
    
    # 2. Basic rules Engine
    can_approve = True
    reasons = []
    
    if fire_prob > 0.40:
        can_approve = False
        reasons.append(f"High baseline fire probability ({fire_prob:.2f}).")
    if temp and temp > 35.0:
        can_approve = False
        reasons.append(f"Extreme temperature ({temp:.1f}°C).")
    if wind and wind > 8.0:
        can_approve = False
        reasons.append(f"High wind speed ({wind:.1f} m/s).")
        
    response = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 2),
        "latitude": lat,
        "longitude": lon,
        "assessment": {
            "fire_prob": round(fire_prob, 3),
            "risk_tier": risk_tier,
            "temp_c": round(temp, 1) if temp else None,
            "wind_ms": round(wind, 1) if wind else None,
            "can_approve": can_approve,
            "rejection_reasons": reasons if not can_approve else []
        }
    }
    return response


@router.post("/api/permits/apply", tags=["Permits"])
async def apply_permit(application: PermitApplication):
    """
    Submit a permit application.
    Validates the application against the current ML risk model.
    """
    require_engine()
    from src.api.dependencies import engine as _eng
    
    # 1. Check ML Risk
    grid_df = pd.DataFrame([{
        "latitude": application.latitude,
        "longitude": application.longitude
    }])
    scored = _eng.predict_now(grid_df=grid_df)
    
    if len(scored) == 0:
        raise HTTPException(status_code=404, detail="Coordinates outside supported grid.")
        
    fire_prob = float(scored["fire_prob"].iloc[0])
    
    # 2. Validate Application Requirements
    rejections = []
    if fire_prob > 0.50:
        rejections.append("ML model indicates critical fire risk at this location.")
    if application.activity_type in ["burn", "extraction"] and fire_prob > 0.25:
        rejections.append("High-risk activity not permitted under current moderate fire risk.")
    if application.area_hectares > 50 and not application.has_fire_extinguisher:
        rejections.append("Large scale activity requires fire extinguishers.")
    if application.activity_type == "tourism" and not application.officer_trained:
        rejections.append("Tourism activities require trained officers.")
        
    status = "REJECTED" if rejections else "APPROVED"
    
    return {
        "application_id": f"PRMT-{int(time.time())}",
        "status": status,
        "fire_prob_at_time": round(fire_prob, 3),
        "rejections": rejections
    }
