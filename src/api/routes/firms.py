"""
src/api/routes/firms.py

NASA FIRMS endpoints for live active fire tracking.
"""

from fastapi import APIRouter, Query, BackgroundTasks
from loguru import logger
import asyncio

from src.data.firms_sync import sync_firms_data, get_firms_geojson
from src.api.json_utils import json_response
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class RegionQuery(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

@router.get("/api/firms/alerts", tags=["FIRMS"])
async def get_firms_alerts(
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    hours: int = Query(24, description="Hours of recency")
):
    """
    Get live active fire alerts as a GeoJSON FeatureCollection.
    """
    region = None
    if all(x is not None for x in [lat_min, lat_max, lon_min, lon_max]):
        region = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max
        }
        
    # We load off-thread since pandas reads CSV
    geojson = await asyncio.to_thread(get_firms_geojson, region, hours)
    return json_response(geojson)

@router.post("/api/firms/sync", tags=["FIRMS"])
async def trigger_firms_sync(background_tasks: BackgroundTasks):
    """
    Manually trigger a sync with the NASA FIRMS API.
    Runs as a background task.
    """
    background_tasks.add_task(sync_firms_data)
    return {"status": "sync_queued", "message": "FIRMS data sync started in the background."}
