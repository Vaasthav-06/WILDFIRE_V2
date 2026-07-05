"""
src/api/dependencies.py

Module-level singletons for the FastAPI application.
Loaded once at startup; shared across all route modules.
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.config import MODELS_DIR, RISK_THRESHOLD_HIGH
from src.inference.engine import WildfireInferenceEngine
from src.explainability.why_engine import WhyEngine
from src.data.evi import EVILookup


# ── Singletons ────────────────────────────────────────────────────────────────

engine: Optional[WildfireInferenceEngine] = None
why_engine: Optional[WhyEngine] = None
evi_lookup: Optional[EVILookup] = None
startup_time: Optional[datetime] = None


def load_all():
    """Initialize all singletons. Called once from app startup."""
    global engine, why_engine, evi_lookup, startup_time

    startup_time = datetime.utcnow()
    logger.info("Loading wildfire risk prediction model...")

    try:
        engine = WildfireInferenceEngine(grid_resolution=0.1)
        why_engine = WhyEngine(
            model=engine.model,
            alert_threshold=RISK_THRESHOLD_HIGH,
            top_n_reasons=3,
            uncertainty_threshold=0.15,
        )
        evi_lookup = EVILookup()
        logger.info(f"Model loaded successfully: {engine.summary()}")
    except FileNotFoundError as e:
        logger.error(
            f"production_model.pkl not found: {e}. "
            "Run scripts/train_models.py first to generate the model file."
        )
        engine = None
        why_engine = None


def require_engine():
    """Raise 503 if the model is not loaded."""
    from fastapi import HTTPException

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Run scripts/train_models.py to generate production_model.pkl, "
                "then restart the API server."
            ),
        )
