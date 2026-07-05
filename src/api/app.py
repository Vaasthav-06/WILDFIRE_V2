"""
src/api/app.py

FastAPI entry point for the Wildfire Risk Prediction System.
This file has been decomposed. All endpoints now live in src/api/routes/.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add root directory to path so absolute imports work
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.dependencies import load_all
from src.api.routes import predictions, analytics, chatbot, permits, firms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    # Startup
    logger.info("Starting Wildfire Risk API...")
    
    # Load ML models, SHAP explainer, and EVI lookup into memory
    # We do this asynchronously to prevent blocking the event loop on slow I/O
    import asyncio
    await asyncio.to_thread(load_all)
    
    logger.info("Application startup complete.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Wildfire Risk API...")


app = FastAPI(
    title="Wildfire Risk Prediction API",
    version="2.0.0",
    description="Decomposed ML microservice for wildfire risk and analytics.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    from src.api.dependencies import engine, why_engine
    from src.api.cache import prediction_cache

    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "explainer_loaded": why_engine is not None,
        "cache_state": prediction_cache.state.value,
        "version": "2.0.0"
    }


# Mount routers
app.include_router(predictions.router)
app.include_router(analytics.router)
app.include_router(chatbot.router)
app.include_router(permits.router)
app.include_router(firms.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)