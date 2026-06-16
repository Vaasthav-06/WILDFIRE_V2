from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Root paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATA_CACHE = ROOT_DIR / os.getenv("DATA_CACHE_DIR", "data/cache")
MODELS_DIR = ROOT_DIR / os.getenv("MODELS_DIR", "models")

# ── New Artifact Outputs ─────────────────────────────────────
OUTPUTS_DIR = ROOT_DIR / "outputs"
EVAL_DIR = OUTPUTS_DIR / "model_evaluation"
SHAP_DIR = OUTPUTS_DIR / "shap_analysis"
LOGS_DIR = ROOT_DIR / "logs"

# ── Ensure dirs exist ────────────────────────────────────────
for _dir in [
    DATA_RAW,
    DATA_PROCESSED,
    DATA_CACHE,
    MODELS_DIR,
    OUTPUTS_DIR,
    EVAL_DIR,
    SHAP_DIR,
    LOGS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ... [Keep your existing API, INDIA_BOUNDS, FEATURE_COLS, and Model constants here] ...

# ── API ──────────────────────────────────────────────────────
NASA_FIRMS_API_KEY = os.getenv("NASA_FIRMS_API_KEY", "")
OPEN_METEO_BASE_URL = os.getenv("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast")
OPEN_METEO_ARCHIVE_URL = os.getenv("OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive")

# ── India bounding box ───────────────────────────────────────
INDIA_BOUNDS = {
    "lat_min": 8.0,
    "lat_max": 37.6,
    "lon_min": 68.1,
    "lon_max": 97.4,
}

# ── Feature columns (Real Data Only) ─────────────────────────
FEATURE_COLS = [
    # Temporal
    "month_sin",
    "month_cos",
    
    # Weather (ERA5)
    "temp",
    "humidity",
    "wind",
    "vpd",
    
    # Vegetation (MODIS)
    "evi",
    
    # Physics-informed interactions
    "vpd_wind",
    
    # Geospatial (Real Satellite Data)
    "elevation_m",
    "lc_tree_cover",
    "lc_shrubland",
    "lc_grassland",
    "lc_cropland",
    "lc_built_up",
    "lc_sparse_veg",
    "lc_snow_ice",
    "lc_water",
    "lc_herbaceous_wetland",
    "lc_mangroves",
    "lc_moss_lichen"
]

TARGET_COL = "fire_detected"

# ── Model ────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))

# ── Inference grid ───────────────────────────────────────────
GRID_RESOLUTION = 0.5  
RISK_THRESHOLD_HIGH = 0.50
RISK_THRESHOLD_CRITICAL = 0.80

# ── Training ─────────────────────────────────────────────────
CV_SAMPLE_SIZE = 200_000  
SHAP_SAMPLE = 5_000