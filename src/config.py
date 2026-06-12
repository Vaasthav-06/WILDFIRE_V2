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
RESULTS_DIR = ROOT_DIR / os.getenv("RESULTS_DIR", "results")
LOGS_DIR = ROOT_DIR / "logs"

# ── Ensure dirs exist ────────────────────────────────────────
for _dir in [
    DATA_RAW,
    DATA_PROCESSED,
    DATA_CACHE,
    MODELS_DIR,
    RESULTS_DIR / "figures",
    RESULTS_DIR / "metrics",
    LOGS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── API ──────────────────────────────────────────────────────
NASA_FIRMS_API_KEY = os.getenv("NASA_FIRMS_API_KEY", "")
OPEN_METEO_BASE_URL = os.getenv(
    "OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast"
)
OPEN_METEO_ARCHIVE_URL = os.getenv(
    "OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive"
)

# ── India bounding box ───────────────────────────────────────
INDIA_BOUNDS = {
    "lat_min": 8.0,
    "lat_max": 37.6,
    "lon_min": 68.1,
    "lon_max": 97.4,
}

# ── Feature columns (single definition, imported everywhere) ─
# Removed: elevation (proxy, near-zero variance)
#          dist_road_km (proxy, std=1.5km across continent — noise not signal)
FEATURE_COLS = [
    # Ecoregion — replaces raw lat/lon to prevent geographic memorisation
    "eco_tropical_moist",
    "eco_tropical_dry",
    "eco_semi_arid",
    "eco_montane",
    "eco_subtropical",
    # Temporal
    "month_sin",
    "month_cos",
    # Weather (ERA5 reanalysis) — primary fire signal
    "temp",
    "humidity",
    "wind",
    "vpd",
    # Drought accumulation — KBDI approximated from ERA5 precip + temp
    "kbdi_approx",
    # Vegetation — MODIS EVI (Enhanced Vegetation Index) 2018-2025
    "evi",
    # Physics-informed interactions
    "vpd_wind",
    "temp_dryness",
]

TARGET_COL = "fire_detected"

# ── Model ────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))

# ── Inference grid ───────────────────────────────────────────
GRID_RESOLUTION = 0.5  # degrees
RISK_THRESHOLD_HIGH = 0.50
RISK_THRESHOLD_CRITICAL = 0.80

# ── Training ─────────────────────────────────────────────────
CV_SAMPLE_SIZE = 200_000  # rows used for spatial CV (speed)
SHAP_SAMPLE = 5_000       # rows for SHAP computation