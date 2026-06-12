"""
scripts/download_era5.py
One-time ERA5 download for India, 2024-2025.

Usage:
    python scripts/download_era5.py
"""

from loguru import logger
from src.data.era5 import download_era5

logger.add("logs/era5_download.log", rotation="10 MB")

# scripts/download_era5.py — change the bottom section to:
if __name__ == "__main__":
    logger.info("Starting ERA5 download for 2018-2023")
    paths = download_era5(years=[2018, 2019, 2020, 2021, 2022, 2023])
    logger.info(f"Done. Files: {paths}")
