# 🔥 WILDFIRE_REBORN

A machine learning pipeline that predicts where wildfires are likely to happen across India — before they do. By learning from 8 years of satellite and climate data, it gives communities and responders an early warning on fire risk.

**Test Temporal AUC (2025 Unseen Data): 0.9611 | Top Model: XGBoost**

## 🎯 What Does This Project Do?

We want to help people stay prepared. Given today's weather conditions and local topography, the platform:

- Predicts next-day wildfire probability at **0.1° grid resolution** across India.
- Outputs a **risk score (0–1)** per grid cell, a **risk tier** (Low / Moderate / High / Extreme), and an indicative **fire spread direction vector** for high-risk zones.
- Explains *why* an area is at risk using a SHAP-powered natural language engine (The **WHY Engine**).
- Plots everything on an **interactive React dashboard** so the danger zones are easy to understand at a glance.

## 📊 Where Do We Get Our Data?

To train our models, we used high-fidelity, physics-based information from trusted scientific sources:

| Source | What it tells us | Coverage |
|---|---|---|
| NASA FIRMS (VIIRS S-NPP C2) | Exactly when and where past wildfires happened | 2018–2025 |
| ERA5 (ECMWF via CDS) | Temperature, humidity, wind, and VPD before and during fires | 2018–2025 |
| MODIS MOD13A3.061 | Vegetation health (EVI index) to measure live fuel dryness | 2018–2025 |
| SRTM & ESA WorldCover | Physical elevation and land cover types via virtual raster mapping | Static |
| Open-Meteo Forecast API | Live weather for real-time inference | Real-time |

## 🧠 How It Works

We built a few different AI models to see which one is smartest at finding fires. Here's the step-by-step process:

**1. Learning from the Past** — We feed the models over 9.4 million rows of historical data. They learn that dry fuel (EVI), high elevation gradients, and extreme fire weather (VPD × Wind) mean a higher chance of fire.

**2. Smart Feature Engineering** — We don't just hand the model raw numbers. We compute specific, physics-informed features:

- **Temporal** — Cyclical month encoding (sin/cos) so the model understands seasonality.
- **Weather** — Temperature, humidity, wind speed, and VPD (Vapor Pressure Deficit).
- **Topography & Fuel** — SRTM Elevation (meters) and MODIS EVI (real satellite greenery values).
- **Land Cover** — ESA WorldCover one-hot encoded categorical classes.
- **Interactions** — VPD × Wind (combined fire spread potential index).

**3. Testing the Models (Temporal Holdout)** — To ensure our models aren't just "memorizing" the map, we use a strict **Temporal Split**. We train the models on years 2018–2024 and test them entirely on unseen future data from 2025. This proves the system actually understands fire physics, not just geography.

**4. The Best Models Win** — We trained five models. After rigorous out-of-time testing and SHAP diagnostic analysis, **XGBoost** emerged as our production champion, perfectly balancing spatial topography with dynamic weather triggers and catching 96% of true fire events.

| Model | Accuracy | Recall (Safety) | Temporal ROC-AUC |
|---|---|---|---|
| **XGBoost (Production Winner)** | 0.9014 | **0.9607** | **0.9611** |
| LightGBM | 0.9001 | 0.9274 | 0.9585 |
| Random Forest | 0.8908 | 0.8769 | 0.9584 |
| CatBoost | 0.8976 | 0.9296 | 0.9560 |
| Logistic Regression (Baseline) | 0.8258 | 0.8645 | 0.8968 |

## 💻 Interactive Dashboard & API

Reading numbers can be boring, so we built a unified visual ecosystem!

- **FastAPI Backend** (`src/api/`): A highly modularized Python backend handling live inference (`predictions`), Fire Risk Index mapping (`analytics`), active NASA FIRMS sync (`firms`), grounded LLM RAG processing (`chatbot`), and ML-validated activity approvals (`permits`).
- **Node.js API Gateway** (`wildfire-gateway-node/`): A secure gateway handling user authentication, request routing, and bridging the frontend to the internal ML services.
- **React Dashboard** (`frontend/`): A responsive, interactive SPA that plots real-time risk tiers, EVI canopy health, live NASA fire overlays, spread vectors, and provides a conversational RAG assistant to stakeholders.

## 💡 Key Design Decisions

**Date-Conditional Negative Sampling:** This is the secret to the model's accuracy. Instead of pulling random "no-fire" coordinates from random days, negatives are drawn from confirmed non-fire cells on the *exact same dates* as fire events, matched to the same ERA5 weather grid. This forces the model to learn the actual *weather signal* of a fire day versus a non-fire day in the same season, rather than just learning "summer = fire."

**Strict Temporal Validation:** Trained strictly on ≤ 2024 data and tested on 2025 data. The fact that the model maintains a ~0.96 AUC on a future year confirms it generalizes across time and physics, completely preventing spatial data leakage.

**Virtual Raster Tiles (VRT):** To map local land cover without exhausting RAM, we stitched 17 fragmented ESA Landcover `.tif` tiles into a tiny, memory-efficient XML virtual map, allowing instantaneous geospatial sampling across 9.4 million rows.

**The "WHY" Engine:** Instead of handing first responders raw SHAP values, the pipeline uses a template engine to translate Shapley vectors into ranked, natural language sentences (e.g., *"Extreme atmospheric dryness (VPD 4.2 kPa) stressing vegetation"*).

## 📁 Project Structure

```
wildfire_v2/
├── frontend/                 # React web application (Risk Map, Chatbot, Analytics)
├── wildfire-gateway-node/    # Express.js API Gateway (Auth & Routing)
├── data/
│   ├── raw/                  # FIRMS CSVs, ERA5 NetCDF, SRTM, ESA VRT
│   └── processed/            # training_data.csv, training_features.csv
├── models/                   # Saved .pkl files + production_model.pkl
├── scripts/
│   ├── build_inference_cache.py  # Builds spatial parquet cache
│   └── train_models.py           # Temporal split, train, eval, save
├── src/
│   ├── api/                    # FastAPI Microservices
│   │   ├── routes/             # Modular API endpoints (predictions, analytics, chatbot, permits, firms)
│   │   ├── cache.py            # Region-aware, state-machine driven caching
│   │   └── app.py              # Lightweight FastAPI shell
│   ├── data/                   # NASA FIRMS sync, EVI lookup, ERA5
│   ├── features/               # Cyclical encoding, interactions
│   ├── explainability/         # The WHY Engine (SHAP -> Text)
│   ├── risk/                   # Unified Fire Risk Index (FRI) logic
│   └── inference/              # WildfireInferenceEngine + Open-Meteo ingest
```

## 🚀 Future Steps

We're always looking to make this better! In the future, we plan to:

- Calibrate the model's precision against massive real-world class imbalance (fires are rare compared to non-fires in a live grid).
- Integrate deeper fire perimeter simulations (like FARSITE) beyond our current indicative spread arrows.
- Expand beyond India to other fire-prone regions in South and Southeast Asia.

Inspired by the need to protect our environment and communities.