# 🔥 WILDFIRE_REBORN

A machine learning pipeline that predicts where wildfires are likely to happen across India — before they do. By learning from 8 years of satellite and climate data, it gives communities and responders an early warning on fire risk.

**Spatial CV AUC: 0.97 | Temporal AUC (2023→2024): 0.97**

---

## 🎯 What Does This Project Do?

We want to help people stay prepared. Given today's weather conditions, the platform:

- Predicts next-day wildfire probability at **0.1° grid resolution** across India
- Outputs a **risk score (0–1)** per grid cell, a **risk tier** (Low / Moderate / High / Extreme), and a **fire spread direction vector** for high-risk zones
- Plots everything on an **interactive React dashboard** so the danger zones are easy to understand at a glance

---

## 📊 Where Do We Get Our Data?

To train our models, we used information from trusted scientific sources:

| Source | What it tells us | Coverage |
|---|---|---|
| NASA FIRMS (VIIRS S-NPP C2) | Exactly when and where past wildfires happened | 2018–2025 |
| ERA5 (ECMWF via CDS) | Temperature, humidity, wind, VPD, precipitation before and during fires | 2018–2025 |
| MODIS MOD13A3.061 (NASA AppEEARS) | Vegetation health (EVI index at 1km monthly) | 2018–2025 |
| Open-Meteo Forecast API | Live weather for real-time inference | Real-time |

---

## 🧠 How It Works

We built a few different AI models to see which one is smartest at finding fires. Here's the step-by-step process:

**1. Learning from the Past** — We feed the models lots of historical data. They learn that hot, dry, and windy days usually mean a higher chance of fire.

**2. Smart Feature Engineering** — We don't just hand the model raw weather numbers. We compute 15 carefully chosen features:

- **Ecoregion** — 5 binary zone flags (tropical moist, tropical dry, semi-arid, montane, subtropical)
- **Temporal** — cyclical month encoding (sin/cos) so the model understands seasonality
- **Weather** — temperature, humidity, wind speed, VPD (Vapor Pressure Deficit)
- **Drought** — KBDI (Keetch-Byram Drought Index), computed per cell from ERA5
- **Vegetation** — MODIS EVI (real satellite values, not proxies)
- **Interactions** — VPD × wind, temperature × normalised KBDI

**3. Testing the Models** — We evaluate using **5-fold Spatial Block CV** (4° × 4° geographic blocks with a 0.5° buffer to prevent spatial leakage) and a strict **temporal holdout** (train ≤ 2023, test ≥ 2024). This makes sure our models are genuinely good at predicting fires everywhere and across time — not just memorising the training data.

**4. The Best Models Win** — We trained five models and combined the top three into a soft-voting ensemble:

| Model | Spatial CV AUC | Temporal AUC | Recall |
|---|---|---|---|
| Logistic Regression (baseline) | 0.9621 ± 0.011 | 0.9612 | 0.932 |
| Random Forest | 0.9700 ± 0.010 | 0.9729 | 0.952 |
| XGBoost | 0.9714 ± 0.009 | 0.9746 | 0.952 |
| CatBoost | 0.9711 ± 0.009 | 0.9743 | 0.950 |
| LightGBM | 0.9713 ± 0.009 | — | — |
| **Ensemble (top 3)** | — | **~0.975** | — |


---

## 🖥️ Interactive Dashboard

Reading numbers can be boring, so we built a visual dashboard!

- `predict_tomorrow.py` calls the **Open-Meteo live weather API** and generates a `predictions.geojson` file
- `run_prediction.py` runs the same pipeline offline with synthetic weather for testing
- The **React Dashboard**  reads the GeoJSON and plots danger zones on an interactive map

---

## 📁 Project Structure

```
wildfire_v2/
├── data/
│   ├── raw/               # FIRMS CSVs, ERA5 NetCDF, MODIS EVI NetCDF
│   └── processed/         # training_data.csv, training_features.csv
├── models/                # Saved .pkl files + production_model.pkl
├── results/
│   ├── figures/           # ROC, PR, confusion matrices, SHAP, reliability
│   └── metrics/           # cv_results.json, classification_report.csv
├── scripts/
│   ├── merge_firms.py          # Merge 2018-2023 + 2024-2026 FIRMS CSVs
│   ├── download_era5.py        # One-time ERA5 download via CDS API
│   ├── build_training_data.py  # FIRMS + ERA5 + negative sampling → training_data.csv
│   ├── build_features.py       # Feature engineering → training_features.csv
│   └── train_models.py         # Train, evaluate, save all models
├── src/
│   ├── config.py               # All paths, feature list, constants
│   ├── data/
│   │   ├── era5.py             # ERA5 download + local lookup
│   │   ├── evi.py              # MODIS EVI lookup (cftime-aware)
│   │   ├── nasa_firms.py       # FIRMS ingestion + landmass filter
│   │   └── negative_sampling.py # Date-conditional true-absence sampling
│   ├── features/
│   │   └── engineering.py      # KBDI, ecoregion, interactions, full pipeline
│   ├── models/
│   │   ├── cross_validation.py # SpatialBlockCV with Fold 4 diagnostics
│   │   ├── train.py            # Training loop, temporal validation, ensemble
│   │   └── evaluate.py         # ROC, PR, confusion matrices, SHAP, ablation
│   └── inference/
│       ├── engine.py           # WildfireInferenceEngine (predict_now / 7day)
│       └── forecast_ingest.py  # Open-Meteo live weather fetch
├── predict_tomorrow.py    # Run live 7-day prediction → GeoJSON
└── run_prediction.py      # Offline test with synthetic weather
```

---

## 💡 Key Design Decisions

**Negative sampling** uses date-conditional true-absence: negatives are drawn from confirmed non-fire cells on the *same dates* as fire events, matched to the same ERA5 weather grid. This forces the model to learn weather signal rather than just seasonal patterns — a genuinely hard problem.

**Spatial block CV** uses 4° × 4° geographic blocks with a 0.5° buffer zone to prevent spatial leakage between train and test. This is stricter than random CV and gives more honest AUC estimates.

**Temporal validation** trains strictly on ≤ 2023 data and tests on ≥ 2024 data. The fact that temporal AUC matches spatial CV AUC confirms the model generalises across time, not just space.

---

## 🚀 Future Steps

We're always looking to make this better! In the future, we plan to:

- Add forest cover data to improve predictions in densely vegetated regions
- Make the map grid even finer and faster to refresh
- Expand beyond India to other fire-prone regions in South and Southeast Asia

Inspired by the need to protect our environment and communities.

---

