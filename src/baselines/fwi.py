"""
src/baselines/fwi.py

Canadian Forest Fire Weather Index (FWI) System
Implements the full van Wagner (1987) equation cascade:
  FFMC -> DMC -> DC -> ISI -> BUI -> FWI

Reference: van Wagner, C.E. (1987). Development and structure of the Canadian
Forest Fire Weather Index System. Canadian Forestry Service Technical Report 35.
Ottawa, Canada.

CRITICAL DESIGN NOTE — Day-Over-Day State:
FWI codes are lagged moisture-content estimates that carry over from the previous
day. Unlike our other features (which are single-day snapshots), each code must
be updated sequentially from an initial startup value. This implementation:
  1. Groups the input DataFrame by (latitude, longitude) cell.
  2. Sorts each cell's rows chronologically.
  3. Applies the van Wagner equations as a recurrence relation.
  4. Uses standard startup values: FFMC0=85, DMC0=6, DC0=15.

Input DataFrame expected columns (matching training_features.csv + ERA5 schema):
  - acq_date     : date of observation
  - latitude     : grid cell latitude
  - longitude    : grid cell longitude
  - temp         : 2m air temperature (°C)
  - humidity     : 2m relative humidity (%)
  - wind         : 10m wind speed (km/h) — NOTE: ERA5 gives m/s; multiply by 3.6
  - precip       : 24h accumulated precipitation (mm)

Output columns added:
  - fwi_ffmc, fwi_dmc, fwi_dc, fwi_isi, fwi_bui, fwi

NOTE on wind units: van Wagner (1987) equations require wind speed in km/h.
ERA5 outputs 10m wind speed as u10, v10 components (m/s). The precip field
in training_features.csv is already in mm (ERA5 total_precip * 1000).
The wind field in training_features.csv is in m/s — convert internally.

NOTE on precipitation: training_features.csv does not include a `precip`
column (it was not used as a model feature). For FWI computation on the
training/test set, precip must be merged from the raw ERA5 data.
The `compute_fwi_for_dataframe()` helper accepts an optional `precip_col`
argument; if the column is absent, 0mm/day is assumed (conservative: FWI
will be slightly overestimated without rain).
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── van Wagner (1987) startup values ─────────────────────────────────────────
FFMC0 = 85.0    # Fine Fuel Moisture Code startup
DMC0  =  6.0    # Duff Moisture Code startup
DC0   = 15.0    # Drought Code startup


# ─────────────────────────────────────────────────────────────────────────────
# Core FFMC equations (van Wagner 1987, Eq. 1-12)
# ─────────────────────────────────────────────────────────────────────────────

def _ffmc(ffmc_prev: float, temp: float, rh: float, wind_kmh: float, rain: float) -> float:
    """
    Fine Fuel Moisture Code (FFMC).
    Represents moisture content of surface litter and fine fuels.
    Range: 0-101; higher = drier (more flammable).
    """
    # Convert FFMC to moisture content
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    # Rain effect
    if rain > 0.5:
        rf = rain - 0.5
        if mo <= 150.0:
            mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        else:
            mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)) + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf)
        mo = min(mo, 250.0)

    # Equilibrium moisture content (drying and wetting)
    ed = 0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    ew = 0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))

    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    elif mo < ew:
        kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
        kw = kl * 0.581 * np.exp(0.0365 * temp)
        m = ew - (ew - mo) * 10.0 ** (-kw)
    else:
        m = mo

    m = np.clip(m, 0, 250)
    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return float(np.clip(ffmc, 0, 101))


def _dmc(dmc_prev: float, temp: float, rh: float, rain: float, month: int) -> float:
    """
    Duff Moisture Code (DMC).
    Represents moisture of loosely compacted organic layers.
    Range: 0+ (no upper limit; higher = drier).
    """
    # Day-length factor for DMC (van Wagner 1987, Table 1) — India approximation
    # India spans ~8-37°N; using 15°N representative day-length correction
    el_lookup = {
        1:  6.5, 2:  7.5, 3:  9.0, 4: 12.8, 5: 13.9,  6: 13.9,
        7: 12.4, 8: 10.9, 9:  9.4, 10: 8.0, 11:  7.0, 12:  6.0,
    }
    el = el_lookup.get(month, 9.0)

    # Rain effect
    rw = rain - 1.5 if rain > 1.5 else 0.0
    if rain > 1.5:
        b = (
            100.0 / (0.5 + 0.3 * dmc_prev) if dmc_prev <= 33.0
            else 14.0 - 1.3 * np.log(dmc_prev) if dmc_prev <= 65.0
            else 6.2 * np.log(dmc_prev) - 17.2
        )
        wmr = 20.0 + 280.0 / np.exp(0.023 * dmc_prev)
        wmr = wmr + 1000 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - np.log(wmr - 20.0))
        dmc_prev = max(pr, 0.0)

    # Temperature too cold → no drying
    if temp < -1.1:
        return dmc_prev

    k = 1.894 * (temp + 1.1) * (100.0 - rh) * el * 1e-6
    return float(dmc_prev + 100.0 * k)


def _dc(dc_prev: float, temp: float, rain: float, month: int) -> float:
    """
    Drought Code (DC).
    Represents deep organic layer moisture, multi-week drying.
    Range: 0+ (no upper limit; higher = drier).
    """
    # Seasonal drying factor for DC (India ~15°N approximate)
    lf_lookup = {
        1: -1.6, 2: -1.6, 3:  0.9, 4:  3.8, 5:  5.8, 6:  6.4,
        7:  5.0, 8:  2.4, 9:  0.4, 10: -1.6, 11: -1.6, 12: -1.6,
    }
    lf = lf_lookup.get(month, 0.0)

    # Rain effect
    if rain > 2.8:
        rd = rain - 2.8
        qr = 800.0 * np.exp(-dc_prev / 400.0)
        qmr = qr + 3.937 * rd
        dc_prev = max(400.0 * np.log(800.0 / qmr), 0.0)

    if temp < -2.8:
        return dc_prev

    v = 0.36 * (temp + 2.8) + lf
    v = max(v, 0.0)
    return float(dc_prev + 0.5 * v)


def _isi(ffmc: float, wind_kmh: float) -> float:
    """
    Initial Spread Index (ISI).
    Combines FFMC (fuel dryness) and wind speed to estimate fire spread rate.
    Range: 0+.
    """
    fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)
    isi = 0.208 * np.exp(0.05039 * wind_kmh) * ff
    return float(max(isi, 0.0))


def _bui(dmc: float, dc: float) -> float:
    """
    Buildup Index (BUI).
    Combines DMC and DC to represent total fuel available for combustion.
    Range: 0+.
    """
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return float(max(bui, 0.0))


def _fwi(isi: float, bui: float) -> float:
    """
    Fire Weather Index (FWI).
    Final combined index. Range: 0+; roughly:
    <5 = Low, 5-17 = Moderate, 17-32 = High, 32-50 = Very High, >50 = Extreme.
    """
    if bui <= 80.0:
        bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
    else:
        bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui)))

    if bb <= 1.0:
        fwi_val = bb
    else:
        fwi_val = np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647)
    return float(max(fwi_val, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_fwi_for_cell(
    dates: np.ndarray,
    temps: np.ndarray,
    rhs: np.ndarray,
    winds_ms: np.ndarray,
    precips_mm: np.ndarray,
    ffmc_init: float = FFMC0,
    dmc_init: float  = DMC0,
    dc_init: float   = DC0,
) -> pd.DataFrame:
    """
    Compute FWI codes for a single grid cell's time series.

    Parameters
    ----------
    dates       : array-like of pd.Timestamp or np.datetime64
    temps       : 2m temperature in °C
    rhs         : relative humidity in %
    winds_ms    : wind speed in m/s (converted internally to km/h)
    precips_mm  : 24h precipitation in mm
    ffmc_init   : FFMC startup value (default: van Wagner 1987)
    dmc_init    : DMC startup value
    dc_init     : DC startup value

    Returns
    -------
    DataFrame with columns: [date, ffmc, dmc, dc, isi, bui, fwi]
    """
    n = len(dates)
    months = pd.to_datetime(dates).month.values if not isinstance(dates[0], int) else dates

    ffmc_arr = np.zeros(n)
    dmc_arr  = np.zeros(n)
    dc_arr   = np.zeros(n)
    isi_arr  = np.zeros(n)
    bui_arr  = np.zeros(n)
    fwi_arr  = np.zeros(n)

    ffmc_cur = ffmc_init
    dmc_cur  = dmc_init
    dc_cur   = dc_init

    for i in range(n):
        t    = float(temps[i])
        rh   = float(np.clip(rhs[i], 0.0, 100.0))
        w_ms = float(winds_ms[i])
        rain = float(max(precips_mm[i], 0.0))
        mo   = int(months[i])
        w_kmh = w_ms * 3.6   # m/s → km/h

        ffmc_cur = _ffmc(ffmc_cur, t, rh, w_kmh, rain)
        dmc_cur  = _dmc(dmc_cur,  t, rh, rain, mo)
        dc_cur   = _dc(dc_cur,    t, rain, mo)
        isi_cur  = _isi(ffmc_cur, w_kmh)
        bui_cur  = _bui(dmc_cur, dc_cur)
        fwi_cur  = _fwi(isi_cur, bui_cur)

        ffmc_arr[i] = ffmc_cur
        dmc_arr[i]  = dmc_cur
        dc_arr[i]   = dc_cur
        isi_arr[i]  = isi_cur
        bui_arr[i]  = bui_cur
        fwi_arr[i]  = fwi_cur

    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "fwi_ffmc": ffmc_arr,
        "fwi_dmc":  dmc_arr,
        "fwi_dc":   dc_arr,
        "fwi_isi":  isi_arr,
        "fwi_bui":  bui_arr,
        "fwi":      fwi_arr,
    })


def compute_fwi_for_dataframe(
    df: pd.DataFrame,
    lat_col:    str = "latitude",
    lon_col:    str = "longitude",
    date_col:   str = "acq_date",
    temp_col:   str = "temp",
    rh_col:     str = "humidity",
    wind_col:   str = "wind",
    precip_col: Optional[str] = "precip",
    ffmc_init: float = FFMC0,
    dmc_init:  float = DMC0,
    dc_init:   float = DC0,
) -> pd.DataFrame:
    """
    Compute FWI codes for an entire DataFrame with multiple grid cells.

    The function groups by (lat, lon), sorts chronologically within each cell,
    applies the van Wagner recurrence, then reassembles.

    If `precip_col` is None or not present in df, 0mm/day is assumed —
    FWI will be slightly overestimated on rainy days but remains directionally
    correct as a ranking metric.

    Returns the input df with six new columns:
      fwi_ffmc, fwi_dmc, fwi_dc, fwi_isi, fwi_bui, fwi
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Handle missing precip
    has_precip = (precip_col is not None) and (precip_col in df.columns)
    if not has_precip:
        df["_precip_tmp"] = 0.0
        precip_col = "_precip_tmp"

    new_cols = ["fwi_ffmc", "fwi_dmc", "fwi_dc", "fwi_isi", "fwi_bui", "fwi"]
    for c in new_cols:
        df[c] = np.nan

    grouped = df.groupby([lat_col, lon_col], sort=False)
    n_cells = len(grouped)
    done = 0

    for (lat, lon), grp in grouped:
        grp_sorted = grp.sort_values(date_col)
        idx = grp_sorted.index

        cell_fwi = compute_fwi_for_cell(
            dates      = grp_sorted[date_col].values,
            temps      = grp_sorted[temp_col].values,
            rhs        = grp_sorted[rh_col].values,
            winds_ms   = grp_sorted[wind_col].values,
            precips_mm = grp_sorted[precip_col].values,
            ffmc_init  = ffmc_init,
            dmc_init   = dmc_init,
            dc_init    = dc_init,
        )

        for c in new_cols:
            df.loc[idx, c] = cell_fwi[c].values

        done += 1
        if done % 1000 == 0:
            print(f"    FWI: processed {done}/{n_cells} cells ...", end="\r", flush=True)

    if "_precip_tmp" in df.columns:
        df.drop(columns=["_precip_tmp"], inplace=True)

    return df
