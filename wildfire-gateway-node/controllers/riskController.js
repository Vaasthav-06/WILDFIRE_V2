import PredictionCache from '../models/PredictionCache.js';
import {
  fetchPrediction24h,
  fetchPrediction7day,
  fetchEviLayer,
  fetchCellPrediction,
  getRegion,
  listRegions,
} from '../services/mlService.js';
import { asyncHandler } from '../utils/asyncHandler.js';

const CACHE_TTL_MINUTES = Number(process.env.CACHE_TTL_MINUTES || 60);

function isFresh(computedAt) {
  const ageMinutes = (Date.now() - new Date(computedAt).getTime()) / 60000;
  return ageMinutes < CACHE_TTL_MINUTES;
}

async function getOrFetch(cacheKey, fetchFn) {
  const cached = await PredictionCache.findOne({ regionKey: cacheKey });
  if (cached && isFresh(cached.computedAt)) {
    return { data: cached.geojsonPayload, cache: 'hit' };
  }

  const fresh = await fetchFn();

  await PredictionCache.findOneAndUpdate(
    { regionKey: cacheKey },
    { regionKey: cacheKey, geojsonPayload: fresh, computedAt: new Date() },
    { upsert: true }
  );

  return { data: fresh, cache: 'miss' };
}

export const getRegions = asyncHandler(async (req, res) => {
  res.json({ regions: listRegions() });
});

export const get24hRisk = asyncHandler(async (req, res) => {
  const regionKey = (req.query.region || 'kmtr').toLowerCase();
  const area = getRegion(regionKey);
  const { data, cache } = await getOrFetch(`${regionKey}_24h`, () => fetchPrediction24h(area));
  res.json({ ...data, _cache: cache, _region: regionKey });
});

export const get7dayRisk = asyncHandler(async (req, res) => {
  const regionKey = (req.query.region || 'kmtr').toLowerCase();
  const area = getRegion(regionKey);
  const { data, cache } = await getOrFetch(`${regionKey}_7day`, () => fetchPrediction7day(area));
  res.json({ ...data, _cache: cache, _region: regionKey });
});

export const getEviLayer = asyncHandler(async (req, res) => {
  const regionKey = (req.query.region || 'kmtr').toLowerCase();
  // Bypass MongoDB caching entirely because the GeoJSON payload exceeds the 16MB document limit
  const data = await fetchEviLayer();
  res.json({ ...data, _cache: 'bypassed', _region: regionKey });
});

export const getCellDetail = asyncHandler(async (req, res) => {
  const { lat, lon } = req.params;
  const forecastDays = Number(req.query.forecast_days || 1);

  const latNum = Number(lat);
  const lonNum = Number(lon);

  if (Number.isNaN(latNum) || Number.isNaN(lonNum)) {
    return res.status(400).json({ error: 'lat and lon must be numeric' });
  }

  const data = await fetchCellPrediction(latNum, lonNum, forecastDays);
  res.json(data);
});