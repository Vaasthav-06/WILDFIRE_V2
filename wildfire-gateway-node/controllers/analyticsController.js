import PredictionCache from '../models/PredictionCache.js';
import {
  fetchFeatureImportance,
  fetchValidationHistory,
  fetchEviTrend,
  fetchFireHistoryStats,
} from '../services/mlService.js';
import { asyncHandler } from '../utils/asyncHandler.js';

// Analytics data changes rarely — cache much longer than risk layers
const ANALYTICS_CACHE_TTL_MINUTES = Number(process.env.ANALYTICS_CACHE_TTL_MINUTES || 720); // 12h

function isFresh(computedAt) {
  const ageMinutes = (Date.now() - new Date(computedAt).getTime()) / 60000;
  return ageMinutes < ANALYTICS_CACHE_TTL_MINUTES;
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

export const getFeatureImportance = asyncHandler(async (req, res) => {
  const { data, cache } = await getOrFetch('feature_importance', fetchFeatureImportance);
  res.json({ ...data, _cache: cache });
});

export const getValidationHistory = asyncHandler(async (req, res) => {
  const { data, cache } = await getOrFetch('validation_history', fetchValidationHistory);
  res.json({ ...data, _cache: cache });
});

export const getEviTrend = asyncHandler(async (req, res) => {
  const { data, cache } = await getOrFetch('evi_trend', fetchEviTrend);
  res.json({ ...data, _cache: cache });
});

export const getFireHistoryStats = asyncHandler(async (req, res) => {
  const { data, cache } = await getOrFetch('fire_history_stats', fetchFireHistoryStats);
  res.json({ ...data, _cache: cache });
});