import client from './client';

export async function fetchFeatureImportance() {
  const res = await client.get('/api/analytics/feature-importance');
  return res.data;
}

export async function fetchValidationHistory() {
  const res = await client.get('/api/analytics/validation-history');
  return res.data;
}

export async function fetchEviTrend() {
  const res = await client.get('/api/analytics/evi-trend');
  return res.data;
}

export async function fetchFireHistoryStats() {
  const res = await client.get('/api/analytics/fire-history-stats');
  return res.data;
}