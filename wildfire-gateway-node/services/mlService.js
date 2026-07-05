import axios from 'axios';

const mlClient = axios.create({
  baseURL: process.env.ML_SERVICE_URL,
  timeout: Number(process.env.ML_SERVICE_TIMEOUT_MS || 90000),
});

export async function fetchPrediction24h(area) {
  const isFullMap = area.label === 'All India';
  const requestBody = {
    area: isFullMap ? null : {
      lat_min: area.latMin,
      lat_max: area.latMax,
      lon_min: area.lonMin,
      lon_max: area.lonMax,
      resolution: area.resolution,
    },
    include_spread: true,
    include_explanation: true,
    min_prob_geojson: 0.15,
  };
  const response = await mlClient.post('/predict/24h', requestBody);
  return response.data;
}

export async function fetchPrediction7day(area) {
  const isFullMap = area.label === 'All India';
  const requestBody = {
    area: isFullMap ? null : {
      lat_min: area.latMin,
      lat_max: area.latMax,
      lon_min: area.lonMin,
      lon_max: area.lonMax,
      resolution: area.resolution,
    },
    include_spread: true,
    include_explanation: false,
    min_prob_geojson: 0.15,
  };
  const response = await mlClient.post('/predict/7day', requestBody);
  return response.data;
}

export async function fetchEviLayer() {
  const response = await mlClient.get('/map/evi');
  return response.data;
}

export async function fetchCellPrediction(lat, lon, forecastDays = 1) {
  const response = await mlClient.get(`/predict/cell/${lat}/${lon}`, {
    params: { forecast_days: forecastDays },
  });
  return response.data;
}

export async function fetchFeatureImportance() {
  const response = await mlClient.get('/analytics/feature-importance');
  return response.data;
}

export async function fetchValidationHistory() {
  const response = await mlClient.get('/analytics/validation-history');
  return response.data;
}

export async function fetchEviTrend() {
  const response = await mlClient.get('/analytics/evi-trend');
  return response.data;
}

export async function fetchFireHistoryStats() {
  const response = await mlClient.get('/analytics/fire-history-stats');
  return response.data;
}

const REGIONS = {
  kmtr: {
    label: 'KMTR Reserve',
    latMin: 8.35, latMax: 8.85, lonMin: 77.15, lonMax: 77.60,
    resolution: 0.05,
    center: [8.6, 77.4],
    zoom: 10,
  },
  /*
  tamil_nadu: {
    label: 'Tamil Nadu',
    latMin: 8.0, latMax: 13.5, lonMin: 76.2, lonMax: 80.3,
    resolution: 0.1,
    center: [10.5, 78.2],
    zoom: 7,
  },
  south_india: {
    label: 'South India',
    latMin: 8.0, latMax: 18.5, lonMin: 74.0, lonMax: 84.5,
    resolution: 0.15,
    center: [13.0, 78.5],
    zoom: 6,
  },
  india: {
    label: 'All India',
    latMin: 8.0, latMax: 37.6, lonMin: 68.1, lonMax: 97.4,
    resolution: 0.2,
    center: [22.5, 82.0],
    zoom: 5,
  },
  */
};

export function getRegion(regionKey) {
  const region = REGIONS[regionKey];
  if (!region) {
    throw Object.assign(new Error(`Unknown region: ${regionKey}`), { statusCode: 400 });
  }
  return region;
}

export function listRegions() {
  return Object.entries(REGIONS).map(([key, r]) => ({
    key, label: r.label, center: r.center, zoom: r.zoom,
  }));
}

export async function checkPermitZone(lat, lon) {
  const response = await mlClient.get('/api/permits/check-zone', {
    params: { lat, lon },
  });
  return response.data;
}

export async function applyForPermit(payload) {
  const response = await mlClient.post('/api/permits/apply', payload);
  return response.data;
}

export async function sendChatbotMessage(message, fire_context = "") {
  const response = await mlClient.post('/api/chatbot', { message, fire_context });
  return response.data;
}

export async function fetchFirmsAlerts(bounds, hours = 24) {
  const params = { hours };
  if (bounds) {
    params.lat_min = bounds.lat_min;
    params.lat_max = bounds.lat_max;
    params.lon_min = bounds.lon_min;
    params.lon_max = bounds.lon_max;
  }
  const response = await mlClient.get('/api/firms/alerts', { params });
  return response.data;
}

export async function triggerFirmsSync() {
  const response = await mlClient.post('/api/firms/sync');
  return response.data;
}
