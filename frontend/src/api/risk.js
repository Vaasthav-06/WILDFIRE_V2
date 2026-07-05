import client from './client';

export async function fetchRegions() {
  const res = await client.get('/api/risk/regions');
  return res.data.regions;
}

export async function fetchRisk24h(region = 'kmtr') {
  const response = await client.get('/api/risk/24h', { params: { region } });
  return response.data;
}

export async function fetchRisk7day(region = 'kmtr') {
  const response = await client.get('/api/risk/7day', { params: { region } });
  return response.data;
}

export async function fetchEvi(region = 'kmtr') {
  const response = await client.get('/api/risk/evi', { params: { region } });
  return response.data;
}

export async function fetchCellDetail(lat, lon, forecastDays = 1) {
  const response = await client.get(`/api/risk/cell/${lat}/${lon}`, {
    params: { forecast_days: forecastDays },
  });
  return response.data;
}

export async function fetchFirmsAlerts(region = 'kmtr', hours = 24) {
  const response = await client.get('/api/firms/alerts', { params: { regionKey: region, hours } });
  return response.data;
}