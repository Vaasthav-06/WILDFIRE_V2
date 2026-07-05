import client from './client';

export async function checkZone(lat, lon) {
  const res = await client.get('/api/permits/check-zone', { params: { lat, lon } });
  return res.data;
}

export async function applyPermit(payload) {
  const res = await client.post('/api/permits/apply', payload);
  return res.data;
}
