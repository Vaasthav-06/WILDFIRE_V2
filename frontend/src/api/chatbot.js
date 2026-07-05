import client from './client';

export async function sendMessage(message, history, lat = null, lon = null) {
  const res = await client.post('/api/chatbot', { message, history, lat, lon });
  return res.data;
}
