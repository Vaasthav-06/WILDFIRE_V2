import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_GATEWAY_URL,
});

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('wildfire_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('wildfire_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default client;