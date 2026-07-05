export function errorHandler(err, req, res, next) {
  console.error('[error]', err.message);

  const status = err.response?.status || err.statusCode || 500;
  const message =
    err.response?.data?.detail ||
    err.message ||
    'Internal server error';

  res.status(status).json({ error: message });
}