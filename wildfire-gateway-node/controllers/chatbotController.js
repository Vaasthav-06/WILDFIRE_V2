import { fetchCellPrediction } from '../services/mlService.js';
import { asyncHandler } from '../utils/asyncHandler.js';
import axios from 'axios';

/**
 * Build a rich fire_context string from ML prediction data.
 * This is injected into the Gemini prompt so it answers with real model data.
 */
function buildFireContext(lat, lon, prediction) {
  const prob = prediction.fire_prob ?? prediction.fire_prob;
  const probPct = prob != null ? `${(prob * 100).toFixed(1)}%` : 'N/A';
  const tier = prediction.risk_tier ?? 'Unknown';
  const w = prediction.weather ?? {};

  const lines = [
    `Location: ${lat}°N, ${lon}°E`,
    `Fire probability: ${probPct}`,
    `Risk tier: ${tier}`,
    `Safe to enter: ${prediction.safe_to_enter ?? (prob < 0.50 ? 'Yes' : 'No')}`,
  ];

  if (w.temp_c != null)      lines.push(`Temperature: ${w.temp_c.toFixed(1)}°C`);
  if (w.humidity_pct != null) lines.push(`Humidity: ${w.humidity_pct.toFixed(0)}%`);
  if (w.wind_ms != null)      lines.push(`Wind speed: ${w.wind_ms.toFixed(1)} m/s`);
  if (w.vpd_kpa != null)      lines.push(`VPD: ${w.vpd_kpa.toFixed(2)} kPa`);
  if (w.elevation_m != null)  lines.push(`Elevation: ${w.elevation_m.toFixed(0)} m`);

  if (prediction.confidence != null)
    lines.push(`Model confidence: ${(prediction.confidence * 100).toFixed(0)}%`);

  if (Array.isArray(prediction.top_reasons) && prediction.top_reasons.length > 0)
    lines.push(`Top risk reasons: ${prediction.top_reasons.join('; ')}`);

  if (Array.isArray(prediction.mitigating_factors) && prediction.mitigating_factors.length > 0)
    lines.push(`Mitigating factors: ${prediction.mitigating_factors.join('; ')}`);

  if (prediction.spread_direction)
    lines.push(`Fire spread direction: ${prediction.spread_direction}`);

  lines.push(`Forecast date: ${prediction.forecast_date ?? 'today'}`);

  return lines.join('\n');
}

export const chat = asyncHandler(async (req, res) => {
  const { message, history, lat, lon } = req.body;

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'message (string) is required' });
  }

  let fire_context = '';

  // ── Step 1: If coordinates are available, query the ML prediction engine ──
  if (lat != null && lon != null && !isNaN(Number(lat)) && !isNaN(Number(lon))) {
    try {
      const prediction = await fetchCellPrediction(Number(lat), Number(lon), 1);
      fire_context = buildFireContext(Number(lat), Number(lon), prediction);
    } catch (err) {
      // Non-fatal: fall back to no context; Gemini will still respond generically
      console.warn(`[chatbot] Could not fetch ML prediction for (${lat}, ${lon}): ${err.message}`);
      fire_context = `Location: ${lat}°N, ${lon}°E (prediction data unavailable)`;
    }
  }

  // ── Step 2: Forward to Python /api/chatbot with real context injected ──
  const mlUrl = process.env.ML_SERVICE_URL;
  const response = await axios.post(
    `${mlUrl}/api/chatbot`,
    { message, fire_context },
    { timeout: Number(process.env.ML_SERVICE_TIMEOUT_MS || 90000) },
  );

  res.json(response.data);
});
