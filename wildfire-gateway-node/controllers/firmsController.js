import { fetchFirmsAlerts, triggerFirmsSync, getRegion } from '../services/mlService.js';

export async function getFirmsAlerts(req, res, next) {
  try {
    const { regionKey, hours } = req.query;
    let bounds = null;
    
    if (regionKey) {
      const region = getRegion(regionKey);
      bounds = {
        lat_min: region.latMin,
        lat_max: region.latMax,
        lon_min: region.lonMin,
        lon_max: region.lonMax,
      };
    }
    
    const data = await fetchFirmsAlerts(bounds, hours || 24);
    res.json(data);
  } catch (error) {
    next(error);
  }
}

export async function triggerSync(req, res, next) {
  try {
    const data = await triggerFirmsSync();
    res.json(data);
  } catch (error) {
    next(error);
  }
}
