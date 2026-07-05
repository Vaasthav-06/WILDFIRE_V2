import { checkPermitZone, applyForPermit } from '../services/mlService.js';
import { asyncHandler } from '../utils/asyncHandler.js';

export const checkZone = asyncHandler(async (req, res) => {
  const { lat, lon } = req.query;
  const latNum = Number(lat);
  const lonNum = Number(lon);

  if (Number.isNaN(latNum) || Number.isNaN(lonNum)) {
    return res.status(400).json({ error: 'lat and lon query params must be numeric' });
  }

  const data = await checkPermitZone(latNum, lonNum);
  res.json(data);
});

export const applyPermit = asyncHandler(async (req, res) => {
  const { applicantName, purpose, lat, lon, plannedDate } = req.body;

  if (!applicantName || !purpose || lat == null || lon == null) {
    return res.status(400).json({
      error: 'applicantName, purpose, lat, and lon are required',
    });
  }

  const validActivities = ["research", "tourism", "timber_harvest", "grazing", "extraction"];
  let activityType = "research";
  if (validActivities.includes(purpose.toLowerCase())) {
    activityType = purpose.toLowerCase();
  } else if (purpose.toLowerCase().includes("burn") || purpose.toLowerCase().includes("extract")) {
    activityType = "extraction";
  }

  const dateStr = plannedDate || new Date().toISOString().split('T')[0];

  const data = await applyForPermit({
    officer_name: applicantName,
    officer_id: req.user?.username || 'GUEST-001',
    zone_name: 'Custom Zone',
    latitude: Number(lat),
    longitude: Number(lon),
    activity_type: activityType,
    start_date: dateStr,
    end_date: dateStr,
    area_hectares: 10,
    has_fire_extinguisher: true,
    has_insurance: true,
    officer_trained: true
  });

  res.json(data);
});
