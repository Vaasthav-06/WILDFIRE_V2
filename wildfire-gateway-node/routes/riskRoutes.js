import { Router } from 'express';
import { requireAuth } from '../middleware/authMiddleware.js';
import { get24hRisk, get7dayRisk, getEviLayer, getCellDetail, getRegions } from '../controllers/riskController.js';

const router = Router();

router.get('/regions', requireAuth, getRegions);
router.get('/24h', requireAuth, get24hRisk);
router.get('/7day', requireAuth, get7dayRisk);
router.get('/evi', requireAuth, getEviLayer);
router.get('/cell/:lat/:lon', requireAuth, getCellDetail);

export default router;