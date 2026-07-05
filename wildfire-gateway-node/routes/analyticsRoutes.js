import { Router } from 'express';
import { requireAuth } from '../middleware/authMiddleware.js';
import {
  getFeatureImportance,
  getValidationHistory,
  getEviTrend,
  getFireHistoryStats,
} from '../controllers/analyticsController.js';

const router = Router();

router.get('/feature-importance', requireAuth, getFeatureImportance);
router.get('/validation-history', requireAuth, getValidationHistory);
router.get('/evi-trend', requireAuth, getEviTrend);
router.get('/fire-history-stats', requireAuth, getFireHistoryStats);

export default router;