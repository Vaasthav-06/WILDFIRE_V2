import express from 'express';
import { getFirmsAlerts, triggerSync } from '../controllers/firmsController.js';
import { requireAuth } from '../middleware/authMiddleware.js';

const router = express.Router();

router.get('/alerts', requireAuth, getFirmsAlerts);
router.post('/sync', requireAuth, triggerSync);

export default router;
