import { Router } from 'express';
import { requireAuth } from '../middleware/authMiddleware.js';
import { checkZone, applyPermit } from '../controllers/permitsController.js';

const router = Router();

router.get('/check-zone', requireAuth, checkZone);
router.post('/apply', requireAuth, applyPermit);

export default router;
