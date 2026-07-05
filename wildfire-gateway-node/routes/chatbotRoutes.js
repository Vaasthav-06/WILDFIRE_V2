import { Router } from 'express';
import { requireAuth } from '../middleware/authMiddleware.js';
import { chat } from '../controllers/chatbotController.js';

const router = Router();

router.post('/', requireAuth, chat);

export default router;
