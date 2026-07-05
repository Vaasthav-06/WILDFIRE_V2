import 'dotenv/config';
import express from 'express';
import cors from 'cors';

import { connectDB } from './config/db.js';
import authRoutes from './routes/authRoutes.js';
import riskRoutes from './routes/riskRoutes.js';
import analyticsRoutes from './routes/analyticsRoutes.js';
import permitsRoutes from './routes/permitsRoutes.js';
import chatbotRoutes from './routes/chatbotRoutes.js';
import firmsRoutes from './routes/firmsRoutes.js';
import { errorHandler } from './middleware/errorHandler.js';

const app = express();

app.use(cors({ origin: process.env.CORS_ORIGIN, credentials: true }));
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use('/api/auth', authRoutes);
app.use('/api/risk', riskRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api/permits', permitsRoutes);
app.use('/api/chatbot', chatbotRoutes);
app.use('/api/firms', firmsRoutes);

app.use(errorHandler);

const PORT = process.env.PORT || 8080;

connectDB(process.env.MONGO_URI).then(() => {
  app.listen(PORT, () => {
    console.log(`[server] Gateway running on http://localhost:${PORT}`);
  });
});