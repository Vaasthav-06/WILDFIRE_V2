import mongoose from 'mongoose';

export async function connectDB(mongoUri) {
  try {
    await mongoose.connect(mongoUri);
    console.log('[db] MongoDB connected');
  } catch (err) {
    console.error('[db] MongoDB connection failed:', err.message);
    process.exit(1);
  }
}