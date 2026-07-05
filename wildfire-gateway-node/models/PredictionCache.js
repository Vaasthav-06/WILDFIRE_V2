import mongoose from 'mongoose';

const predictionCacheSchema = new mongoose.Schema({
  regionKey: { type: String, required: true, unique: true }, // e.g. "kmtr_24h"
  geojsonPayload: { type: mongoose.Schema.Types.Mixed, required: true },
  computedAt: { type: Date, required: true, default: Date.now },
});

export default mongoose.model('PredictionCache', predictionCacheSchema);