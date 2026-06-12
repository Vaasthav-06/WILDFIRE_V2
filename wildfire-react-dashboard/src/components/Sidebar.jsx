import { X, Thermometer, Droplets, Wind, Leaf } from 'lucide-react';

export default function Sidebar({ cell, onClose }) {
  if (!cell) return null;

  const { properties = {}, geometry } = cell;
  const { fire_prob, spread_bearing_deg, spread_intensity, temp_c, humidity_pct, wind_ms, vpd_kpa, kbdi, ndvi } = properties;
  
  // Use the exact clicked coordinates if provided, else fallback to geometry
  const lon = cell.clickLngLat ? cell.clickLngLat[0] : geometry?.coordinates?.[0];
  const lat = cell.clickLngLat ? cell.clickLngLat[1] : geometry?.coordinates?.[1];

  const getRiskDetails = (prob) => {
    if (prob >= 0.8) return { label: 'CRITICAL RISK', color: 'bg-red-500/20 text-red-400 border-red-500/50' };
    if (prob >= 0.5) return { label: 'HIGH RISK', color: 'bg-orange-500/20 text-orange-400 border-orange-500/50' };
    if (prob >= 0.3) return { label: 'MODERATE RISK', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50' };
    return { label: 'LOW RISK', color: 'bg-blue-500/20 text-blue-400 border-blue-500/50' };
  };

  const risk = getRiskDetails(fire_prob);

  return (
    <div className={`glass-panel absolute top-5 right-5 bottom-5 w-96 p-6 flex flex-col gap-6 transform transition-transform duration-300 z-50 overflow-y-auto`}>
      <button 
        onClick={onClose}
        className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-white/10"
      >
        <X size={20} />
      </button>

      <div>
        <h2 className="text-xl font-bold mb-2">Cell Details</h2>
        <div className="text-sm text-gray-400 font-mono mb-4">
          Lat: {lat != null ? Number(lat).toFixed(4) : 'N/A'}, Lon: {lon != null ? Number(lon).toFixed(4) : 'N/A'}
        </div>
        
        <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold border ${risk.color} mb-3 tracking-wider`}>
          {risk.label}
        </div>
        
        <div className="text-5xl font-black mb-1 text-white">
          {fire_prob != null ? (Number(fire_prob) * 100).toFixed(1) : '0.0'}<span className="text-2xl text-gray-400">%</span>
        </div>
        <div className="text-xs text-gray-400 uppercase tracking-widest">Ignition Probability</div>
      </div>

      <div className="w-full h-px bg-white/10 my-2"></div>

      <div>
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Driving Factors</h3>
        <div className="grid grid-cols-2 gap-3">
          
          <div className="bg-black/30 p-3 rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-gray-400 uppercase mb-1">
              <Thermometer size={14} /> Temp
            </div>
            <div className="text-lg font-semibold">{temp_c != null ? `${Number(temp_c).toFixed(1)}°C` : 'N/A'}</div>
          </div>
          
          <div className="bg-black/30 p-3 rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-gray-400 uppercase mb-1">
              <Droplets size={14} /> Humidity
            </div>
            <div className="text-lg font-semibold">{humidity_pct != null ? `${Number(humidity_pct).toFixed(0)}%` : 'N/A'}</div>
          </div>

          <div className="bg-black/30 p-3 rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-gray-400 uppercase mb-1">
              <Wind size={14} /> Wind
            </div>
            <div className="text-lg font-semibold">{wind_ms != null ? `${Number(wind_ms).toFixed(1)} m/s` : 'N/A'}</div>
          </div>

          <div className="bg-black/30 p-3 rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-gray-400 uppercase mb-1">
              <Thermometer size={14} className="opacity-0"/> VPD
            </div>
            <div className="text-lg font-semibold">{vpd_kpa != null ? `${Number(vpd_kpa).toFixed(2)} kPa` : 'N/A'}</div>
          </div>

          <div className="bg-black/30 p-3 rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-gray-400 uppercase mb-1">
              KBDI Drought
            </div>
            <div className="text-lg font-semibold">{kbdi != null ? Number(kbdi).toFixed(0) : 'N/A'}</div>
          </div>

          <div className="bg-black/30 p-3 rounded-xl border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-gray-400 uppercase mb-1">
              <Leaf size={14} /> NDVI
            </div>
            <div className="text-lg font-semibold">{ndvi != null ? Number(ndvi).toFixed(2) : 'N/A'}</div>
          </div>

        </div>
      </div>

      {spread_bearing_deg != null && spread_bearing_deg !== undefined && (
        <>
          <div className="w-full h-px bg-white/10 my-2"></div>
          <div>
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Spread Vector</h3>
            <div className="flex items-center gap-4 bg-black/30 p-4 rounded-xl border border-white/5">
              <div 
                className="w-10 h-10 flex items-center justify-center bg-white/10 rounded-full border border-white/20"
                style={{ transform: `rotate(${spread_bearing_deg}deg)` }}
              >
                <div className="w-0.5 h-6 bg-white relative">
                  <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-white transform rotate-45"></div>
                </div>
              </div>
              <div>
                <div className="font-semibold">{Number(spread_bearing_deg).toFixed(0)}° Bearing</div>
                <div className="text-sm text-gray-400 capitalize">Intensity: <span className="text-white">{spread_intensity || 'unknown'}</span></div>
              </div>
            </div>
          </div>
        </>
      )}

    </div>
  );
}
