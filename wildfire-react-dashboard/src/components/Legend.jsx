export default function Legend() {
  return (
    <div className="glass-panel absolute bottom-5 left-5 p-4 flex flex-col gap-3 z-40 w-64">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-1">Ignition Probability</h3>
      
      <div className="flex items-center gap-3 text-sm">
        <div className="w-4 h-4 rounded shadow-sm bg-red-500/80 border border-red-500"></div>
        <span>Critical (&gt; 80%)</span>
      </div>
      
      <div className="flex items-center gap-3 text-sm">
        <div className="w-4 h-4 rounded shadow-sm bg-orange-500/80 border border-orange-500"></div>
        <span>High (50% - 80%)</span>
      </div>

      <div className="flex items-center gap-3 text-sm">
        <div className="w-4 h-4 rounded shadow-sm bg-yellow-500/60 border border-yellow-500"></div>
        <span>Moderate (40% - 50%)</span>
      </div>

      <div className="flex items-center gap-3 text-sm">
        <div className="w-4 h-4 rounded shadow-sm bg-blue-500/10 border border-blue-500/50"></div>
        <span>Low (&lt; 40%)</span>
      </div>
      
      <div className="w-full h-px bg-white/10 my-1"></div>
      
      <div className="flex items-center gap-3 text-sm mt-1">
        <div className="w-4 flex justify-center text-white">↑</div>
        <span>Spread Direction</span>
      </div>
    </div>
  );
}
