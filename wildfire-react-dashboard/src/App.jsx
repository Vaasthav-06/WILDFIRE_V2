import { useState } from 'react';
import MapView from './components/MapView';
import Sidebar from './components/Sidebar';
import Legend from './components/Legend';

function App() {
  const [selectedCell, setSelectedCell] = useState(null);

  return (
    <div className="relative w-screen h-screen bg-background overflow-hidden">
      {/* Map Layer */}
      <div className="absolute inset-0">
        <MapView onSelectCell={setSelectedCell} />
      </div>

      {/* Header Overlay */}
      <header className="glass-panel absolute top-5 left-5 p-5 w-96 z-40">
        <h1 className="text-2xl font-extrabold mb-2 text-gradient">AI Wildfire Prediction</h1>
        <p className="text-sm text-gray-400 leading-relaxed">
          Daily risk forecasting across India based on ERA5 climate data, NDVI, and an ensemble machine learning model.
        </p>
      </header>

      {/* Legend Overlay */}
      <Legend />

      {/* Sidebar Overlay */}
      <Sidebar 
        cell={selectedCell} 
        onClose={() => setSelectedCell(null)} 
      />
    </div>
  );
}

export default App;
