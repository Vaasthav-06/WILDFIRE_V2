import { createContext, useContext, useState } from 'react';

/**
 * LocationContext — shared selected-cell state across Map, Chatbot, and Permits.
 *
 * Usage:
 *   const { selectedCell, setSelectedCell, clearSelectedCell } = useLocation();
 *   selectedCell = { lat: 8.6, lon: 77.4 } | null
 */
const LocationContext = createContext(null);

export function LocationProvider({ children }) {
  const [selectedCell, setSelectedCell] = useState(null); // { lat, lon }

  function clearSelectedCell() {
    setSelectedCell(null);
  }

  return (
    <LocationContext.Provider value={{ selectedCell, setSelectedCell, clearSelectedCell }}>
      {children}
    </LocationContext.Provider>
  );
}

export function useLocation() {
  const ctx = useContext(LocationContext);
  if (!ctx) throw new Error('useLocation must be used within a LocationProvider');
  return ctx;
}
