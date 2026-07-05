import { useEffect, useState, useCallback } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import { fetchRisk24h, fetchRisk7day, fetchEvi, fetchFirmsAlerts } from '../api/risk';
import RiskLegend from './RiskLegend';
import LayerControl from './LayerControl';
import RegionSwitcher from './RegionSwitcher';
import DaySlider from './DaySlider';
import SpreadArrows from './SpreadArrows';
import CellDetailPanel from '../pages/CellDetailPage';
import { useLocation } from '../context/LocationContext';

const REGION_VIEWS = {
  kmtr: { center: [8.6, 77.4], zoom: 10 },
  tamil_nadu: { center: [10.5, 78.2], zoom: 7 },
  south_india: { center: [13.0, 78.5], zoom: 6 },
  india: { center: [22.5, 82.0], zoom: 5 },
};

function riskColor(prob) {
  if (prob >= 0.65) return '#7a0d0d';
  if (prob >= 0.45) return '#d9622b';
  if (prob >= 0.25) return '#e0b03a';
  return '#4a7c59';
}

function riskStyle(feature) {
  const prob = feature.properties?.fire_prob ?? 0;
  return { fillColor: riskColor(prob), fillOpacity: 0.55, color: riskColor(prob), weight: 1 };
}

function eviColor(evi) {
  if (evi > 0.60) return '#205c2a'; // Dark Green
  if (evi > 0.45) return '#4a7c59'; // Green
  if (evi > 0.30) return '#a3c9a8'; // Light Green
  if (evi > 0.15) return '#e0b03a'; // Yellow
  return '#8b5a2b'; // Brown
}

function eviStyle(feature) {
  const evi = feature.properties?.evi ?? 0;
  return { fillColor: eviColor(evi), fillOpacity: 0.35, color: '#1a2420', weight: 0.8, opacity: 0.8 };
}

export default function RiskMap() {
  const [activeLayer, setActiveLayer] = useState('24h');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [activeRegion, setActiveRegion] = useState('kmtr');
  const [day, setDay] = useState(1);
  
  const [layer24h, setLayer24h] = useState(null);
  const [layer7day, setLayer7day] = useState(null);
  const [layerEvi, setLayerEvi] = useState(null);
  const [layerFirms, setLayerFirms] = useState(null);
  const [showFirms, setShowFirms] = useState(true);

  function handleRegionChange(newRegion) {
    setActiveRegion(newRegion);
    setLayer24h(null);
    setLayer7day(null);
    setLayerEvi(null);
    setLayerFirms(null);
    setDay(1);
  }

  // Use global LocationContext so Chatbot + Permits share the selected cell
  const { selectedCell, setSelectedCell } = useLocation();
  const [localPanelCell, setLocalPanelCell] = useState(null); // drives CellDetailPanel independently

  useEffect(() => {
    let canceled = false;
    async function loadLayer(layerId) {
      setLoading(true);
      setError(null);
      try {
        if (layerId === '24h' && !layer24h) setLayer24h(await fetchRisk24h(activeRegion));
        else if (layerId === '7day' && !layer7day) setLayer7day(await fetchRisk7day(activeRegion));
        else if (layerId === 'evi' && !layerEvi) setLayerEvi(await fetchEvi(activeRegion));

        if (showFirms && !layerFirms) {
          setLayerFirms(await fetchFirmsAlerts(activeRegion));
        }
      } catch (err) {
        if (!canceled) setError('Could not load layer. Is the backend running?');
      } finally {
        if (!canceled) setLoading(false);
      }
    }
    loadLayer(activeLayer);
    return () => { canceled = true; };
  }, [activeLayer, activeRegion, showFirms]);

  function onEachRiskFeature(feature, layer) {
    const p = feature.properties || {};
    const pct = p.fire_prob != null ? `${Math.round(p.fire_prob * 100)}%` : 'N/A';
    const fri = p.fri != null ? p.fri : 'N/A';
    layer.bindPopup(
      `<strong>Risk: ${pct}</strong><br/>FRI: ${fri} / 100<br/>Tier: ${p.risk_tier ?? '—'}<br/>` +
      `Click marker again or use "Inspect" below for full detail.`
    );
    layer.on('click', () => {
      if (p.latitude != null && p.longitude != null) {
        const cell = { lat: p.latitude, lon: p.longitude };
        setSelectedCell(cell);        // global — shared with Chatbot & Permits
        setLocalPanelCell(cell);      // local — opens the detail panel
      }
    });
  }

  function onEachEviFeature(feature, layer) {
    const p = feature.properties || {};
    const eviVal = p.evi != null ? p.evi.toFixed(4) : 'N/A';
    layer.bindPopup(
      `<strong>EVI: ${eviVal}</strong><br/>Cell: ${p.cell_id ?? 'Unknown'}`
    );
  }

  const summary =
    activeLayer === '24h' ? layer24h?.summary :
      activeLayer === '7day' ? layer7day?.daily_summaries?.[day - 1] :
        null;

  const activeGeojson =
    activeLayer === '24h' ? layer24h?.geojson_grid :
      activeLayer === '7day' ? layer7day?.geojson_snapshots?.[day - 1] :
        activeLayer === 'evi' ? layerEvi :
          null;

  const spreadArrows = activeLayer === '24h' ? layer24h?.geojson_arrows : null;

  const isNarrow = typeof window !== 'undefined' && window.innerWidth < 640;

  return (
    <div style={{ position: 'relative', height: '100%', width: '100%' }}>
      {loading && <div style={styles.overlayMessage}>Loading {activeLayer} layer…</div>}
      {error && <div style={{ ...styles.overlayMessage, color: '#e5605a' }}>{error}</div>}

      <MapContainer key={activeRegion} center={REGION_VIEWS[activeRegion].center} zoom={REGION_VIEWS[activeRegion].zoom} style={{ height: '100%', width: '100%' }}>
        <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {activeGeojson && (
          <GeoJSON
            key={`${activeLayer}-${day}`}
            data={activeGeojson}
            style={activeLayer === 'evi' ? eviStyle : riskStyle}
            onEachFeature={activeLayer === 'evi' ? onEachEviFeature : onEachRiskFeature}
          />
        )}
        {activeLayer === 'evi' && activeGeojson && <FitBounds geojson={activeGeojson} />}
        {spreadArrows && <SpreadArrows data={spreadArrows} />}

        {showFirms && layerFirms && (
          <GeoJSON
            key={`firms-${activeRegion}`}
            data={layerFirms}
            pointToLayer={(feature, latlng) => {
              // Find matching grid cell in layer24h to check FRI
              let isCritical = false;
              if (layer24h && layer24h.geojson_grid) {
                // simple distance check for cell matching
                const cell = layer24h.geojson_grid.features.find(f => {
                  const props = f.properties;
                  return props.latitude && props.longitude &&
                    Math.abs(props.latitude - latlng.lat) <= 0.05 &&
                    Math.abs(props.longitude - latlng.lng) <= 0.05;
                });
                if (cell && cell.properties.fri > 75) {
                  isCritical = true;
                }
              }

              const className = isCritical ? 'firms-alert-pulse' : 'firms-alert-dot';
              return L.marker(latlng, {
                icon: L.divIcon({ className, iconSize: isCritical ? [14, 14] : [12, 12] })
              }).bindPopup(`<strong>Active Fire (NASA FIRMS)</strong><br/>Acquired: ${feature.properties.acq_date} ${feature.properties.acq_time}<br/>Confidence: ${feature.properties.confidence}`);
            }}
          />
        )}
      </MapContainer>

      {summary && (
        <div style={{ ...styles.summaryCard, ...(isNarrow ? { left: '10px', right: '10px', width: 'auto' } : {}) }}>
          <div><strong>Mean risk:</strong> {(summary.mean_fire_prob * 100).toFixed(1)}%</div>
          <div><strong>High-risk cells:</strong> {summary.n_high_risk}</div>
        </div>
      )}

      <div style={{ position: 'absolute', bottom: '20px', left: isNarrow ? '10px' : '20px', right: isNarrow ? '10px' : 'auto', zIndex: 1000, display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        <LayerControl activeLayer={activeLayer} onChange={setActiveLayer} />
        <RegionSwitcher activeRegion={activeRegion} onChange={handleRegionChange} />
        <button
          style={{ ...styles.layerBtn, background: showFirms ? '#a33' : 'rgba(15, 21, 18, 0.8)' }}
          onClick={() => setShowFirms(!showFirms)}
        >
          {showFirms ? 'Hide Live Fires' : 'Show Live Fires'}
        </button>
      </div>
      {activeLayer === '7day' && layer7day && (
        <DaySlider day={day} maxDay={layer7day.n_forecast_days || 7} onChange={setDay} />
      )}
      {activeLayer !== 'evi' && <RiskLegend />}

      {localPanelCell && (
        <CellDetailPanel
          lat={localPanelCell.lat}
          lon={localPanelCell.lon}
          onClose={() => setLocalPanelCell(null)}
        />
      )}
    </div>
  );
}

const styles = {
  overlayMessage: { position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)', zIndex: 1000, background: 'rgba(15, 21, 18, 0.9)', color: '#e8ede9', padding: '8px 16px', borderRadius: '8px', fontSize: '0.85rem' },
  summaryCard: { position: 'absolute', top: '20px', left: '20px', zIndex: 1000, background: 'rgba(15, 21, 18, 0.9)', color: '#e8ede9', padding: '12px 16px', borderRadius: '8px', fontSize: '0.85rem', lineHeight: 1.6 },
  layerBtn: { border: '1px solid #4a7c59', color: '#e8ede9', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.85rem', whiteSpace: 'nowrap' },
};

function FitBounds({ geojson }) {
  const map = useMap();
  useEffect(() => {
    if (geojson && geojson.features && geojson.features.length > 0) {
      try {
        const bounds = L.geoJSON(geojson).getBounds();
        if (bounds.isValid()) {
          map.fitBounds(bounds, { padding: [20, 20] });
        }
      } catch (err) {
        // ignore
      }
    }
  }, [geojson, map]);
  return null;
}