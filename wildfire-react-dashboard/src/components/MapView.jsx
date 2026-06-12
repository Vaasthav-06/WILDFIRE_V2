import React, { useState, useEffect, useMemo } from 'react';
import Map, { Source, Layer } from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

// Move static layer definitions OUTSIDE the component to prevent recreation on every render
const gridLayer = {
  id: 'risk-grid',
  type: 'circle',
  paint: {
    'circle-radius': [
      'interpolate', ['linear'], ['zoom'],
      4, 2.5,
      10, 5,
      14, 8
    ],
    'circle-color': [
      'interpolate', ['linear'], ['get', 'fire_prob'],
      0.0, 'rgba(59, 130, 246, 0.6)',
      0.4, 'rgba(234, 179, 8, 0.8)',
      0.5, 'rgba(249, 115, 22, 0.9)',
      0.8, 'rgba(239, 68, 68, 1.0)'
    ],
    'circle-opacity': 1.0,
    'circle-stroke-width': 1.5,
    'circle-stroke-color': 'rgba(0, 0, 0, 0.5)'
  }
};

const arrowLayer = {
  id: 'risk-arrows',
  type: 'symbol',
  layout: {
    'icon-image': 'arrow-icon',
    'icon-rotate': ['get', 'spread_bearing_deg'],
    'icon-rotation-alignment': 'map',
    'icon-allow-overlap': true,
    'icon-size': [
      'match',
      ['get', 'spread_intensity'],
      'extreme', 1.0,
      'rapid', 0.8,
      'moderate', 0.6,
      'light', 0.4,
      0.5
    ]
  },
  paint: {
    'icon-opacity': 1.0,
    'icon-halo-color': '#000',
    'icon-halo-width': 1
  }
};

// Wrap in React.memo to completely prevent re-renders from parent App state changes
const MapView = React.memo(({ onSelectCell }) => {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch('/predictions.geojson')
      .then(res => res.json())
      .then(json => {
        // Convert Polygons to Points (centroids) so MapLibre doesn't draw a circle at all 4 corners
        const pointsFeatures = json.features.map(f => {
           if (f.geometry && f.geometry.type === 'Polygon') {
             const coords = f.geometry.coordinates[0];
             const lng = (coords[0][0] + coords[2][0]) / 2;
             const lat = (coords[0][1] + coords[2][1]) / 2;
             return {
               ...f,
               geometry: { type: 'Point', coordinates: [lng, lat] }
             };
           }
           return f;
        });
        setData({ ...json, features: pointsFeatures });
      })
      .catch(err => console.error("Could not load geojson:", err));
  }, []);

  // Filter high risk features to show arrows
  const arrowData = useMemo(() => {
    if (!data) return null;
    const features = data.features.filter(f => f.properties.fire_prob >= 0.5 && f.properties.spread_bearing_deg !== null);
    return { ...data, features };
  }, [data]);


  // Create an arrow image for the map
  useEffect(() => {
    // wait until map loads to add image
  }, []);

  return (
    <Map
      initialViewState={{
        longitude: 78.9629,
        latitude: 20.5937,
        zoom: 4.5
      }}
      mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
      mapLib={maplibregl}
      interactiveLayerIds={['risk-grid']}
      onClick={(e) => {
        if (e.features && e.features.length > 0) {
          const feature = e.features[0];
          feature.clickLngLat = [e.lngLat.lng, e.lngLat.lat];
          onSelectCell(feature);
        } else {
          onSelectCell(null);
        }
      }}
      onMouseEnter={(e) => {
        // Direct DOM manipulation for cursor avoids brutal React re-renders on every mouse move
        e.target.getCanvas().style.cursor = 'pointer';
      }}
      onMouseLeave={(e) => {
        e.target.getCanvas().style.cursor = '';
      }}
      onLoad={(e) => {
        const map = e.target;
        // Simple base64 arrow icon
        const arrowSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>`;
        const img = new Image(24, 24);
        img.onload = () => map.addImage('arrow-icon', img);
        img.src = 'data:image/svg+xml;base64,' + btoa(arrowSvg);
      }}
    >
      {data && (
        <Source type="geojson" data={data}>
          <Layer {...gridLayer} />
        </Source>
      )}
      
      {arrowData && (
        <Source type="geojson" data={arrowData}>
          <Layer {...arrowLayer} />
        </Source>
      )}
    </Map>
  );
});

export default MapView;
