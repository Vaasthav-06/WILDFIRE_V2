import { GeoJSON } from 'react-leaflet';

function arrowStyle() {
  return { color: '#ffd166', weight: 2, opacity: 0.85 };
}

export default function SpreadArrows({ data }) {
  if (!data || !data.features?.length) return null;
  return <GeoJSON data={data} style={arrowStyle} />;
}