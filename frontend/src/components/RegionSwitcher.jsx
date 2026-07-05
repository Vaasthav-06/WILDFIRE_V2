import { useEffect, useState } from 'react';
import { fetchRegions } from '../api/risk';

export default function RegionSwitcher({ activeRegion, onChange }) {
  const [regions, setRegions] = useState([]);

  useEffect(() => {
    fetchRegions().then(setRegions).catch(() => setRegions([]));
  }, []);

  if (regions.length === 0) return null;

  return (
    <select
      value={activeRegion}
      onChange={(e) => onChange(e.target.value)}
      style={styles.select}
    >
      {regions.map((r) => (
        <option key={r.key} value={r.key}>{r.label}</option>
      ))}
    </select>
  );
}

const styles = {
  select: {
    background: '#1a2420',
    color: '#e8ede9',
    border: '1px solid #2e3b34',
    borderRadius: '6px',
    padding: '6px 10px',
    fontSize: '0.8rem',
  },
};
