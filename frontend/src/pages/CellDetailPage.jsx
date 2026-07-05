import { useEffect, useState } from 'react';
import { fetchCellDetail } from '../api/risk';

export default function CellDetailPanel({ lat, lon, onClose }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (lat == null || lon == null) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchCellDetail(lat, lon)
      .then((data) => { if (!cancelled) setDetail(data); })
      .catch(() => { if (!cancelled) setError('Could not load cell detail.'); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [lat, lon]);

  if (lat == null || lon == null) return null;

  return (
    <div style={styles.panel}>
      <button onClick={onClose} style={styles.closeBtn}>✕</button>

      {loading && <p style={styles.muted}>Loading explanation…</p>}
      {error && <p style={{ color: '#e5605a' }}>{error}</p>}

      {detail && (
        <>
          <h2 style={styles.title}>
            {(detail.fire_prob * 100).toFixed(1)}% risk
          </h2>
          <div style={styles.tier}>{detail.risk_tier}</div>

          <p style={styles.coords}>
            {lat.toFixed(4)}, {lon.toFixed(4)}
          </p>

          {detail.fri != null && (
            <div style={styles.row}>
              <span>Fire Risk Index (FRI)</span>
              <strong>{detail.fri} / 100 ({detail.fri_tier})</strong>
            </div>
          )}

          {detail.confidence != null && (
            <div style={styles.row}>
              <span>Model confidence</span>
              <strong>{Math.round(detail.confidence * 100)}%</strong>
            </div>
          )}

          {detail.model_std != null && (
            <div style={styles.row}>
              <span>Uncertainty (σ)</span>
              <strong>{detail.model_std.toFixed(3)}</strong>
            </div>
          )}

          {Array.isArray(detail.top_reasons) && detail.top_reasons.length > 0 && (
            <>
              <h3 style={styles.sectionTitle}>Top Reasons</h3>
              <ul style={styles.list}>
                {detail.top_reasons.map((reason, i) => (
                  <li key={i} style={styles.listItem}>{reason}</li>
                ))}
              </ul>
            </>
          )}

          {Array.isArray(detail.mitigating_factors) && detail.mitigating_factors.length > 0 && (
            <>
              <h3 style={styles.sectionTitle}>Mitigating Factors</h3>
              <ul style={styles.list}>
                {detail.mitigating_factors.map((factor, i) => (
                  <li key={i} style={{ ...styles.listItem, color: '#7fb88a' }}>{factor}</li>
                ))}
              </ul>
            </>
          )}

          {detail.spread_bearing_deg != null && (
            <div style={styles.row}>
              <span>Spread direction</span>
              <strong>{Math.round(detail.spread_bearing_deg)}°</strong>
            </div>
          )}

          {detail.shap_values && (
            <>
              <h3 style={styles.sectionTitle}>Feature Contributions (SHAP)</h3>
              <div style={styles.shapList}>
                {Object.entries(detail.shap_values)
                  .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                  .slice(0, 8)
                  .map(([feature, value]) => (
                    <div key={feature} style={styles.shapRow}>
                      <span style={styles.shapLabel}>{feature}</span>
                      <div style={styles.shapBarTrack}>
                        <div
                          style={{
                            ...styles.shapBarFill,
                            width: `${Math.min(Math.abs(value) * 200, 100)}%`,
                            background: value > 0 ? '#d9622b' : '#4a7c59',
                            marginLeft: value > 0 ? '50%' : undefined,
                            marginRight: value < 0 ? '50%' : undefined,
                          }}
                        />
                      </div>
                      <span style={styles.shapValue}>{value.toFixed(3)}</span>
                    </div>
                  ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

const styles = {
  panel: {
    position: 'absolute',
    top: 0,
    right: 0,
    height: '100%',
    width: '360px',
    background: '#1a2420',
    color: '#e8ede9',
    padding: '24px',
    overflowY: 'auto',
    zIndex: 1500,
    boxShadow: '-4px 0 16px rgba(0,0,0,0.4)',
  },
  closeBtn: {
    position: 'absolute',
    top: '16px',
    right: '16px',
    background: 'transparent',
    border: 'none',
    color: '#e8ede9',
    fontSize: '1.1rem',
    cursor: 'pointer',
  },
  title: { fontSize: '1.6rem', margin: '0 0 4px 0' },
  tier: { color: '#d9622b', fontWeight: 600, marginBottom: '4px' },
  coords: { color: '#9db3a8', fontSize: '0.8rem', marginBottom: '20px' },
  row: { display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #2e3b34', fontSize: '0.9rem' },
  sectionTitle: { fontSize: '0.95rem', marginTop: '20px', marginBottom: '8px', color: '#9db3a8', textTransform: 'uppercase', letterSpacing: '0.05em' },
  list: { margin: 0, paddingLeft: '18px' },
  listItem: { fontSize: '0.9rem', marginBottom: '6px', lineHeight: 1.4 },
  muted: { color: '#9db3a8' },
  shapList: { display: 'flex', flexDirection: 'column', gap: '8px' },
  shapRow: { display: 'grid', gridTemplateColumns: '100px 1fr 50px', alignItems: 'center', gap: '8px', fontSize: '0.75rem' },
  shapLabel: { color: '#9db3a8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  shapBarTrack: { position: 'relative', height: '10px', background: '#0f1512', borderRadius: '4px', overflow: 'hidden' },
  shapBarFill: { position: 'absolute', top: 0, bottom: 0, borderRadius: '4px' },
  shapValue: { textAlign: 'right', color: '#e8ede9' },
};