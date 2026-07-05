export default function RiskLegend() {
  const tiers = [
    { label: 'Extreme', color: '#7a0d0d' },
    { label: 'High', color: '#d9622b' },
    { label: 'Moderate', color: '#e0b03a' },
    { label: 'Low', color: '#4a7c59' },
  ];

  return (
    <div style={styles.container}>
      <div style={styles.title}>Fire Risk</div>
      {tiers.map((t) => (
        <div key={t.label} style={styles.row}>
          <span style={{ ...styles.swatch, background: t.color }} />
          <span>{t.label}</span>
        </div>
      ))}
    </div>
  );
}

const styles = {
  container: { position: 'absolute', bottom: '20px', right: '20px', zIndex: 1000, background: 'rgba(15, 21, 18, 0.9)', color: '#e8ede9', padding: '10px 14px', borderRadius: '8px', fontSize: '0.8rem' },
  title: { fontWeight: 600, marginBottom: '6px' },
  row: { display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' },
  swatch: { width: '12px', height: '12px', borderRadius: '3px', display: 'inline-block' },
};