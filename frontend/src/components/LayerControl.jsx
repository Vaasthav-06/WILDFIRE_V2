export default function LayerControl({ activeLayer, onChange }) {
  const layers = [
    { id: '24h', label: '24h Risk' },
    { id: '7day', label: '7-Day Forecast' },
    { id: 'evi', label: 'EVI' },
  ];

  return (
    <div style={styles.container}>
      {layers.map((l) => (
        <button
          key={l.id}
          onClick={() => onChange(l.id)}
          style={{
            ...styles.button,
            ...(activeLayer === l.id ? styles.active : {}),
          }}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
}

const styles = {
  container: {
    position: 'absolute',
    top: '20px',
    right: '20px',
    zIndex: 1000,
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    background: 'rgba(15, 21, 18, 0.9)',
    padding: '8px',
    borderRadius: '8px',
  },
  button: {
    background: 'transparent',
    border: '1px solid #2e3b34',
    color: '#e8ede9',
    padding: '8px 14px',
    borderRadius: '6px',
    fontSize: '0.8rem',
    cursor: 'pointer',
    textAlign: 'left',
  },
  active: {
    background: '#d9622b',
    border: '1px solid #d9622b',
    fontWeight: 600,
  },
};