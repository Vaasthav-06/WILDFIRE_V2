export default function DaySlider({ day, maxDay, onChange }) {
  return (
    <div style={styles.container}>
      <div style={styles.label}>Day {day} of {maxDay}</div>
      <input
        type="range"
        min={1}
        max={maxDay}
        value={day}
        onChange={(e) => onChange(Number(e.target.value))}
        style={styles.slider}
      />
    </div>
  );
}

const styles = {
  container: {
    position: 'absolute',
    bottom: '20px',
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 1000,
    background: 'rgba(15, 21, 18, 0.9)',
    color: '#e8ede9',
    padding: '10px 16px',
    borderRadius: '8px',
    width: '260px',
    textAlign: 'center',
  },
  label: { fontSize: '0.8rem', marginBottom: '6px' },
  slider: { width: '100%' },
};