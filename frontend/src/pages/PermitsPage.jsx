import { useState, useEffect } from 'react';
import { checkZone, applyPermit } from '../api/permits';
import { useLocation } from '../context/LocationContext';

export default function PermitsPage() {
  const { selectedCell } = useLocation();
  const [lat, setLat] = useState('8.6');
  const [lon, setLon] = useState('77.4');
  const [zoneResult, setZoneResult] = useState(null);
  const [checking, setChecking] = useState(false);

  const [applicantName, setApplicantName] = useState('');
  const [purpose, setPurpose] = useState('');
  const [plannedDate, setPlannedDate] = useState('');
  const [applyResult, setApplyResult] = useState(null);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState(null);

  // Pre-fill lat/lon from map selection
  useEffect(() => {
    if (selectedCell) {
      setLat(String(selectedCell.lat));
      setLon(String(selectedCell.lon));
      setZoneResult(null);
    }
  }, [selectedCell]);

  async function handleCheckZone(e) {
    e.preventDefault();
    setChecking(true);
    setError(null);
    setZoneResult(null);
    try {
      const data = await checkZone(Number(lat), Number(lon));
      setZoneResult(data);
    } catch (err) {
      setError('Could not check zone status.');
    } finally {
      setChecking(false);
    }
  }

  async function handleApply(e) {
    e.preventDefault();
    setApplying(true);
    setError(null);
    setApplyResult(null);
    try {
      const data = await applyPermit({
        applicantName,
        purpose,
        lat: Number(lat),
        lon: Number(lon),
        plannedDate,
      });
      setApplyResult(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Permit application failed.');
    } finally {
      setApplying(false);
    }
  }

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>Fire Permits</h1>

      {selectedCell && (
        <div style={styles.contextBanner}>
          📍 Using map selection: {selectedCell.lat.toFixed(4)}°N, {selectedCell.lon.toFixed(4)}°E
        </div>
      )}

      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>Check Zone Status</h2>
        <form onSubmit={handleCheckZone} style={styles.form}>
          <label style={styles.label}>
            Latitude
            <input value={lat} onChange={(e) => setLat(e.target.value)} style={styles.input} />
          </label>
          <label style={styles.label}>
            Longitude
            <input value={lon} onChange={(e) => setLon(e.target.value)} style={styles.input} />
          </label>
          <button type="submit" disabled={checking} style={styles.button}>
            {checking ? 'Checking…' : 'Check Zone'}
          </button>
        </form>

        {zoneResult && (
          <div style={styles.resultBox}>
            <pre style={styles.pre}>{JSON.stringify(zoneResult, null, 2)}</pre>
          </div>
        )}
      </section>

      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>Apply for Permit</h2>
        <form onSubmit={handleApply} style={styles.form}>
          <label style={styles.label}>
            Applicant Name
            <input value={applicantName} onChange={(e) => setApplicantName(e.target.value)} style={styles.input} required />
          </label>
          <label style={styles.label}>
            Purpose
            <input value={purpose} onChange={(e) => setPurpose(e.target.value)} style={styles.input} required />
          </label>
          <label style={styles.label}>
            Planned Date
            <input type="date" value={plannedDate} onChange={(e) => setPlannedDate(e.target.value)} style={styles.input} />
          </label>
          <button type="submit" disabled={applying} style={styles.button}>
            {applying ? 'Submitting…' : 'Submit Application'}
          </button>
        </form>

        {applyResult && (
          <div style={styles.resultBox}>
            <pre style={styles.pre}>{JSON.stringify(applyResult, null, 2)}</pre>
          </div>
        )}
      </section>

      {error && <p style={{ color: '#e5605a' }}>{error}</p>}
    </div>
  );
}

const styles = {
  page: { padding: '32px', background: '#0f1512', minHeight: '100vh', color: '#e8ede9' },
  title: { fontSize: '1.5rem', marginBottom: '24px' },
  contextBanner: {
    marginBottom: '20px', padding: '10px 16px', borderRadius: '8px',
    background: 'rgba(74, 222, 128, 0.08)', border: '1px solid rgba(74, 222, 128, 0.25)',
    color: '#4ade80', fontSize: '0.85rem', fontWeight: 500,
  },
  section: { marginBottom: '32px', background: '#1a2420', padding: '20px', borderRadius: '12px' },
  sectionTitle: { fontSize: '1rem', marginBottom: '16px', color: '#9db3a8', textTransform: 'uppercase', letterSpacing: '0.05em' },
  form: { display: 'flex', flexDirection: 'column', gap: '12px', maxWidth: '360px' },
  label: { display: 'flex', flexDirection: 'column', fontSize: '0.85rem', gap: '4px', color: '#9db3a8' },
  input: { padding: '0.6rem 0.8rem', borderRadius: '8px', border: '1px solid #2e3b34', background: '#0f1512', color: '#e8ede9' },
  button: { padding: '0.65rem', borderRadius: '8px', border: 'none', background: '#d9622b', color: '#fff', fontWeight: 600, cursor: 'pointer', marginTop: '4px' },
  resultBox: { marginTop: '16px', background: '#0f1512', padding: '12px', borderRadius: '8px', maxHeight: '200px', overflowY: 'auto' },
  pre: { margin: 0, fontSize: '0.75rem', whiteSpace: 'pre-wrap', color: '#9db3a8' },
};
