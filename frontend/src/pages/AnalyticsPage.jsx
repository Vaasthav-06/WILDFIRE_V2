import { useEffect, useState } from 'react';
import Skeleton from '../components/Skeleton';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from 'recharts';
import {
  fetchFeatureImportance,
  fetchValidationHistory,
  fetchEviTrend,
  fetchFireHistoryStats,
} from '../api/analytics';

export default function AnalyticsPage() {
  const [featureImportance, setFeatureImportance] = useState(null);
  const [validationHistory, setValidationHistory] = useState(null);
  const [eviTrend, setEviTrend] = useState(null);
  const [fireHistory, setFireHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAll() {
      setLoading(true);
      setError(null);
      try {
        const [fi, vh, nt, fh] = await Promise.all([
          fetchFeatureImportance(),
          fetchValidationHistory(),
          fetchEviTrend(),
          fetchFireHistoryStats(),
        ]);
        if (!cancelled) {
          setFeatureImportance(fi);
          setValidationHistory(vh);
          setEviTrend(nt);
          setFireHistory(fh);
        }
      } catch (err) {
        if (!cancelled) setError('Could not load analytics data.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadAll();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
  return (
    <div style={styles.page}>
      <h1 style={styles.pageTitle}>Model & Fire Analytics</h1>
      {[1, 2, 3, 4].map((i) => (
        <section key={i} style={styles.section}>
          <Skeleton height={16} width="200px" style={{ marginBottom: '16px' }} />
          <Skeleton height={280} />
        </section>
      ))}
    </div>
  );
}
  if (error) return <div style={styles.page}><p style={{ color: '#e5605a' }}>{error}</p></div>;

  // Expecting: featureImportance.features = [{ name, importance }, ...]
  const featureRows = (featureImportance?.features || [])
    .slice()
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 12);

  // Expecting: validationHistory.runs = [{ date, auc, f1 }, ...]
  const validationRows = validationHistory?.runs || [];

  // Expecting: eviTrend.points = [{ date, mean_evi }, ...]
  const eviRows = eviTrend?.points || [];

  // Expecting: fireHistory = { total_fires, by_year: [{ year, count }], by_severity: [...] }
  const fireByYear = fireHistory?.by_year || [];

  return (
    <div style={styles.page}>
      <h1 style={styles.pageTitle}>Model & Fire Analytics</h1>

      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>Feature Importance</h2>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={featureRows} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e3b34" />
            <XAxis type="number" stroke="#9db3a8" />
            <YAxis type="category" dataKey="name" stroke="#9db3a8" width={100} />
            <Tooltip contentStyle={{ background: '#1a2420', border: '1px solid #2e3b34' }} />
            <Bar dataKey="importance" fill="#d9622b" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>Model Validation History</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={validationRows}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e3b34" />
            <XAxis dataKey="date" stroke="#9db3a8" />
            <YAxis domain={[0, 1]} stroke="#9db3a8" />
            <Tooltip contentStyle={{ background: '#1a2420', border: '1px solid #2e3b34' }} />
            <Legend />
            <Line type="monotone" dataKey="auc" stroke="#d9622b" strokeWidth={2} />
            <Line type="monotone" dataKey="f1" stroke="#4a7c59" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>EVI Trend (Vegetation Health)</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={eviRows}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e3b34" />
            <XAxis dataKey="date" stroke="#9db3a8" />
            <YAxis domain={[-1, 1]} stroke="#9db3a8" />
            <Tooltip contentStyle={{ background: '#1a2420', border: '1px solid #2e3b34' }} />
            <Line type="monotone" dataKey="mean_evi" stroke="#7fb88a" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section style={styles.section}>
        <h2 style={styles.sectionTitle}>Fire History by Year</h2>
        {fireHistory?.total_fires != null && (
          <p style={styles.muted}>Total recorded fires: {fireHistory.total_fires}</p>
        )}
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={fireByYear}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e3b34" />
            <XAxis dataKey="year" stroke="#9db3a8" />
            <YAxis stroke="#9db3a8" />
            <Tooltip contentStyle={{ background: '#1a2420', border: '1px solid #2e3b34' }} />
            <Bar dataKey="count" fill="#e0b03a" />
          </BarChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}

const styles = {
  page: { padding: '32px', background: '#0f1512', minHeight: '100vh', color: '#e8ede9' },
  pageTitle: { fontSize: '1.5rem', marginBottom: '24px' },
  section: { marginBottom: '40px', background: '#1a2420', padding: '20px', borderRadius: '12px' },
  sectionTitle: { fontSize: '1rem', marginBottom: '16px', color: '#9db3a8', textTransform: 'uppercase', letterSpacing: '0.05em' },
  muted: { color: '#9db3a8', fontSize: '0.85rem' },
};