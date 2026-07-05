import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from './AuthContext';

export default function RegisterPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await register(username, password);
      navigate('/login');
    } catch (err) {
      setError(err.response?.data?.error || 'Registration failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.wrapper}>
      <form onSubmit={handleSubmit} style={styles.form}>
        <h1 style={styles.title}>Create Account</h1>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          style={styles.input}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={styles.input}
          required
        />
        {error && <p style={styles.error}>{error}</p>}
        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? 'Creating…' : 'Register'}
        </button>
        <p style={styles.link}>
          Already have an account? <Link to="/login" style={{ color: '#d9622b' }}>Sign in</Link>
        </p>
      </form>
    </div>
  );
}

const styles = {
  wrapper: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0f1512' },
  form: { background: '#1a2420', padding: '2.5rem', borderRadius: '12px', width: '320px', display: 'flex', flexDirection: 'column', gap: '0.75rem' },
  title: { color: '#e8ede9', fontSize: '1.25rem', marginBottom: '0.5rem' },
  input: { padding: '0.65rem 0.8rem', borderRadius: '8px', border: '1px solid #2e3b34', background: '#0f1512', color: '#e8ede9' },
  button: { marginTop: '0.5rem', padding: '0.7rem', borderRadius: '8px', border: 'none', background: '#d9622b', color: '#fff', fontWeight: 600, cursor: 'pointer' },
  error: { color: '#e5605a', fontSize: '0.85rem' },
  link: { color: '#9db3a8', fontSize: '0.8rem', textAlign: 'center', marginTop: '0.5rem' },
};