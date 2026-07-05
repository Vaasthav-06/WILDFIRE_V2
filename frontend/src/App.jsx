import { Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import LoginPage from './auth/LoginPage';
import RegisterPage from './auth/RegisterPage';
import MapPage from './pages/MapPage';
import AnalyticsPage from './pages/AnalyticsPage';
import PermitsPage from './pages/PermitsPage';
import Chatbot from './components/Chatbot';
import { useAuth } from './auth/AuthContext';
import { LocationProvider } from './context/LocationContext';
import ErrorBoundary from './components/ErrorBoundary';

function ProtectedRoute({ children }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function NavBar() {
  const location = useLocation();
  const { logout } = useAuth();
  if (['/login', '/register'].includes(location.pathname)) return null;

  return (
    <nav style={navStyles.bar}>
      <Link to="/map" style={navStyles.link}>Map</Link>
      <Link to="/analytics" style={navStyles.link}>Analytics</Link>
      <Link to="/permits" style={navStyles.link}>Permits</Link>
      <button onClick={logout} style={navStyles.logout}>Logout</button>
    </nav>
  );
}

export default function App() {
  const location = useLocation();
  const showChatbot = !['/login', '/register'].includes(location.pathname);

  return (
    <LocationProvider>
      <NavBar />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/map" element={<ProtectedRoute><ErrorBoundary><MapPage /></ErrorBoundary></ProtectedRoute>} />
        <Route path="/analytics" element={<ProtectedRoute><ErrorBoundary><AnalyticsPage /></ErrorBoundary></ProtectedRoute>} />
        <Route path="/permits" element={<ProtectedRoute><ErrorBoundary><PermitsPage /></ErrorBoundary></ProtectedRoute>} />
        <Route path="*" element={<Navigate to="/map" replace />} />
      </Routes>
      {showChatbot && <Chatbot />}
    </LocationProvider>
  );
}

const navStyles = {
  bar: {
    display: 'flex', gap: '20px', padding: '12px 16px', background: '#1a2420',
    alignItems: 'center', flexWrap: 'wrap', rowGap: '8px',
  },
  link: { color: '#e8ede9', textDecoration: 'none', fontSize: '0.85rem', whiteSpace: 'nowrap' },
  logout: { marginLeft: 'auto', background: 'transparent', border: '1px solid #2e3b34', color: '#9db3a8', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.8rem' },
};