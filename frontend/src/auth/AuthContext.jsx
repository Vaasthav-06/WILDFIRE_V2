import { createContext, useContext, useState, useCallback } from 'react';
import client from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem('wildfire_token'));

  const login = useCallback(async (username, password) => {
    const response = await client.post('/api/auth/login', { username, password });
    const { token: newToken } = response.data;
    localStorage.setItem('wildfire_token', newToken);
    setToken(newToken);
  }, []);

  const register = useCallback(async (username, password) => {
    await client.post('/api/auth/register', { username, password });
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('wildfire_token');
    setToken(null);
  }, []);

  return (
    <AuthContext.Provider value={{ token, login, register, logout, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}