import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={styles.box}>
          <p style={styles.title}>Something went wrong in this section.</p>
          <p style={styles.detail}>{this.state.error?.message}</p>
          <button onClick={() => this.setState({ hasError: false, error: null })} style={styles.retry}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const styles = {
  box: { padding: '24px', background: '#1a2420', color: '#e8ede9', borderRadius: '12px', textAlign: 'center' },
  title: { fontWeight: 600, marginBottom: '8px' },
  detail: { color: '#9db3a8', fontSize: '0.8rem', marginBottom: '12px' },
  retry: { background: '#d9622b', border: 'none', color: '#fff', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer' },
};
