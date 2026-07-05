import { useState, useRef, useEffect } from 'react';
import { sendMessage } from '../api/chatbot';
import { useLocation } from '../context/LocationContext';

const QUICK_ACTIONS = [
  { label: '🔥 Is this safe?', text: 'Is this location safe to enter right now?' },
  { label: '🌿 Can I burn here?', text: 'Can I do controlled burning or crop residue burning at this location today?' },
  { label: '❓ Why high risk?', text: 'Why is this area at high wildfire risk? Explain the key factors.' },
  { label: '📋 Permit advice?', text: 'Should I issue a permit for activities at this location today?' },
];

/** Render newlines as line breaks inside message bubbles */
function MessageText({ text }) {
  return (
    <span>
      {text.split('\n').map((line, i, arr) => (
        <span key={i}>
          {line}
          {i < arr.length - 1 && <br />}
        </span>
      ))}
    </span>
  );
}

export default function Chatbot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const bottomRef = useRef(null);
  const { selectedCell } = useLocation();

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // When a new cell is selected, inject a context-change notice
  const prevCellRef = useRef(null);
  useEffect(() => {
    if (!selectedCell) return;
    const key = `${selectedCell.lat}_${selectedCell.lon}`;
    if (prevCellRef.current === key) return;
    prevCellRef.current = key;
    if (messages.length > 0) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: `📍 Location changed to ${selectedCell.lat.toFixed(4)}°N, ${selectedCell.lon.toFixed(4)}°E`,
        },
      ]);
    }
  }, [selectedCell]);

  async function doSend(text) {
    if (!text.trim() || sending) return;

    const userMsg = { role: 'user', text };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput('');
    setSending(true);

    try {
      const data = await sendMessage(
        userMsg.text,
        nextMessages.map((m) => ({ role: m.role, text: m.text })),
        selectedCell?.lat ?? null,
        selectedCell?.lon ?? null,
      );
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.reply || data.response || 'No reply.' },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: '⚠️ Something went wrong. Please try again.' },
      ]);
    } finally {
      setSending(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    doSend(input);
  }

  function handleQuickAction(text) {
    if (!selectedCell) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: '📍 Please click a cell on the map first so I can check the real risk for your location.' },
      ]);
      return;
    }
    doSend(text);
  }

  /* ── Collapsed FAB ──────────────────────────────────────────────────── */
  if (!open) {
    return (
      <button
        id="chatbot-fab"
        onClick={() => setOpen(true)}
        style={styles.fab}
        title="Open Wildfire Assistant"
      >
        💬
        {selectedCell && <span style={styles.fabBadge} />}
      </button>
    );
  }

  /* ── Expanded widget ────────────────────────────────────────────────── */
  return (
    <div style={styles.widget}>
      {/* Header */}
      <div style={styles.header}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>🌲 Wildfire Assistant</span>
          {selectedCell ? (
            <span style={styles.locationBadge}>
              📍 {selectedCell.lat.toFixed(3)}°N, {selectedCell.lon.toFixed(3)}°E
            </span>
          ) : (
            <span style={styles.locationHint}>Click a map cell to analyse a location</span>
          )}
        </div>
        <button onClick={() => setOpen(false)} style={styles.closeBtn} title="Close">✕</button>
      </div>

      {/* Quick actions */}
      <div style={styles.quickBar}>
        {QUICK_ACTIONS.map((qa) => (
          <button
            key={qa.label}
            style={styles.quickBtn}
            onClick={() => handleQuickAction(qa.text)}
            disabled={sending}
            title={qa.text}
          >
            {qa.label}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div style={styles.body}>
        {messages.length === 0 && (
          <p style={styles.muted}>
            {selectedCell
              ? `Ask anything about ${selectedCell.lat.toFixed(2)}°N, ${selectedCell.lon.toFixed(2)}°E — or use a quick action above.`
              : 'Select a map cell, then ask about risk, permits, or patrol safety.'}
          </p>
        )}

        {messages.map((m, i) => {
          if (m.role === 'system') {
            return (
              <div key={i} style={styles.systemMsg}>{m.text}</div>
            );
          }
          return (
            <div key={i} style={m.role === 'user' ? styles.userBubble : styles.botBubble}>
              <MessageText text={m.text} />
            </div>
          );
        })}

        {sending && (
          <div style={styles.botBubble}>
            <span style={styles.typing}>
              <span>●</span><span>●</span><span>●</span>
            </span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} style={styles.inputRow}>
        <input
          id="chatbot-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={selectedCell ? 'Ask about this location…' : 'Type a message…'}
          style={styles.input}
          disabled={sending}
        />
        <button type="submit" disabled={sending || !input.trim()} style={styles.sendBtn}>
          {sending ? '⏳' : '↑'}
        </button>
      </form>
    </div>
  );
}

const styles = {
  fab: {
    position: 'fixed', bottom: '24px', right: '24px', zIndex: 2000,
    width: '54px', height: '54px', borderRadius: '50%', border: 'none',
    background: 'linear-gradient(135deg, #d9622b, #b8431c)',
    fontSize: '1.35rem', cursor: 'pointer',
    boxShadow: '0 4px 16px rgba(217,98,43,0.45)',
    transition: 'transform 0.15s ease, box-shadow 0.15s ease',
  },
  fabBadge: {
    position: 'absolute', top: '6px', right: '6px',
    width: '10px', height: '10px', borderRadius: '50%',
    background: '#4ade80', border: '2px solid #1a2420',
  },
  widget: {
    position: 'fixed', bottom: '24px', right: '24px', zIndex: 2000,
    width: '360px', maxHeight: '540px',
    background: '#1a2420', color: '#e8ede9',
    borderRadius: '16px', display: 'flex', flexDirection: 'column',
    boxShadow: '0 12px 40px rgba(0,0,0,0.6)', overflow: 'hidden',
    border: '1px solid #2e3b34',
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
    padding: '14px 16px 10px', background: '#0f1512',
    borderBottom: '1px solid #2e3b34',
  },
  locationBadge: {
    fontSize: '0.72rem', color: '#4ade80', fontWeight: 500,
  },
  locationHint: {
    fontSize: '0.72rem', color: '#6b8a77', fontStyle: 'italic',
  },
  closeBtn: {
    background: 'transparent', border: 'none', color: '#9db3a8',
    cursor: 'pointer', fontSize: '1rem', lineHeight: 1, paddingTop: '2px',
  },
  quickBar: {
    display: 'flex', gap: '6px', padding: '8px 10px',
    overflowX: 'auto', background: '#131d19',
    borderBottom: '1px solid #2e3b34',
    scrollbarWidth: 'none',
  },
  quickBtn: {
    flexShrink: 0, padding: '5px 10px', borderRadius: '20px',
    border: '1px solid #2e3b34', background: '#1a2420', color: '#c8ddd1',
    fontSize: '0.72rem', cursor: 'pointer', whiteSpace: 'nowrap',
    transition: 'background 0.15s ease, border-color 0.15s ease',
  },
  body: {
    flex: 1, padding: '12px', overflowY: 'auto',
    display: 'flex', flexDirection: 'column', gap: '8px',
    scrollbarWidth: 'thin', scrollbarColor: '#2e3b34 transparent',
  },
  muted: { color: '#6b8a77', fontSize: '0.8rem', lineHeight: 1.5, margin: 0 },
  systemMsg: {
    alignSelf: 'center', fontSize: '0.72rem', color: '#9db3a8',
    background: '#0f1512', padding: '4px 12px', borderRadius: '20px',
    border: '1px solid #2e3b34',
  },
  userBubble: {
    alignSelf: 'flex-end', background: 'linear-gradient(135deg, #d9622b, #b8431c)',
    padding: '9px 13px', borderRadius: '14px 14px 3px 14px',
    fontSize: '0.85rem', maxWidth: '85%', lineHeight: 1.45,
  },
  botBubble: {
    alignSelf: 'flex-start', background: '#243028',
    padding: '9px 13px', borderRadius: '14px 14px 14px 3px',
    fontSize: '0.85rem', maxWidth: '90%', lineHeight: 1.55,
    border: '1px solid #2e3b34',
  },
  typing: {
    display: 'flex', gap: '5px', alignItems: 'center',
    '& span': { animation: 'bounce 1.2s infinite' },
  },
  inputRow: {
    display: 'flex', gap: '8px', padding: '10px 12px',
    borderTop: '1px solid #2e3b34', background: '#0f1512',
  },
  input: {
    flex: 1, padding: '0.5rem 0.75rem', borderRadius: '10px',
    border: '1px solid #2e3b34', background: '#1a2420', color: '#e8ede9',
    fontSize: '0.85rem', outline: 'none',
  },
  sendBtn: {
    padding: '0.5rem 0.9rem', borderRadius: '10px', border: 'none',
    background: 'linear-gradient(135deg, #d9622b, #b8431c)', color: '#fff',
    cursor: 'pointer', fontSize: '1rem', fontWeight: 700,
    transition: 'opacity 0.15s ease',
  },
};
