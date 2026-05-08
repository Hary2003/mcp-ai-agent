// ChatWindow.jsx — fully self-contained, no external CSS file needed
// Just drop this file in src/components/ and import it. That's it.

import { useState, useRef, useEffect } from "react";
import API from "../api/api";

// ─── Inject styles once into <head> so no CSS file is needed ─────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, #root {
  height: 100%;
  margin: 0;
  padding: 0;
}

body {
  background: #0A0A0A;
  font-family: 'DM Sans', sans-serif;
  -webkit-font-smoothing: antialiased;
}

.mcp-shell {
  display: flex;
  flex-direction: column;
  height: 100dvh;
  max-width: 780px;
  margin: 0 auto;
  background: #0A0A0A;
}

/* ── Header ── */
.mcp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 24px 16px;
  border-bottom: 1px solid #242424;
  flex-shrink: 0;
}
.mcp-title {
  font-family: 'Instrument Serif', Georgia, serif;
  font-style: italic;
  font-size: 22px;
  color: #F0EDE8;
  letter-spacing: -0.02em;
  line-height: 1;
  margin: 0;
}
.mcp-subtitle {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: #8A8580;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-top: 4px;
}
.mcp-header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.mcp-chips {
  display: flex;
  gap: 5px;
}
.mcp-chip {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  padding: 3px 8px;
  border-radius: 99px;
  border: 1px solid #242424;
  color: #3D3D3D;
}
.mcp-status {
  display: flex;
  align-items: center;
  gap: 6px;
  background: #191919;
  border: 1px solid #242424;
  border-radius: 99px;
  padding: 5px 11px;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: #8A8580;
}
.mcp-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #22C55E;
  animation: mcp-pulse 2.4s ease-in-out infinite;
}
@keyframes mcp-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.5; }
}

/* ── Messages ── */
.mcp-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 24px 12px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  scroll-behavior: smooth;
}
.mcp-messages::-webkit-scrollbar { width: 4px; }
.mcp-messages::-webkit-scrollbar-thumb { background: #242424; border-radius: 2px; }

/* Empty state */
.mcp-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  text-align: center;
  padding: 60px 0;
}
.mcp-empty-icon {
  font-size: 32px;
  color: #3D3D3D;
  line-height: 1;
}
.mcp-empty-title {
  font-family: 'Instrument Serif', Georgia, serif;
  font-style: italic;
  font-size: 20px;
  color: #F0EDE8;
  margin: 0;
}
.mcp-empty-sub {
  font-size: 13px;
  color: #8A8580;
  max-width: 260px;
  line-height: 1.6;
}

/* Message row */
.mcp-row {
  display: flex;
  flex-direction: column;
  animation: mcp-in 0.26s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.mcp-row.user  { align-items: flex-end; }
.mcp-row.agent { align-items: flex-start; }
@keyframes mcp-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.mcp-meta {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 6px;
}
.mcp-row.user .mcp-meta { flex-direction: row-reverse; }

.mcp-av {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 500;
  flex-shrink: 0;
}
.mcp-av.user  { background: #78450A; color: #F59E0B; border: 1px solid #78450A; font-family: 'DM Mono', monospace; }
.mcp-av.agent { background: #191919; color: #8A8580; border: 1px solid #242424; }

.mcp-lbl {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: #3D3D3D;
}

.mcp-bubble {
  max-width: 76%;
  padding: 12px 15px;
  border-radius: 14px;
  font-size: 14px;
  line-height: 1.65;
  word-break: break-word;
  white-space: pre-wrap;
}
.mcp-bubble.user  {
  background: #1A1508;
  border: 1px solid #78450A;
  color: #F0EDE8;
  border-bottom-right-radius: 3px;
}
.mcp-bubble.agent {
  background: #111111;
  border: 1px solid #242424;
  color: #F0EDE8;
  border-bottom-left-radius: 3px;
}

/* Source badges */
.mcp-badges {
  display: flex;
  gap: 5px;
  margin-top: 8px;
  flex-wrap: wrap;
}
.mcp-badge {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 99px;
  border: 1px solid #242424;
  color: #8A8580;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.mcp-badge.notion { border-color: #444; color: #888; }
.mcp-badge.github { border-color: rgba(110,64,201,.6); color: #a78bfa; }
.mcp-badge.slack  { border-color: rgba(54,99,63,.6);  color: #4ade80; }
.mcp-badge.linear { border-color: rgba(26,95,158,.6); color: #60a5fa; }

/* Thinking */
.mcp-thinking {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  animation: mcp-in 0.26s cubic-bezier(0.16,1,0.3,1) both;
}
.mcp-thinking-bbl {
  background: #111;
  border: 1px solid #242424;
  border-radius: 14px;
  border-bottom-left-radius: 3px;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.mcp-thinking-lbl {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: #8A8580;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.mcp-dots { display: flex; gap: 5px; }
.mcp-dot-b {
  width: 5px; height: 5px; border-radius: 50%;
  background: #F59E0B;
  animation: mcp-bounce 1.2s ease-in-out infinite;
}
.mcp-dot-b:nth-child(2) { animation-delay: .18s; }
.mcp-dot-b:nth-child(3) { animation-delay: .36s; }
@keyframes mcp-bounce {
  0%,80%,100% { transform:scale(.7); opacity:.4; }
  40%         { transform:scale(1);  opacity:1;  }
}

/* Error */
.mcp-error {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(239,68,68,.06);
  border: 1px solid rgba(239,68,68,.25);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  color: #EF4444;
  max-width: 76%;
}

/* ── Input area ── */
.mcp-input-area {
  padding: 12px 18px 20px;
  border-top: 1px solid #1C1C1C;
  flex-shrink: 0;
}
.mcp-input-card {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  background: #111111;
  border: 1px solid #242424;
  border-radius: 14px;
  padding: 10px 10px 10px 15px;
  transition: border-color .2s, box-shadow .2s;
}
.mcp-input-card:focus-within {
  border-color: #78450A;
  box-shadow: 0 0 0 3px rgba(245,158,11,.1);
}
.mcp-textarea {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: #F0EDE8;
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  line-height: 1.5;
  resize: none;
  min-height: 22px;
  max-height: 120px;
  padding: 1px 0;
}
.mcp-textarea::placeholder { color: #3D3D3D; }
.mcp-send {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: #F59E0B;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: background .15s, transform .1s, opacity .15s;
}
.mcp-send:hover:not(:disabled) { background: #D97706; }
.mcp-send:active:not(:disabled) { transform: scale(0.93); }
.mcp-send:disabled { opacity: .3; cursor: not-allowed; }
.mcp-hint {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: #3D3D3D;
  text-align: center;
  margin-top: 9px;
  letter-spacing: .04em;
}
`;

function injectStyles() {
    if (document.getElementById("mcp-styles")) return;
    const el = document.createElement("style");
    el.id = "mcp-styles";
    el.textContent = CSS;
    document.head.appendChild(el);
}

// ─── Detect which sources the agent mentioned ─────────────────────────────────
function detectSources(text) {
    const map = { notion: /notion/i, github: /github/i, slack: /slack/i, linear: /linear/i };
    return Object.entries(map).filter(([, re]) => re.test(text)).map(([k]) => k);
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function ChatWindow() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const bottomRef = useRef(null);
    const textareaRef = useRef(null);

    // Inject CSS once on mount
    useEffect(() => { injectStyles(); }, []);

    // Scroll to bottom on new messages
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    const autoResize = () => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 120) + "px";
    };

    const sendMessage = async () => {
        const text = input.trim();
        if (!text || loading) return;

        setMessages(prev => [...prev, { role: "user", text }]);
        setInput("");
        setError("");
        setLoading(true);
        if (textareaRef.current) textareaRef.current.style.height = "auto";

        try {
            const res = await API.post("/chat", { message: text });
            setMessages(prev => [...prev, { role: "agent", text: res.data.response }]);
        } catch (err) {
            setError(
                err.response?.data?.detail ||
                "Could not reach the agent. Is the backend running on port 8000?"
            );
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="mcp-shell">

            {/* Header */}
            <header className="mcp-header">
                <div>
                    <h1 className="mcp-title">MCP Agent</h1>
                    <p className="mcp-subtitle">Multi-source AI assistant</p>
                </div>
                <div className="mcp-header-right">
                    <div className="mcp-chips">
                        {["Notion", "GitHub", "Slack", "Linear"].map(s => (
                            <span key={s} className="mcp-chip">{s}</span>
                        ))}
                    </div>
                    <div className="mcp-status">
                        <span className="mcp-dot" />
                        online
                    </div>
                </div>
            </header>

            {/* Messages */}
            <main className="mcp-messages">
                {messages.length === 0 && !loading && (
                    <div className="mcp-empty">
                        <div className="mcp-empty-icon">✦</div>
                        <p className="mcp-empty-title">Ask anything</p>
                        <p className="mcp-empty-sub">
                            I'll search Notion, GitHub, Slack, and Linear to find the best answer.
                        </p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div key={i} className={`mcp-row ${msg.role}`}>
                        <div className="mcp-meta">
                            <div className={`mcp-av ${msg.role}`}>
                                {msg.role === "user" ? "U" : "✦"}
                            </div>
                            <span className="mcp-lbl">{msg.role === "user" ? "You" : "Agent"}</span>
                        </div>
                        <div className={`mcp-bubble ${msg.role}`}>{msg.text}</div>
                        {msg.role === "agent" && (() => {
                            const srcs = detectSources(msg.text);
                            return srcs.length > 0 ? (
                                <div className="mcp-badges">
                                    {srcs.map(s => <span key={s} className={`mcp-badge ${s}`}>{s}</span>)}
                                </div>
                            ) : null;
                        })()}
                    </div>
                ))}

                {loading && (
                    <div className="mcp-thinking">
                        <div className="mcp-meta">
                            <div className="mcp-av agent">✦</div>
                            <span className="mcp-lbl">Agent</span>
                        </div>
                        <div className="mcp-thinking-bbl">
                            <span className="mcp-thinking-lbl">Searching</span>
                            <div className="mcp-dots">
                                <span className="mcp-dot-b" />
                                <span className="mcp-dot-b" />
                                <span className="mcp-dot-b" />
                            </div>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="mcp-error">⚠ {error}</div>
                )}

                <div ref={bottomRef} />
            </main>

            {/* Input */}
            <footer className="mcp-input-area">
                <div className="mcp-input-card">
                    <textarea
                        ref={textareaRef}
                        className="mcp-textarea"
                        placeholder="Ask about your codebase, docs, or team..."
                        value={input}
                        rows={1}
                        onChange={(e) => { setInput(e.target.value); autoResize(); }}
                        onKeyDown={handleKeyDown}
                        disabled={loading}
                    />
                    <button
                        className="mcp-send"
                        onClick={sendMessage}
                        disabled={loading || !input.trim()}
                        aria-label="Send"
                    >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                            stroke="#000" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13" />
                            <polygon points="22 2 15 22 11 13 2 9 22 2" />
                        </svg>
                    </button>
                </div>
                <p className="mcp-hint">Enter to send · Shift+Enter for new line</p>
            </footer>
        </div>
    );
}
