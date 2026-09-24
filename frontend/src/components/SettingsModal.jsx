import React, { useState, useEffect } from 'react'
import { Settings, X, Shield, Cpu, Sliders, Trash2, Check, User } from 'lucide-react'

export default function SettingsModal({ isOpen, onClose, user, onClearHistory, onToast }) {
  const [topK, setTopK] = useState(() => Number(localStorage.getItem('ragkno_top_k') || 5))
  const [useReranker, setUseReranker] = useState(() => localStorage.getItem('ragkno_reranker') !== 'false')
  const [streaming, setStreaming] = useState(() => localStorage.getItem('ragkno_streaming') !== 'false')
  const [savedNotice, setSavedNotice] = useState(false)

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown)
      return () => window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  const handleSave = () => {
    localStorage.setItem('ragkno_top_k', String(topK))
    localStorage.setItem('ragkno_reranker', String(useReranker))
    localStorage.setItem('ragkno_streaming', String(streaming))
    setSavedNotice(true)
    setTimeout(() => setSavedNotice(false), 2000)
    onToast?.({ type: 'success', message: 'Settings saved successfully.' })
  }

  return (
    <div className="app-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="app-modal-card" role="dialog" aria-modal="true" onMouseDown={(e) => e.stopPropagation()}>
        <header className="app-modal-header">
          <h3><Settings size={20} /> Settings</h3>
          <button className="app-modal-close-btn" type="button" onClick={onClose} aria-label="Close settings">
            <X size={18} />
          </button>
        </header>

        <div className="app-modal-body">
          {/* Account Profile Section */}
          <section className="settings-section">
            <span className="settings-section-title">Account Profile</span>
            <div className="settings-row">
              <div className="settings-label">
                <strong>{user?.name || 'Google User'}</strong>
                <span>{user?.email || 'Logged in via Google OAuth'}</span>
              </div>
              <span className="shortcut-kbd">Connected</span>
            </div>
          </section>

          {/* AI Model Section */}
          <section className="settings-section">
            <span className="settings-section-title">AI & Reasoning Engine</span>
            <div className="settings-row">
              <div className="settings-label">
                <strong>Primary LLM Model</strong>
                <span>Google Gemini 2.5 Flash (Ultra-fast multimodal)</span>
              </div>
              <span className="shortcut-kbd">gemini-2.5-flash</span>
            </div>

            <div className="settings-row">
              <div className="settings-label">
                <strong>Real-time Streaming</strong>
                <span>Stream AI tokens as they are generated</span>
              </div>
              <input
                type="checkbox"
                checked={streaming}
                onChange={(e) => setStreaming(e.target.checked)}
                style={{ width: '18px', height: '18px', cursor: 'pointer' }}
              />
            </div>
          </section>

          {/* RAG Retrieval Tuning */}
          <section className="settings-section">
            <span className="settings-section-title">RAG Retrieval Tuning</span>
            <div className="settings-row">
              <div className="settings-label">
                <strong>Retrieved Documents (Top-K): {topK}</strong>
                <span>Number of source document chunks used per answer</span>
              </div>
              <input
                type="range"
                min="2"
                max="10"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{ width: '120px', cursor: 'pointer' }}
              />
            </div>

            <div className="settings-row">
              <div className="settings-label">
                <strong>Semantic CrossEncoder Re-ranking</strong>
                <span>Rerank chunks using ms-marco-MiniLM-L-6-v2</span>
              </div>
              <input
                type="checkbox"
                checked={useReranker}
                onChange={(e) => setUseReranker(e.target.checked)}
                style={{ width: '18px', height: '18px', cursor: 'pointer' }}
              />
            </div>
          </section>

          {/* Data Actions */}
          <section className="settings-section">
            <span className="settings-section-title">Data & Chat History</span>
            <div className="settings-row">
              <div className="settings-label">
                <strong>Clear All Local Thread Cache</strong>
                <span>Reset stored chat threads for this browser session</span>
              </div>
              <button
                type="button"
                onClick={() => {
                  if (window.confirm('Are you sure you want to clear your local thread cache?')) {
                    onClearHistory?.()
                    onToast?.({ type: 'info', message: 'Local chat cache cleared.' })
                  }
                }}
                className="app-modal-close-btn"
                style={{ color: '#ef4444', border: '1px solid #fecaca', padding: '0.4rem 0.8rem', borderRadius: '8px' }}
              >
                <Trash2 size={16} style={{ marginRight: '4px' }} /> Clear Cache
              </button>
            </div>
          </section>
        </div>

        <footer className="app-modal-footer">
          {savedNotice && <span style={{ color: '#16a34a', fontSize: '13px', fontWeight: 600 }}>Saved!</span>}
          <button
            type="button"
            onClick={onClose}
            className="app-modal-close-btn"
            style={{ padding: '0.5rem 1rem', border: '1px solid #cbd5e1' }}
          >
            Close
          </button>
          <button
            type="button"
            onClick={handleSave}
            className="feedback-send-btn"
          >
            <Check size={16} style={{ marginRight: '4px' }} /> Save Changes
          </button>
        </footer>
      </div>
    </div>
  )
}
