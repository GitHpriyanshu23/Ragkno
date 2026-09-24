import React, { useEffect } from 'react'
import { CircleHelp, X, Keyboard, FileText, Cloud, CheckCircle } from 'lucide-react'

export default function HelpModal({ isOpen, onClose }) {
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

  return (
    <div className="app-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="app-modal-card" role="dialog" aria-modal="true" onMouseDown={(e) => e.stopPropagation()}>
        <header className="app-modal-header">
          <h3><CircleHelp size={20} /> Help & Documentation</h3>
          <button className="app-modal-close-btn" type="button" onClick={onClose} aria-label="Close help">
            <X size={18} />
          </button>
        </header>

        <div className="app-modal-body">
          {/* Quick Start */}
          <section className="settings-section">
            <span className="settings-section-title">Quick Start Guide</span>
            <div className="settings-row" style={{ alignItems: 'flex-start' }}>
              <div className="settings-label">
                <strong>1. Add Your Data Sources</strong>
                <span>Click the <strong>+</strong> button in chat or go to <strong>Data Center</strong>. You can connect your Google Drive or upload PDFs, DOCX, and TXT files.</span>
              </div>
            </div>
            <div className="settings-row" style={{ alignItems: 'flex-start' }}>
              <div className="settings-label">
                <strong>2. Ask Anything in Chat</strong>
                <span>Type your question. RagKno searches your indexed documents using dense vectors and BM25 keywords, re-ranks the best chunks, and cites sources with `[1]` badges.</span>
              </div>
            </div>
            <div className="settings-row" style={{ alignItems: 'flex-start' }}>
              <div className="settings-label">
                <strong>3. Inspect Sources</strong>
                <span>Click on any citation pill or the source badge to view the exact passage and file name the AI referenced.</span>
              </div>
            </div>
          </section>

          {/* Keyboard Shortcuts */}
          <section className="settings-section">
            <span className="settings-section-title"><Keyboard size={14} style={{ display: 'inline', marginRight: '4px' }} /> Keyboard Shortcuts</span>
            <div className="shortcuts-grid">
              <span>Send Message</span>
              <kbd className="shortcut-kbd">Enter</kbd>

              <span>New line in input</span>
              <kbd className="shortcut-kbd">Shift + Enter</kbd>

              <span>Open Settings</span>
              <kbd className="shortcut-kbd">Shift + ⌘ + ,</kbd>

              <span>Close Modals / Drawers</span>
              <kbd className="shortcut-kbd">Escape</kbd>
            </div>
          </section>

          {/* FAQs */}
          <section className="settings-section">
            <span className="settings-section-title">Common Questions</span>
            <div className="settings-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
              <strong>Is my Google Drive data private?</strong>
              <span>Yes. All indexed documents and Drive tokens are strictly isolated to your Google account in Supabase. No other user can search or see your data.</span>
            </div>
            <div className="settings-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
              <strong>What file types are supported?</strong>
              <span>Currently PDF, DOCX, and plain TXT files from both local upload and Google Drive.</span>
            </div>
          </section>
        </div>

        <footer className="app-modal-footer">
          <button
            type="button"
            onClick={onClose}
            className="feedback-send-btn"
          >
            Got it
          </button>
        </footer>
      </div>
    </div>
  )
}
