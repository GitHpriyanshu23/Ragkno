import React, { useEffect } from 'react'
import { Info, X, ShieldCheck, Database, Cpu, Layers } from 'lucide-react'

export default function AboutModal({ isOpen, onClose }) {
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
          <h3><Info size={20} /> About RagKno</h3>
          <button className="app-modal-close-btn" type="button" onClick={onClose} aria-label="Close about">
            <X size={18} />
          </button>
        </header>

        <div className="app-modal-body">
          <p style={{ fontSize: '0.95rem', lineHeight: '1.6', color: '#52525b', margin: 0 }}>
            <strong>RagKno</strong> is a high-performance Retrieval-Augmented Generation (RAG) platform designed to search, synthesize, and answer questions across your Google Drive and documents with strict user privacy.
          </p>

          <section className="settings-section">
            <span className="settings-section-title">Core Technology Stack</span>
            <div className="settings-row">
              <div className="settings-label">
                <strong><Layers size={16} style={{ display: 'inline', marginRight: '6px' }} /> Hybrid Retrieval Engine</strong>
                <span>Combines Dense Vector Search (all-MiniLM-L6-v2) + Sparse BM25 Keyword Search using Reciprocal Rank Fusion (RRF).</span>
              </div>
            </div>

            <div className="settings-row">
              <div className="settings-label">
                <strong><Cpu size={16} style={{ display: 'inline', marginRight: '6px' }} /> Semantic CrossEncoder Re-ranking</strong>
                <span>Applies ms-marco-MiniLM-L-6-v2 cross-attention scoring to verify context relevance before passing to the LLM.</span>
              </div>
            </div>

            <div className="settings-row">
              <div className="settings-label">
                <strong><Database size={16} style={{ display: 'inline', marginRight: '6px' }} /> Supabase PostgreSQL + ChromaDB</strong>
                <span>Stores chat threads, message histories, and Google Drive tokens with multi-tenant row-level scoping.</span>
              </div>
            </div>

            <div className="settings-row">
              <div className="settings-label">
                <strong><ShieldCheck size={16} style={{ display: 'inline', marginRight: '6px' }} /> Strict Privacy Isolation</strong>
                <span>Each user's indexed data and Google Drive tokens are cryptographically isolated. Zero cross-user data leakage.</span>
              </div>
            </div>
          </section>

          <div style={{ padding: '0.75rem 1rem', background: '#f8fafc', borderRadius: '12px', fontSize: '0.82rem', color: '#64748b', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>RagKno Core v1.0.0</span>
            <span style={{ fontWeight: 600 }}>Created by Priyanshu Urmaliya</span>
          </div>
        </div>

        <footer className="app-modal-footer">
          <button
            type="button"
            onClick={onClose}
            className="feedback-send-btn"
          >
            Close
          </button>
        </footer>
      </div>
    </div>
  )
}
