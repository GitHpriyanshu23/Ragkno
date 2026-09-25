import React, { useState, useEffect } from 'react'
import {
  Settings,
  X,
  Search,
  Cpu,
  Database,
  CircleHelp,
  Info,
  ChevronDown,
  Shield,
  Trash2,
  Layers,
  Sparkles,
  ExternalLink,
  Download,
  LogOut,
  RefreshCw,
  Activity,
  ArrowRight,
  MessageSquareHeart,
} from 'lucide-react'
import { syncDrive, API_BASE } from '@/api'
import { useI18n } from '../lib/i18n.jsx'

const NAV_TABS = [
  { id: 'general', label: 'General', icon: Settings },
  { id: 'ai', label: 'AI & RAG Engine', icon: Cpu },
  { id: 'data', label: 'Data Controls', icon: Database },
  { id: 'help', label: 'Help & FAQ', icon: CircleHelp },
  { id: 'about', label: 'About RagKno', icon: Info },
]

export function applyTheme() {
  try {
    document.documentElement.classList.remove('dark')
    document.documentElement.removeAttribute('data-theme')
  } catch {}
}

export default function SettingsModal({
  isOpen,
  onClose,
  initialTab = 'general',
  user,
  onClearHistory,
  onToast,
  onNavigateData,
  onSampleQuestion,
  onLogout,
  onOpenFeedback,
}) {
  const { language: currentLang, setLanguage: setI18nLanguage, t } = useI18n()
  const [activeTab, setActiveTab] = useState(initialTab)
  const [searchQuery, setSearchQuery] = useState('')

  // Settings State persisted in localStorage
  const [streaming, setStreaming] = useState(() => localStorage.getItem('ragkno_streaming') !== 'false')
  const [useReranker, setUseReranker] = useState(() => localStorage.getItem('ragkno_reranker') !== 'false')
  const [topK, setTopK] = useState(() => Number(localStorage.getItem('ragkno_top_k') || 5))
  const [model, setModel] = useState(() => localStorage.getItem('ragkno_model') || 'gemini-2.5-flash')
  const [language, setLanguage] = useState(() => currentLang || localStorage.getItem('ragkno_lang') || 'auto')

  // Operational states
  const [isSyncingDrive, setIsSyncingDrive] = useState(false)
  const [healthStatus, setHealthStatus] = useState(null)
  const [testingHealth, setTestingHealth] = useState(false)

  // Sync active tab when initialTab or isOpen changes
  useEffect(() => {
    if (isOpen) {
      setActiveTab(initialTab || 'general')
      setSearchQuery('')
    }
  }, [isOpen, initialTab])

  // Handle escape key
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose?.()
    }
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown)
      return () => window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  // Setting update handlers with automatic localStorage persistence
  const handleToggleStreaming = () => {
    const nextVal = !streaming
    setStreaming(nextVal)
    localStorage.setItem('ragkno_streaming', String(nextVal))
    onToast?.({ type: 'info', message: `Response streaming ${nextVal ? 'enabled' : 'disabled'}` })
  }

  const handleToggleReranker = () => {
    const nextVal = !useReranker
    setUseReranker(nextVal)
    localStorage.setItem('ragkno_reranker', String(nextVal))
    onToast?.({ type: 'info', message: `Semantic CrossEncoder ${nextVal ? 'enabled' : 'disabled'}` })
  }

  const handleTopKChange = (e) => {
    const val = Number(e.target.value)
    setTopK(val)
    localStorage.setItem('ragkno_top_k', String(val))
    onToast?.({ type: 'info', message: `Retrieval context set to ${val} chunks` })
  }

  const handleModelChange = (e) => {
    const val = e.target.value
    setModel(val)
    localStorage.setItem('ragkno_model', val)
    onToast?.({ type: 'info', message: `AI reasoning model set to ${val}` })
  }

  const handleLanguageChange = (e) => {
    const val = e.target.value
    setLanguage(val)
    setI18nLanguage(val)
  }

  const handleExportConversations = () => {
    try {
      const rawThreads = localStorage.getItem('rag_threads_store_v1')
      const threads = rawThreads ? JSON.parse(rawThreads) : []
      const exportPayload = {
        app: 'RagKno',
        version: '1.0.0',
        exportedAt: new Date().toISOString(),
        user: user?.email || 'anonymous',
        threadsCount: threads.length,
        threads,
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exportPayload, null, 2))
      const downloadAnchor = document.createElement('a')
      downloadAnchor.setAttribute('href', dataStr)
      downloadAnchor.setAttribute('download', `ragkno-conversations-${new Date().toISOString().slice(0, 10)}.json`)
      document.body.appendChild(downloadAnchor)
      downloadAnchor.click()
      downloadAnchor.remove()
      onToast?.({ type: 'success', message: `Exported ${threads.length} conversation threads as JSON` })
    } catch (err) {
      console.error('Export failed:', err)
      onToast?.({ type: 'error', message: 'Failed to export conversations' })
    }
  }

  const handleSyncDriveNow = async () => {
    setIsSyncingDrive(true)
    try {
      const res = await syncDrive()
      onToast?.({
        type: 'success',
        message: `Google Drive synced successfully (${res.files_indexed || 0} files up to date)`,
      })
    } catch (err) {
      console.error('Drive sync failed:', err)
      onToast?.({ type: 'error', message: err.message || 'Failed to sync Google Drive' })
    } finally {
      setIsSyncingDrive(false)
    }
  }

  const handleCheckHealth = async () => {
    setTestingHealth(true)
    const t0 = performance.now()
    try {
      const res = await fetch(`${API_BASE}/health`)
      const t1 = performance.now()
      const data = await res.json().catch(() => ({}))
      const latency = Math.round(t1 - t0)
      setHealthStatus({
        ok: true,
        latency,
        status: data.status || 'healthy',
        timestamp: new Date().toLocaleTimeString(),
      })
      onToast?.({ type: 'success', message: `Backend operational (${latency}ms roundtrip)` })
    } catch (err) {
      setHealthStatus({
        ok: false,
        error: err.message,
        timestamp: new Date().toLocaleTimeString(),
      })
      onToast?.({ type: 'error', message: 'Backend health check failed' })
    } finally {
      setTestingHealth(false)
    }
  }

  if (!isOpen) return null

  // Filter tabs if search query matches tab title or keywords
  const filteredTabs = NAV_TABS.filter((tab) => {
    if (!searchQuery.trim()) return true
    const q = searchQuery.toLowerCase()
    return (
      tab.label.toLowerCase().includes(q) ||
      (tab.id === 'ai' && (q.includes('model') || q.includes('stream') || q.includes('rag') || q.includes('gemini') || q.includes('rerank'))) ||
      (tab.id === 'data' && (q.includes('cache') || q.includes('drive') || q.includes('clear') || q.includes('history') || q.includes('export') || q.includes('sync'))) ||
      (tab.id === 'help' && (q.includes('guide') || q.includes('faq') || q.includes('shortcut') || q.includes('feedback') || q.includes('how')))
    )
  })

  return (
    <div className="settings-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        className="settings-modal-dialog"
        role="dialog"
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
      >
        {/* Left Sidebar */}
        <aside className="settings-modal-sidebar">
          <div className="settings-sidebar-top">
            <button
              type="button"
              className="settings-close-btn"
              onClick={onClose}
              aria-label="Close settings"
              title="Close (Esc)"
            >
              <X size={16} />
            </button>
          </div>

          <div className="settings-search-wrap">
            <Search size={14} className="settings-search-icon" />
            <input
              type="text"
              placeholder="Search settings"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="settings-search-input"
            />
          </div>

          <nav className="settings-nav-list" aria-label="Settings navigation">
            {filteredTabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id
              const tabLabels = {
                general: t('tabGeneral') || 'General',
                ai: t('tabAiRag') || 'AI & RAG Engine',
                data: t('tabData') || 'Data Controls',
                help: t('tabHelp') || 'Help & FAQ',
                about: t('tabAbout') || 'About RagKno',
              }
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`settings-nav-item ${isActive ? 'active' : ''}`}
                >
                  <Icon size={16} />
                  <span>{tabLabels[tab.id] || tab.label}</span>
                </button>
              )
            })}
          </nav>
        </aside>

        {/* Right Content Area */}
        <main className="settings-modal-content">
          {/* GENERAL TAB */}
          {activeTab === 'general' && (
            <>
              <header className="settings-content-header">
                <h2>{t('tabGeneral') || 'General Settings'}</h2>
              </header>
              <div className="settings-content-scroll">
                {/* Account Banner */}
                <div className="settings-banner-card">
                  <div className="settings-banner-icon">
                    <Shield size={20} />
                  </div>
                  <div className="settings-banner-text">
                    <h4>Account & Privacy Protection</h4>
                    <p>
                      Logged in as <strong>{user?.name || 'Google User'}</strong> ({user?.email || 'OAuth session'}).
                      Your personal Drive files, indices, and chat logs are cryptographically scoped to your identity.
                    </p>
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                      <span className="settings-badge-success">Isolated & Encrypted</span>
                    </div>
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">{t('language')}</span>
                    <span className="settings-row-desc">{t('languageDesc')}</span>
                  </div>
                  <div className="settings-select-wrap">
                    <select
                      className="settings-select"
                      value={language}
                      onChange={handleLanguageChange}
                    >
                      <option value="auto">Auto-detect</option>
                      <option value="en">English</option>
                      <option value="es">Español (Spanish)</option>
                      <option value="fr">Français (French)</option>
                      <option value="de">Deutsch (German)</option>
                      <option value="hi">हिंदी (Hindi)</option>
                    </select>
                    <ChevronDown size={14} className="settings-select-chevron" />
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">{t('streaming')}</span>
                    <span className="settings-row-desc">{t('streamingDesc')}</span>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={streaming}
                    onClick={handleToggleStreaming}
                    className={`settings-switch ${streaming ? 'checked' : ''}`}
                  >
                    <span className="settings-switch-knob" />
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">{t('exportConversations')}</span>
                    <span className="settings-row-desc">{t('exportConversationsDesc')}</span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={handleExportConversations}
                  >
                    <Download size={14} /> {t('exportJson')}
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">{t('sessionSignOut')}</span>
                    <span className="settings-row-desc">{t('sessionSignOutDesc')}</span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-danger"
                    onClick={() => {
                      if (window.confirm('Are you sure you want to log out of RagKno?')) {
                        onLogout?.()
                      }
                    }}
                  >
                    <LogOut size={14} /> {t('logOut')}
                  </button>
                </div>
              </div>
            </>
          )}

          {/* AI & RAG ENGINE TAB */}
          {activeTab === 'ai' && (
            <>
              <header className="settings-content-header">
                <h2>AI & Reasoning Engine</h2>
              </header>
              <div className="settings-content-scroll">
                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Primary LLM Model</span>
                    <span className="settings-row-desc">
                      Foundation model used for source synthesis and question answering
                    </span>
                  </div>
                  <div className="settings-select-wrap">
                    <select
                      className="settings-select"
                      value={model}
                      onChange={handleModelChange}
                    >
                      <option value="gemini-2.5-flash">Gemini 2.5 Flash (Ultra-fast)</option>
                      <option value="gemini-1.5-pro">Gemini 1.5 Pro (Deep reasoning)</option>
                      <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
                    </select>
                    <ChevronDown size={14} className="settings-select-chevron" />
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Real-time Streaming</span>
                    <span className="settings-row-desc">Stream answer tokens smoothly via Server-Sent Events</span>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={streaming}
                    onClick={handleToggleStreaming}
                    className={`settings-switch ${streaming ? 'checked' : ''}`}
                  >
                    <span className="settings-switch-knob" />
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Semantic CrossEncoder Re-ranking</span>
                    <span className="settings-row-desc">
                      Re-ranks hybrid search chunks with ms-marco-MiniLM-L-6-v2 cross-attention scoring
                    </span>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={useReranker}
                    onClick={handleToggleReranker}
                    className={`settings-switch ${useReranker ? 'checked' : ''}`}
                  >
                    <span className="settings-switch-knob" />
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Retrieved Documents (Top-K)</span>
                    <span className="settings-row-desc">
                      Number of candidate source passages provided into LLM context window
                    </span>
                  </div>
                  <div className="settings-select-wrap">
                    <select
                      className="settings-select"
                      value={topK}
                      onChange={handleTopKChange}
                    >
                      <option value="3">3 Chunks (Concise)</option>
                      <option value="5">5 Chunks (Balanced)</option>
                      <option value="7">7 Chunks (Thorough)</option>
                      <option value="10">10 Chunks (Deep context)</option>
                    </select>
                    <ChevronDown size={14} className="settings-select-chevron" />
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Reset AI Preferences</span>
                    <span className="settings-row-desc">Restore Gemini 2.5 Flash, top_k: 5, reranker: ON</span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={() => {
                      setModel('gemini-2.5-flash')
                      setTopK(5)
                      setUseReranker(true)
                      setStreaming(true)
                      localStorage.setItem('ragkno_model', 'gemini-2.5-flash')
                      localStorage.setItem('ragkno_top_k', '5')
                      localStorage.setItem('ragkno_reranker', 'true')
                      localStorage.setItem('ragkno_streaming', 'true')
                      onToast?.({ type: 'info', message: 'AI settings restored to recommended defaults' })
                    }}
                  >
                    <RefreshCw size={14} /> Reset Defaults
                  </button>
                </div>
              </div>
            </>
          )}

          {/* DATA CONTROLS TAB */}
          {activeTab === 'data' && (
            <>
              <header className="settings-content-header">
                <h2>Data Controls</h2>
              </header>
              <div className="settings-content-scroll">
                <div className="settings-banner-card">
                  <div className="settings-banner-icon">
                    <Database size={20} />
                  </div>
                  <div className="settings-banner-text">
                    <h4>Multi-Tenant Security Architecture</h4>
                    <p>
                      Each user's vectors in ChromaDB and Google Drive tokens in Supabase PostgreSQL are strictly isolated.
                      No other user or query can access or discover your documents.
                    </p>
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Google Drive Synchronization</span>
                    <span className="settings-row-desc">
                      Rescan Google Drive for newly added or updated files
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      type="button"
                      className="settings-btn-secondary"
                      onClick={handleSyncDriveNow}
                      disabled={isSyncingDrive}
                    >
                      <RefreshCw size={14} className={isSyncingDrive ? 'spin' : ''} />
                      {isSyncingDrive ? 'Syncing...' : 'Sync Now'}
                    </button>
                    <button
                      type="button"
                      className="settings-btn-secondary"
                      onClick={() => {
                        onClose?.()
                        onNavigateData?.()
                      }}
                    >
                      Data Center <ArrowRight size={14} />
                    </button>
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Clear Local Thread Cache</span>
                    <span className="settings-row-desc">
                      Reset locally stored conversation threads and temporary history in this browser
                    </span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-danger"
                    onClick={() => {
                      if (window.confirm('Are you sure you want to clear your local thread cache? This cannot be undone.')) {
                        onClearHistory?.()
                        onToast?.({ type: 'info', message: 'Local chat cache cleared.' })
                      }
                    }}
                  >
                    <Trash2 size={14} /> Clear Cache
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Export All Conversations</span>
                    <span className="settings-row-desc">Backup all conversations and sources as a formatted JSON document</span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={handleExportConversations}
                  >
                    <Download size={14} /> Export Backup
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Backend & Database Ping</span>
                    <span className="settings-row-desc">
                      {healthStatus
                        ? `Latency: ${healthStatus.latency}ms • Status: ${healthStatus.status} (${healthStatus.timestamp})`
                        : 'Verify database connectivity and live API health'}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={handleCheckHealth}
                    disabled={testingHealth}
                  >
                    <Activity size={14} className={testingHealth ? 'spin' : ''} />
                    {testingHealth ? 'Testing...' : 'Ping Server'}
                  </button>
                </div>
              </div>
            </>
          )}

          {/* HELP & FAQ TAB */}
          {activeTab === 'help' && (
            <>
              <header className="settings-content-header">
                <h2>Help & Documentation</h2>
              </header>
              <div className="settings-content-scroll">
                <div className="settings-banner-card">
                  <div className="settings-banner-icon">
                    <Sparkles size={20} />
                  </div>
                  <div className="settings-banner-text">
                    <h4>Getting Started with RagKno</h4>
                    <p>
                      Synthesize insights from your documents, resumes, spreadsheets, and Google Drive files with
                      instant verified citations.
                    </p>
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">1. Connect Your Sources</span>
                    <span className="settings-row-desc">
                      Open Data Center to connect Google Drive or upload PDFs, DOCX, and TXT files.
                    </span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={() => {
                      onClose?.()
                      onNavigateData?.()
                    }}
                  >
                    Open Data Center <ArrowRight size={14} />
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">2. Inquire Naturally</span>
                    <span className="settings-row-desc">
                      Ask complex questions. RagKno conducts hybrid vector + BM25 keyword search, ranks candidate passages, and references exact passages.
                    </span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={() => {
                      onClose?.()
                      onSampleQuestion?.('What are the main insights in my connected documents?')
                    }}
                  >
                    Try Sample Query
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">3. Trace Source Citations</span>
                    <span className="settings-row-desc">
                      Click any <code>[1]</code> citation pill in an AI response to preview the exact chunk and file it came from.
                    </span>
                  </div>
                  <span className="settings-badge-success">Interactive Pills</span>
                </div>

                {onOpenFeedback && (
                  <div className="settings-row-item">
                    <div className="settings-row-info">
                      <span className="settings-row-title">Share Feedback</span>
                      <span className="settings-row-desc">
                        Help us improve RagKno with your feedback, suggestions, or bug reports.
                      </span>
                    </div>
                    <button
                      type="button"
                      className="settings-btn-secondary"
                      onClick={() => {
                        onClose?.()
                        onOpenFeedback?.()
                      }}
                    >
                      <MessageSquareHeart size={14} /> Give Feedback
                    </button>
                  </div>
                )}

                <div className="settings-row-item" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.75rem' }}>
                  <span className="settings-row-title">Keyboard Shortcuts</span>
                  <div className="settings-shortcut-grid" style={{ width: '100%' }}>
                    <span className="settings-shortcut-label">Send Message</span>
                    <kbd className="settings-kbd">Enter</kbd>

                    <span className="settings-shortcut-label">New line in prompt</span>
                    <kbd className="settings-kbd">Shift + Enter</kbd>

                    <span className="settings-shortcut-label">Open Settings</span>
                    <kbd className="settings-kbd">Shift + ⌘ + ,</kbd>

                    <span className="settings-shortcut-label">Close Dialogs</span>
                    <kbd className="settings-kbd">Escape</kbd>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ABOUT RAGKNO TAB */}
          {activeTab === 'about' && (
            <>
              <header className="settings-content-header">
                <h2>About RagKno</h2>
              </header>
              <div className="settings-content-scroll">
                <div className="settings-banner-card">
                  <div className="settings-banner-icon">
                    <Layers size={20} />
                  </div>
                  <div className="settings-banner-text">
                    <h4>RagKno Core v1.0.0</h4>
                    <p>
                      Created by <strong>Priyanshu Urmaliya</strong>. Built for high-precision document synthesis, strict multi-tenant privacy, and lightning-fast hybrid retrieval.
                    </p>
                  </div>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">System Health & Latency Test</span>
                    <span className="settings-row-desc">
                      {healthStatus
                        ? `API: ${healthStatus.status} • Latency: ${healthStatus.latency}ms • Tested: ${healthStatus.timestamp}`
                        : 'Measure round-trip API latency and service readiness'}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={handleCheckHealth}
                    disabled={testingHealth}
                  >
                    <Activity size={14} className={testingHealth ? 'spin' : ''} />
                    {testingHealth ? 'Measuring...' : 'Run Diagnostics'}
                  </button>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Hybrid Retrieval (RRF)</span>
                    <span className="settings-row-desc">
                      Merges dense vector search (all-MiniLM-L6-v2) with sparse BM25 keyword matching via Reciprocal Rank Fusion.
                    </span>
                  </div>
                  <span className="settings-badge">Dense + Sparse</span>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">CrossEncoder Precision Reranker</span>
                    <span className="settings-row-desc">
                      Applies ms-marco-MiniLM-L-6-v2 cross-attention scoring to eliminate false positives.
                    </span>
                  </div>
                  <span className="settings-badge">ms-marco-MiniLM</span>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Database & Vector Storage</span>
                    <span className="settings-row-desc">
                      Supabase Cloud PostgreSQL for multi-tenant threads + ChromaDB for local persistent embeddings.
                    </span>
                  </div>
                  <span className="settings-badge">Postgres + Chroma</span>
                </div>

                <div className="settings-row-item">
                  <div className="settings-row-info">
                    <span className="settings-row-title">Creator & Portfolio</span>
                    <span className="settings-row-desc">
                      Priyanshu Urmaliya's developer profile and open-source work
                    </span>
                  </div>
                  <button
                    type="button"
                    className="settings-btn-secondary"
                    onClick={() => window.open('https://github.com/priyanshuurmaliya', '_blank', 'noopener,noreferrer')}
                  >
                    GitHub Profile <ExternalLink size={14} />
                  </button>
                </div>
              </div>
            </>
          )}

        </main>
      </div>
    </div>
  )
}
