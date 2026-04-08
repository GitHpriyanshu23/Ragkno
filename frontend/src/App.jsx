import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import {
  ArrowUp,
  Cloud,
  Database,
  FileText,
  FolderSync,
  Globe,
  Link2,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Send,
  Shield,
  Upload,
  UserCircle2,
} from 'lucide-react'
import {
  disconnectDrive,
  getAuthStatus,
  getAuthUrl,
  getDriveFiles,
  getIndexedSources,
  ingestFiles,
  ingestUrl,
  queryRAG,
  queryRAGStream,
  resetChatMemory,
  syncDrive,
  unindexSource,
} from './api.js'

function AppShell({ children, toasts, onDismissToast }) {
  const location = useLocation()
  const isChatRoute = location.pathname === '/chat'

  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="top-nav-inner">
          <Link to="/" className="brand">RAGKNO</Link>
          <nav className="top-links">
            <NavLink to="/" className={({ isActive }) => `top-link ${isActive ? 'active' : ''}`}>Home</NavLink>
            <NavLink to="/data" className={({ isActive }) => `top-link ${isActive ? 'active' : ''}`}>Data</NavLink>
            <NavLink to="/chat" className={({ isActive }) => `top-link ${isActive ? 'active' : ''}`}>Chat</NavLink>
          </nav>
          <button className="icon-btn" aria-label="Profile">
            <UserCircle2 size={18} />
          </button>
        </div>
      </header>
      <main className={isChatRoute ? 'chat-main' : ''}>{children}</main>
      {!isChatRoute && (
        <footer className="site-footer">
          <div>© 2026 RAGKNO ARCHIVE. ALL RIGHTS RESERVED.</div>
          <div className="footer-links">
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
            <a href="#">API Documentation</a>
          </div>
        </footer>
      )}

      {toasts.length > 0 && (
        <div className="toast-stack" role="status" aria-live="polite">
          {toasts.map((toast) => (
            <div key={toast.id} className={`toast ${toast.type || 'info'}`}>
              <p>{toast.message}</p>
              <div className="toast-actions">
                {toast.actionLabel && toast.onAction && (
                  <button
                    className="toast-action"
                    onClick={() => {
                      toast.onAction()
                      onDismissToast(toast.id)
                    }}
                  >
                    {toast.actionLabel}
                  </button>
                )}
                <button className="toast-close" onClick={() => onDismissToast(toast.id)}>Dismiss</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function HomePage() {
  const words = useMemo(() => ['Drive Docs', 'PDFs', 'Links', 'Social Media'], [])
  const [wordIndex, setWordIndex] = useState(0)
  const [charIndex, setCharIndex] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)

  const currentWord = words[wordIndex]
  const renderedWords = currentWord.slice(0, charIndex)

  useEffect(() => {
    let timeoutMs = isDeleting ? 45 : 95

    if (!isDeleting && charIndex === currentWord.length) {
      timeoutMs = 1200
    }

    if (isDeleting && charIndex === 0) {
      timeoutMs = 260
    }

    const timer = window.setTimeout(() => {
      if (!isDeleting) {
        if (charIndex < currentWord.length) {
          setCharIndex((prev) => prev + 1)
          return
        }
        setIsDeleting(true)
        return
      }

      if (charIndex > 0) {
        setCharIndex((prev) => prev - 1)
        return
      }

      setIsDeleting(false)
      setWordIndex((prev) => (prev + 1) % words.length)
    }, timeoutMs)

    return () => window.clearTimeout(timer)
  }, [charIndex, currentWord, isDeleting, words.length])

  return (
    <>
      <section className="hero-section">
        <div className="hero-grid">
          <div className="hero-left">
            <p className="hero-kicker">Intelligent Retrieval System</p>
            <h1>
              Rag application<br />
              for your<br />
              <span className="typed-word">{renderedWords || '\u00A0'}</span>
            </h1>
            <p className="hero-copy">
              Connect, process, and query your knowledge base in one seamless, high-performance interface.
              Engineered for speed, precision, and absolute clarity.
            </p>
            <div className="hero-actions">
              <Link to="/data" className="btn-primary-solid">Get Started</Link>
              <a href="#" className="btn-link">View Documentation</a>
            </div>
          </div>
          <div className="hero-card" aria-hidden="true">
            <div className="card-line" />
            <div className="card-lines" />
            <div className="card-grid">
              <div />
              <div />
            </div>
            <div className="glass-note">System Response</div>
          </div>
        </div>
      </section>

      <section className="feature-section">
        <div className="feature-top">
          <div className="feature-head">
            <h2>
              Structural Purity.<br />
              Information at Scale.
            </h2>
            <p>Every interaction is designed to minimize friction and maximize insight retrieval.</p>
          </div>
          <div className="feature-label">Section 01 // Capabilities</div>
        </div>
        <div className="feature-grid">
          <article className="feature-card wide">
            <FolderSync size={28} />
            <h3>Universal Knowledge Mapping</h3>
            <p>Map relationships between documents, links, and drive assets with semantic precision.</p>
          </article>
          <article className="feature-card dark">
            <Shield size={28} />
            <h3>Hardened Privacy</h3>
            <p>Enterprise-grade controls keep each query and source isolated and secure.</p>
          </article>
          <article className="feature-card">
            <ArrowUp size={28} />
            <h3>Millisecond Latency</h3>
            <p>Retrieve top context chunks instantly from your indexed knowledge base.</p>
          </article>
          <article className="feature-image-card">
            <img
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuDDG2l_srBp6LKa1W9Xd7RuaCMjPa8_mBYk9g8o0Mb2OdiaNAtgWUVP_8GxsZhi0uQgW7J0TeLxCurrbC3SdR-5UAyT9Yh8ZwpvL7LjqONmc397dHkEe8C8TAZ_4UHtU7i9QqXGOzrsuWFWCl0OdR3sCU3xRw3SG12aLB9YjwgoNwF1zyCfaHsnRDwYwryD1vq6rR6-VihcBK17ioHhqkiUXxG4D1laZ_UlMQj3BYgljjkQWBt3nCHOYO8oWXk6BU8ag3E3FSQyk4s"
              alt="Abstract digital visualization of data nodes and network connections"
            />
            <div className="feature-image-overlay">
              <h3>Deep Contextual Intelligence</h3>
            </div>
          </article>
        </div>
      </section>

      <section className="connect-section">
        <div className="connect-text">
          <h2>Connect Everything.</h2>
          <ol>
            <li>Google Drive & Workspace</li>
            <li>Complex PDFs & Tables</li>
            <li>Live Web Crawling</li>
          </ol>
        </div>
        <div className="connect-tiles">
          <div className="tile large" />
          <div className="tile" />
          <div className="tile dark-tile">Optimized For Enterprise Deployment</div>
        </div>
      </section>

      <section className="cta-section">
        <h2>Ready to archive the future?</h2>
        <div className="cta-actions">
          <Link className="btn-primary-solid" to="/data">Get Started Now</Link>
          <Link className="btn-outline" to="/chat">Request Demo</Link>
        </div>
      </section>
    </>
  )
}

function DataPage({ onToast }) {
  const location = useLocation()
  const [connected, setConnected] = useState(false)
  const [files, setFiles] = useState([])
  const [selectedIds, setSelectedIds] = useState([])
  const [indexedSources, setIndexedSources] = useState([])
  const [indexedSourceKeys, setIndexedSourceKeys] = useState(new Set())
  const [loading, setLoading] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [disconnecting, setDisconnecting] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [addingUrl, setAddingUrl] = useState(false)
  const [removingKey, setRemovingKey] = useState('')
  const [notice, setNotice] = useState('')
  const [urlInput, setUrlInput] = useState('')
  const fileInputRef = useRef(null)

  function showError(message, retryAction) {
    const text = String(message || 'Request failed.')
    setNotice(text)
    onToast?.({
      type: 'error',
      message: text,
      actionLabel: retryAction ? 'Retry' : null,
      onAction: retryAction || null,
    })
  }

  useEffect(() => {
    void bootstrap()
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    if (params.get('connected') === '1') {
      setNotice('Google Drive connected successfully.')
    }
  }, [location.search])

  async function bootstrap() {
    try {
      const [{ connected }] = await Promise.all([
        getAuthStatus(),
      ])
      setConnected(connected)
      await refreshIndexedSources()
      if (connected) await refreshFiles()
    } catch {
      showError('Backend is unreachable. Start FastAPI on port 8000.', bootstrap)
    }
  }

  async function refreshIndexedSources() {
    try {
      const result = await getIndexedSources()
      const sources = result.sources || []
      setIndexedSources(sources)
      setIndexedSourceKeys(new Set(sources.map((item) => item.key)))
    } catch {
      // Keep page usable even if this endpoint fails.
    }
  }

  function normalizeSourceKey(value) {
    return String(value || '').trim().toLowerCase().replace(/\/+$/, '')
  }

  function driveSourceKey(name) {
    return normalizeSourceKey(`drive://${name || ''}`)
  }

  async function refreshFiles() {
    setLoading(true)
    setNotice('')
    try {
      const { files } = await getDriveFiles()
      setFiles(files)
      setSelectedIds([])
    } catch (error) {
      showError(error.message, refreshFiles)
    } finally {
      setLoading(false)
    }
  }

  async function connectDrive() {
    try {
      const { url } = await getAuthUrl()
      window.location.href = url
    } catch (error) {
      showError(error.message, connectDrive)
    }
  }

  async function syncSelected() {
    if (selectedIds.length === 0) {
      setNotice('Select at least one source before syncing.')
      return
    }

    const selectedFiles = files.filter((file) => selectedIds.includes(file.id))
    const unsyncedFiles = selectedFiles.filter((file) => !indexedSourceKeys.has(driveSourceKey(file.name)))
    const unsyncedIds = unsyncedFiles.map((file) => file.id)

    if (unsyncedIds.length === 0) {
      setNotice('All selected Drive files are already indexed.')
      return
    }

    setSyncing(true)
    setNotice('')
    try {
      const result = await syncDrive(unsyncedIds)
      await refreshIndexedSources()
      setNotice(result.message || 'Sync completed.')
      onToast?.({ type: 'success', message: result.message || 'Sync completed.' })
    } catch (error) {
      showError(error.message, syncSelected)
    } finally {
      setSyncing(false)
    }
  }

  async function disconnect() {
    setDisconnecting(true)
    setNotice('')
    try {
      await disconnectDrive()
      setConnected(false)
      setFiles([])
      setSelectedIds([])
      setNotice('Google Drive disconnected.')
    } catch (error) {
      showError(error.message, disconnect)
    } finally {
      setDisconnecting(false)
    }
  }

  function selectAll() {
    setSelectedIds(files.map((file) => file.id))
  }

  function clearSelection() {
    setSelectedIds([])
  }

  function toggleSelection(fileId) {
    setSelectedIds((prev) => (
      prev.includes(fileId) ? prev.filter((id) => id !== fileId) : [...prev, fileId]
    ))
  }

  function onUploadPick() {
    fileInputRef.current?.click()
  }

  async function ingestPickedFiles(picked) {
    const uniqueFiles = picked.filter((file) => !indexedSourceKeys.has(normalizeSourceKey(file.name)))
    if (uniqueFiles.length === 0) {
      setNotice('Selected files are already indexed.')
      return
    }

    setUploading(true)
    setNotice('')
    try {
      const result = await ingestFiles(uniqueFiles)
      await refreshIndexedSources()
      setNotice(result.message || `Indexed ${uniqueFiles.length} uploaded file(s).`)
      onToast?.({ type: 'success', message: result.message || 'Files indexed.' })
    } catch (error) {
      showError(error.message)
    } finally {
      setUploading(false)
    }
  }

  async function onUploadChange(event) {
    const picked = Array.from(event.target.files || [])
    if (picked.length === 0) {
      return
    }

    await ingestPickedFiles(picked)
    event.target.value = ''
  }

  async function addUrl() {
    if (!urlInput.trim()) {
      setNotice('Enter a URL first.')
      return
    }

    const normalizedUrl = normalizeSourceKey(urlInput)
    if (indexedSourceKeys.has(normalizedUrl)) {
      setNotice('This URL is already indexed.')
      return
    }

    setAddingUrl(true)
    setNotice('')
    try {
      const result = await ingestUrl(urlInput.trim())
      await refreshIndexedSources()
      setNotice(result.message || 'URL indexed successfully.')
      onToast?.({ type: 'success', message: result.message || 'URL indexed successfully.' })
      setUrlInput('')
    } catch (error) {
      showError(error.message, addUrl)
    } finally {
      setAddingUrl(false)
    }
  }

  async function removeIndexedSource(item) {
    if (!item?.key) {
      return
    }

    const confirmed = window.confirm(`Unindex this source?\n\n${item.source}`)
    if (!confirmed) {
      return
    }

    setRemovingKey(item.key)
    setNotice('')
    try {
      const result = await unindexSource(item.key)
      await refreshIndexedSources()
      if (connected) {
        await refreshFiles()
      }
      setNotice(result.message || 'Source unindexed successfully.')
      onToast?.({ type: 'info', message: result.message || 'Source unindexed successfully.' })
    } catch (error) {
      showError(error.message, () => removeIndexedSource(item))
    } finally {
      setRemovingKey('')
    }
  }

  return (
    <section className="data-page">
      <header className="data-header">
        <span>Ingestion Engine</span>
        <h1>Data Sources.</h1>
      </header>

      <div className="data-grid">
        <article className="panel connect-drive">
          <Cloud size={34} />
          <h3>Connect Google Drive</h3>
          <p>Sync folders or selected files directly into your retrieval archive.</p>
          {!connected && <button className="btn-primary-solid" onClick={connectDrive}>Connect</button>}
          {connected && (
            <>
              <button className="btn-outline" onClick={refreshFiles} disabled={loading}>Refresh Files</button>
              <button className="btn-outline" onClick={disconnect} disabled={disconnecting}>
                {disconnecting ? 'Disconnecting...' : 'Disconnect'}
              </button>
            </>
          )}
        </article>

        <article className="panel upload-panel" onClick={onUploadPick}>
          <Upload size={40} />
          <h3>Upload PDFs</h3>
          <p>Drag and drop documents or click to browse.</p>
          <small>Supported: PDF, DOCX, TXT (Max 50MB)</small>
          {uploading && <small><RefreshCw size={12} className="spin" /> Uploading and indexing...</small>}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            multiple
            onChange={onUploadChange}
            hidden
          />
        </article>

        <article className="panel url-panel">
          <h3>Add Website URL</h3>
          <div className="url-row">
            <input
              type="url"
              placeholder="https://example.com/documentation"
              value={urlInput}
              onChange={(event) => setUrlInput(event.target.value)}
            />
            <button className="btn-primary-solid" onClick={addUrl} disabled={addingUrl}>
              {addingUrl ? <><RefreshCw size={14} className="spin" /> Adding...</> : 'Add URL'}
            </button>
          </div>
          <div className="chip-row">
            <span><Link2 size={12} /> JavaScript Rendering</span>
            <span><Link2 size={12} /> Recursive Crawling</span>
          </div>
        </article>
      </div>

      <section className="indexed-section">
        <div className="indexed-head">
          <div>
            <h2>Already Indexed</h2>
            <p>Manage sources currently stored in your vector database.</p>
          </div>
        </div>

        {indexedSources.length === 0 && (
          <p className="notice">No sources indexed yet.</p>
        )}

        {indexedSources.length > 0 && (
          <div className="indexed-sources-box">
            <p className="indexed-meta">{indexedSources.length} unique source(s) already indexed.</p>
            <ul className="indexed-sources-list">
              {indexedSources.map((item) => (
                <li key={item.key}>
                  <div className="indexed-item-main">
                    <span className="indexed-type">{item.type}</span>
                    <span className="indexed-name">{item.source}</span>
                  </div>
                  <button
                    className="btn-outline indexed-remove"
                    onClick={() => removeIndexedSource(item)}
                    disabled={removingKey === item.key}
                  >
                    {removingKey === item.key ? 'Unindexing...' : 'Unindex'}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section className="sources-section">
        <div className="sources-head">
          <div>
            <h2>Active Sources</h2>
            <p>Real-time status of your knowledge base inventory.</p>
          </div>
          <div className="sources-actions">
            <button className="btn-outline" onClick={refreshFiles} disabled={loading || !connected}>
              <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh All
            </button>
            <button className="btn-outline" onClick={selectAll} disabled={!connected || files.length === 0}>Select All</button>
            <button className="btn-outline" onClick={clearSelection} disabled={!connected || selectedIds.length === 0}>Clear</button>
            <button className="btn-primary-solid" onClick={syncSelected} disabled={!connected || syncing || selectedIds.length === 0}>
              {syncing ? <><RefreshCw size={14} className="spin" /> Syncing...</> : `Sync Selected (${selectedIds.length})`}
            </button>
          </div>
        </div>

        {!connected && <p className="notice">Connect Google Drive to load sources.</p>}
        {notice && <p className="notice">{notice}</p>}
        {connected && loading && <p className="notice">Loading source list...</p>}
        {connected && !loading && files.length === 0 && <p className="notice">No supported files found in Drive.</p>}

        {connected && !loading && (
          <ul className="source-list">
            {files.map((file) => {
              const selected = selectedIds.includes(file.id)
              const indexed = indexedSourceKeys.has(driveSourceKey(file.name))

              return (
                <li key={file.id} className="source-row" onClick={() => toggleSelection(file.id)}>
                  <div className="source-main">
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleSelection(file.id)}
                      onClick={(event) => event.stopPropagation()}
                    />
                    <div className="source-icon">
                      {file.mimeType === 'application/pdf' ? <FileText size={18} /> : <Database size={18} />}
                    </div>
                    <div>
                      <strong>{file.name}</strong>
                      <small>{file.mimeType}</small>
                    </div>
                  </div>
                  <div className={`status ${indexed ? 'ok' : 'pending'}`}>
                    {indexed ? 'Indexed' : 'Pending'}
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </section>
    </section>
  )
}

function ChatPage({ onToast }) {
  const CHAT_THREADS_KEY = 'ragkno_chat_threads_v1'
  const ACTIVE_THREAD_KEY = 'ragkno_active_thread_v1'

  function makeMessageId(prefix = 'msg') {
    const now = Date.now()
    return window.crypto?.randomUUID?.() || `${prefix}-${now}-${Math.random().toString(36).slice(2, 8)}`
  }

  function makeThreadTitleFromMessage(text) {
    const trimmed = String(text || '').trim()
    if (!trimmed) {
      return 'New Chat'
    }
    return trimmed.length > 48 ? `${trimmed.slice(0, 48)}...` : trimmed
  }

  function createThread() {
    const now = Date.now()
    return {
      id: window.crypto?.randomUUID?.() || `thread-${now}`,
      sessionId: window.crypto?.randomUUID?.() || `session-${now}`,
      title: 'New Chat',
      createdAt: now,
      updatedAt: now,
      messages: [],
    }
  }

  function normalizeMessage(message, index = 0) {
    const role = message?.role === 'assistant' ? 'assistant' : 'user'
    return {
      id: message?.id || makeMessageId(`legacy-${index}`),
      role,
      text: String(message?.text || ''),
      ts: String(message?.ts || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
      sources: Array.isArray(message?.sources) ? message.sources : [],
      streaming: Boolean(message?.streaming),
    }
  }

  function loadThreadsState() {
    try {
      const rawThreads = window.localStorage.getItem(CHAT_THREADS_KEY)
      const rawActive = window.localStorage.getItem(ACTIVE_THREAD_KEY)

      if (!rawThreads) {
        const first = createThread()
        return { threads: [first], activeId: first.id }
      }

      const parsed = JSON.parse(rawThreads)
      if (!Array.isArray(parsed) || parsed.length === 0) {
        const first = createThread()
        return { threads: [first], activeId: first.id }
      }

      const validThreads = parsed
        .filter((thread) => thread?.id && thread?.sessionId && Array.isArray(thread?.messages))
        .map((thread) => ({
          ...thread,
          messages: thread.messages.map((message, index) => normalizeMessage(message, index)),
        }))
      if (validThreads.length === 0) {
        const first = createThread()
        return { threads: [first], activeId: first.id }
      }

      const activeThreadExists = rawActive && validThreads.some((thread) => thread.id === rawActive)
      return {
        threads: validThreads,
        activeId: activeThreadExists ? rawActive : validThreads[0].id,
      }
    } catch {
      const first = createThread()
      return { threads: [first], activeId: first.id }
    }
  }

  function renderBoldText(text, keyPrefix) {
    const parts = String(text || '').split(/(\*\*[^*]+\*\*)/g)
    return parts.map((part, partIndex) => (
      part.startsWith('**') && part.endsWith('**')
        ? <strong key={`${keyPrefix}-s-${partIndex}`}>{part.slice(2, -2)}</strong>
        : <span key={`${keyPrefix}-t-${partIndex}`}>{part}</span>
    ))
  }

  function renderInlineText(text, keyPrefix, message) {
    const value = String(text || '')
    const citationRegex = /\[(\d+)\]/g
    const nodes = []
    let lastIndex = 0
    let match
    let counter = 0

    while ((match = citationRegex.exec(value)) !== null) {
      const before = value.slice(lastIndex, match.index)
      if (before) {
        nodes.push(...renderBoldText(before, `${keyPrefix}-b-${counter}`))
      }

      const citationIndex = Number(match[1])
      const hasSource = Array.isArray(message?.sources)
        && message.sources.some((source) => Number(source.index) === citationIndex)

      if (hasSource) {
        nodes.push(
          <button
            key={`${keyPrefix}-c-${counter}`}
            className="citation-ref"
            onMouseEnter={() => setHoveredCitation({ messageId: message.id, sourceIndex: citationIndex })}
            onMouseLeave={() => setHoveredCitation(null)}
            onFocus={() => setHoveredCitation({ messageId: message.id, sourceIndex: citationIndex })}
            onBlur={() => setHoveredCitation(null)}
            type="button"
          >
            [{citationIndex}]
          </button>,
        )
      } else {
        nodes.push(<span key={`${keyPrefix}-c-${counter}`}>{match[0]}</span>)
      }

      lastIndex = match.index + match[0].length
      counter += 1
    }

    const tail = value.slice(lastIndex)
    if (tail) {
      nodes.push(...renderBoldText(tail, `${keyPrefix}-tail`))
    }

    return nodes
  }

  function renderAssistantText(text, message) {
    const blocks = String(text || '').split(/\n\n+/)
    return blocks.map((block, blockIndex) => {
      const lines = block.split('\n').filter((line) => line.trim())
      const isBulletBlock = lines.length > 0 && lines.every((line) => /^[-*]\s+/.test(line.trim()))

      if (isBulletBlock) {
        return (
          <ul key={`ul-${blockIndex}`}>
            {lines.map((line, lineIndex) => (
              <li key={`li-${blockIndex}-${lineIndex}`}>
                {renderInlineText(line.replace(/^[-*]\s+/, ''), `b-${blockIndex}-${lineIndex}`, message)}
              </li>
            ))}
          </ul>
        )
      }

      return (
        <p key={`p-${blockIndex}`}>
          {lines.map((line, lineIndex) => (
            <span key={`ln-${blockIndex}-${lineIndex}`}>
              {lineIndex > 0 && <br />}
              {renderInlineText(line, `p-${blockIndex}-${lineIndex}`, message)}
            </span>
          ))}
        </p>
      )
    })
  }

  const initialState = useMemo(() => loadThreadsState(), [])
  const [threads, setThreads] = useState(initialState.threads)
  const [activeThreadId, setActiveThreadId] = useState(initialState.activeId)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [hoveredCitation, setHoveredCitation] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [expandedSourceMessages, setExpandedSourceMessages] = useState(new Set())
  const [hoveredSourceMessage, setHoveredSourceMessage] = useState(null)
  const bottomRef = useRef(null)

  const sortedThreads = useMemo(
    () => [...threads].sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0)),
    [threads],
  )

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) || sortedThreads[0],
    [threads, activeThreadId, sortedThreads],
  )

  const messages = activeThread?.messages || []

  useEffect(() => {
    window.localStorage.setItem(CHAT_THREADS_KEY, JSON.stringify(threads))
    if (activeThread?.id) {
      window.localStorage.setItem(ACTIVE_THREAD_KEY, activeThread.id)
    }
  }, [threads, activeThread])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    if (!activeThread && threads.length > 0) {
      setActiveThreadId(threads[0].id)
    }
  }, [activeThread, threads])

  function updateActiveThread(updater) {
    if (!activeThread) {
      return
    }

    setThreads((prev) => prev.map((thread) => {
      if (thread.id !== activeThread.id) {
        return thread
      }
      return updater(thread)
    }))
  }

  function updateMessageById(messageId, updater) {
    updateActiveThread((thread) => ({
      ...thread,
      updatedAt: Date.now(),
      messages: thread.messages.map((message) => {
        if (message.id !== messageId) {
          return message
        }
        return updater(message)
      }),
    }))
  }

  function toggleMessageSources(messageId) {
    setExpandedSourceMessages((prev) => {
      const next = new Set(prev)
      if (next.has(messageId)) {
        next.delete(messageId)
      } else {
        next.add(messageId)
      }
      return next
    })
  }

  async function sendMessage(event) {
    event.preventDefault()
    const text = input.trim()
    if (!text || loading) {
      return
    }

    if (!activeThread) {
      return
    }

    const userMessage = {
      id: makeMessageId('user'),
      role: 'user',
      text,
      ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      sources: [],
      streaming: false,
    }
    const assistantMessageId = makeMessageId('assistant')
    const assistantMessage = {
      id: assistantMessageId,
      role: 'assistant',
      text: '',
      ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      sources: [],
      streaming: true,
    }

    updateActiveThread((thread) => ({
      ...thread,
      title: thread.messages.some((msg) => msg.role === 'user') ? thread.title : makeThreadTitleFromMessage(text),
      updatedAt: Date.now(),
      messages: [...thread.messages, userMessage, assistantMessage],
    }))
    setInput('')
    setLoading(true)
    setError('')

    try {
      const streamDonePayload = await queryRAGStream(text, 3, activeThread.sessionId, {
        onMeta: (meta) => {
          const metaSources = Array.isArray(meta?.sources) ? meta.sources : []
          updateMessageById(assistantMessageId, (message) => ({
            ...message,
            sources: metaSources,
          }))

          if (meta?.session_id) {
            updateActiveThread((thread) => ({
              ...thread,
              updatedAt: Date.now(),
              sessionId: meta.session_id,
            }))
          }
        },
        onToken: (token) => {
          updateMessageById(assistantMessageId, (message) => ({
            ...message,
            text: `${message.text}${token}`,
          }))
        },
      })

      updateMessageById(assistantMessageId, (message) => ({
        ...message,
        text: streamDonePayload?.answer || message.text || 'No relevant documents found.',
        sources: Array.isArray(streamDonePayload?.sources) && streamDonePayload.sources.length > 0
          ? streamDonePayload.sources
          : message.sources,
        streaming: false,
      }))
    } catch (streamError) {
      try {
        const result = await queryRAG(text, 3, activeThread.sessionId)
        updateMessageById(assistantMessageId, (message) => ({
          ...message,
          text: result.answer || message.text || 'No relevant documents found.',
          sources: Array.isArray(result.sources) ? result.sources : message.sources,
          streaming: false,
        }))
      } catch (fallbackError) {
        updateMessageById(assistantMessageId, (message) => ({
          ...message,
          streaming: false,
          text: message.text || 'The response failed. Please retry.',
        }))
        setError(fallbackError.message)
        onToast?.({
          type: 'error',
          message: fallbackError.message,
          actionLabel: 'Retry',
          onAction: () => {},
        })
      }

      if (streamError?.message) {
        onToast?.({ type: 'info', message: `Stream interrupted, used fallback response. ${streamError.message}` })
      }
    } finally {
      setLoading(false)
    }
  }

  async function startNewChat() {
    const newThread = createThread()
    setThreads((prev) => [newThread, ...prev])
    setActiveThreadId(newThread.id)
    setError('')
    setInput('')
  }

  async function resetCurrentChatMemory() {
    if (!activeThread?.sessionId) {
      return
    }

    try {
      await resetChatMemory(activeThread.sessionId)
    } catch {
      // Keep local reset behavior even if backend reset fails.
    }

    updateActiveThread((thread) => ({
      ...thread,
      title: 'New Chat',
      updatedAt: Date.now(),
      messages: [],
    }))
    setError('')
    setInput('')
  }

  async function deleteThread(threadId) {
    const target = threads.find((thread) => thread.id === threadId)
    if (!target) {
      return
    }

    try {
      await resetChatMemory(target.sessionId)
    } catch {
      // Keep local deletion even if backend reset fails.
    }

    setThreads((prev) => {
      const remaining = prev.filter((thread) => thread.id !== threadId)
      if (remaining.length > 0) {
        return remaining
      }
      return [createThread()]
    })

    if (activeThreadId === threadId) {
      const fallback = threads.find((thread) => thread.id !== threadId)
      if (fallback) {
        setActiveThreadId(fallback.id)
      }
    }
  }

  return (
    <section className={`chat-page ${sidebarCollapsed ? 'collapsed' : ''}`}>
      {sidebarCollapsed && (
        <aside className="chat-collapsed-rail">
          <button
            className="sidebar-toggle-btn"
            type="button"
            onClick={() => setSidebarCollapsed(false)}
            aria-label="Reveal sidebar"
            title="Reveal sidebar"
          >
            <PanelLeftOpen size={16} />
          </button>
        </aside>
      )}

      {!sidebarCollapsed && (
        <aside className="chat-sidebar">
          <div className="chat-sidebar-header">
            <button
              className="sidebar-toggle-btn"
              type="button"
              onClick={() => setSidebarCollapsed(true)}
              aria-label="Collapse sidebar"
              title="Collapse sidebar"
            >
              <PanelLeftClose size={16} />
            </button>
          </div>

          <h4>Previous Conversations</h4>
          <button className="history-btn" onClick={startNewChat}>New Chat</button>
          <button className="history-btn" onClick={resetCurrentChatMemory}>Clear Active Chat</button>

          {sortedThreads.map((thread) => (
            <div key={thread.id} className={`history-item ${thread.id === activeThread?.id ? 'active' : ''}`}>
              <button
                className={`history-btn ${thread.id === activeThread?.id ? 'active' : ''}`}
                onClick={() => setActiveThreadId(thread.id)}
              >
                {thread.title || 'New Chat'}
              </button>
              <button
                className="history-delete-btn"
                aria-label={`Delete ${thread.title || 'chat'}`}
                onClick={(event) => {
                  event.stopPropagation()
                  void deleteThread(thread.id)
                }}
              >
                ×
              </button>
            </div>
          ))}

          <h4>Source Settings</h4>
          <div className="toggle-row">
            <span><Database size={14} /> Internal Docs</span>
            <span className="toggle on" />
          </div>
          <div className="toggle-row">
            <span><Globe size={14} /> Web Index</span>
            <span className="toggle" />
          </div>
        </aside>
      )}

      <section className="chat-canvas">
        <div className="message-stack">
          {!messages.length && !loading && (
            <div className="message assistant">
              <div className="bubble">Start a new conversation. Your chat title will be created from your first message.</div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <div className="bubble">
                {message.role === 'assistant' ? renderAssistantText(message.text, message) : message.text}
                {message.role === 'assistant' && message.streaming && <span className="typing-cursor" aria-hidden="true" />}
              </div>

              {message.role === 'assistant' && Array.isArray(message.sources) && message.sources.length > 0 && (
                <div
                  className="message-sources"
                  onMouseEnter={() => setHoveredSourceMessage(message.id)}
                  onMouseLeave={() => setHoveredSourceMessage(null)}
                >
                  <button
                    type="button"
                    className={`source-trigger ${expandedSourceMessages.has(message.id) ? 'open' : ''}`}
                    onClick={() => toggleMessageSources(message.id)}
                  >
                    source
                  </button>

                  {hoveredSourceMessage === message.id && !expandedSourceMessages.has(message.id) && (
                    <div className="source-hover-preview">
                      {message.sources.slice(0, 3).map((source, sourceIndex) => {
                        const index = Number(source.index || sourceIndex + 1)
                        const fullText = String(source.text || '')
                        const previewText = String(source.preview || fullText.slice(0, 120))
                        return (
                          <p key={`${message.id}-preview-${index}`}>
                            <strong>[{index}]</strong> {previewText}
                          </p>
                        )
                      })}
                    </div>
                  )}

                  {expandedSourceMessages.has(message.id) && (
                    <div className="source-cards">
                      {message.sources.map((source, sourceIndex) => {
                        const index = Number(source.index || sourceIndex + 1)
                        const highlighted = hoveredCitation?.messageId === message.id
                          && Number(hoveredCitation.sourceIndex) === index
                        const fullText = String(source.text || '')

                        return (
                          <article key={`${message.id}-${index}`} className={`source-card ${highlighted ? 'highlighted' : ''}`}>
                            <header>
                              <span className="source-index">[{index}]</span>
                              {source.link ? (
                                <a href={source.link} target="_blank" rel="noreferrer">{source.title || source.source}</a>
                              ) : (
                                <strong>{source.title || source.source}</strong>
                              )}
                            </header>
                            <p>{fullText}</p>
                            <div className="source-meta-row">
                              <small>Relevance: {Number(source.score || 0).toFixed(3)}</small>
                            </div>
                          </article>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}

              <span>{message.role === 'user' ? 'User' : 'RAGKNO AI'} • {message.ts}</span>
            </div>
          ))}

          {error && <p className="notice">{error}</p>}
          <div ref={bottomRef} />
        </div>

        <form className="chat-input-row" onSubmit={sendMessage}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask anything..."
          />
          <button type="submit" className="send-btn" disabled={!input.trim() || loading}>
            <Send size={16} />
          </button>
        </form>
        <p className="chat-disclaimer">RAGKNO AI can make mistakes. Verify critical information.</p>
      </section>
    </section>
  )
}

function ScrollToTop() {
  const { pathname } = useLocation()

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  return null
}

function HandleOAuthRedirect() {
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    if (params.get('drive') === 'connected') {
      navigate('/data?connected=1', { replace: true })
    }
  }, [location.search, navigate])

  return null
}

export default function App() {
  const [toasts, setToasts] = useState([])

  const pushToast = useCallback((toast) => {
    if (!toast?.message) {
      return
    }
    const id = window.crypto?.randomUUID?.() || `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const payload = {
      id,
      type: toast.type || 'info',
      message: toast.message,
      actionLabel: toast.actionLabel || null,
      onAction: toast.onAction || null,
      ttl: Number(toast.ttl ?? 4200),
    }
    setToasts((prev) => [...prev, payload])
  }, [])

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
  }, [])

  useEffect(() => {
    if (toasts.length === 0) {
      return undefined
    }

    const timers = toasts.map((toast) => window.setTimeout(() => dismissToast(toast.id), toast.ttl))
    return () => timers.forEach((timer) => window.clearTimeout(timer))
  }, [toasts, dismissToast])

  return (
    <BrowserRouter>
      <ScrollToTop />
      <HandleOAuthRedirect />
      <AppShell toasts={toasts} onDismissToast={dismissToast}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/data" element={<DataPage onToast={pushToast} />} />
          <Route path="/chat" element={<ChatPage onToast={pushToast} />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}
