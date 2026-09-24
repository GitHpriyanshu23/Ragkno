import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BrowserRouter, Link, Navigate, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import Lenis from 'lenis'
import { HugeiconsIcon } from '@hugeicons/react'
import { LayoutAlignLeftIcon } from '@hugeicons/core-free-icons'
import {
  ArrowUp,
  ChevronDown,
  ChevronRight,
  ChevronsUpDown,
  Cloud,
  CircleHelp,
  Database,
  Download,
  FileText,
  FolderSync,
  Globe,
  Info,
  Link2,
  LogOut,
  Menu,
  MessageSquare,
  MessageSquareHeart,
  PanelLeft,
  PanelLeftClose,
  Plus,
  RefreshCw,
  Search,
  Send,
  Shield,
  Settings,
  SquarePen,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import {
  disconnectDrive,
  getAuthStatus,
  getAuthUrl,
  getCurrentUser,
  getDriveFiles,
  getGoogleLoginUrl,
  getIndexedSources,
  ingestFiles,
  ingestUrl,
  logoutUser,
  queryRAG,
  queryRAGStream,
  resetChatMemory,
  syncDrive,
  unindexSource,
  submitFeedback,
  API_BASE,
} from './api.js'
import FeedbackModal from './components/FeedbackModal.jsx'
import ChatGPTSettingsModal from './components/ChatGPTSettingsModal.jsx'
import DotGrid from './components/DotGrid.jsx'
import brandLogo from './assets/figma-logo-mark.svg'
import { useI18n } from './lib/i18n.jsx'
import googleLogo from './assets/google-logo-2025.webp'
import heroImage from './assets/ragkno-hero.png'
import serverRacksImage from './assets/server-racks.png'
import serverCablesImage from './assets/server-cables.png'
import ctaTexture from './assets/cta-texture.png'

function AppShell({ children, toasts, onDismissToast }) {
  const location = useLocation()
  const isChatRoute = location.pathname === '/chat'
  const isAppRoute = location.pathname.startsWith('/chat')
  const isHomeRoute = location.pathname === '/'
  const isLoginRoute = location.pathname === '/login'
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => {
      const currentY = window.scrollY
      setIsScrolled(currentY > 12)
    }

    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  return (
    <div className="app-shell">
      {!isAppRoute && !isLoginRoute && (
        <header className={`top-nav ${isHomeRoute ? 'home-nav' : 'inner-nav'} ${isScrolled ? 'scrolled' : ''}`}>
          <div className="top-nav-track">
            <div className="top-nav-shell">
              <div className="top-nav-inner">
                <Link to="/" className="brand">
                  <img src={brandLogo} alt="RAGKNO logo" className="brand-logo" />
                  <span>RAGKNO</span>
                </Link>
                <nav className="top-links">
                  <NavLink to="/" className={({ isActive }) => `top-link ${isActive ? 'active' : ''}`}>Home</NavLink>
                  <NavLink to="/data" className={({ isActive }) => `top-link ${isActive ? 'active' : ''}`}>Data</NavLink>
                  <NavLink to="/chat" className={({ isActive }) => `top-link ${isActive ? 'active' : ''}`}>Chat</NavLink>
                </nav>
                <div className="top-actions">
                  <Link to="/chat" className="navbar-button secondary">Login</Link>
                  <Link to="/data" className="navbar-button primary">Get started</Link>
                  <button
                    className="mobile-menu-btn"
                    type="button"
                    aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
                    aria-expanded={mobileOpen}
                    onClick={() => setMobileOpen((prev) => !prev)}
                  >
                    {mobileOpen ? <X size={18} /> : <Menu size={18} />}
                  </button>
                </div>
              </div>
              {mobileOpen && (
                <nav className="mobile-menu" aria-label="Mobile navigation">
                  <NavLink to="/" className={({ isActive }) => `mobile-menu-link ${isActive ? 'active' : ''}`}>Home</NavLink>
                  <NavLink to="/data" className={({ isActive }) => `mobile-menu-link ${isActive ? 'active' : ''}`}>Data</NavLink>
                  <NavLink to="/chat" className={({ isActive }) => `mobile-menu-link ${isActive ? 'active' : ''}`}>Chat</NavLink>
                  <div className="mobile-menu-actions">
                    <Link to="/chat" className="navbar-button secondary" onClick={() => setMobileOpen(false)}>Login</Link>
                    <Link to="/data" className="navbar-button primary" onClick={() => setMobileOpen(false)}>Get started</Link>
                  </div>
                </nav>
              )}
            </div>
          </div>
        </header>
      )}
      <main className={isAppRoute ? 'chat-main' : isHomeRoute || isLoginRoute ? 'home-main' : 'page-main'}>{children}</main>
      {!isAppRoute && !isLoginRoute && (
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
  const words = useMemo(() => ['Docs', 'Links', 'PDFs'], [])
  const [wordIndex, setWordIndex] = useState(0)

  const currentWord = words[wordIndex]

  useEffect(() => {
    const timer = window.setInterval(() => {
      setWordIndex((prev) => (prev + 1) % words.length)
    }, 1800)

    return () => window.clearInterval(timer)
  }, [words.length])

  return (
    <>
      <section className="hero-section">
        <img className="hero-image" src={heroImage} alt="Open field landscape representing an accessible knowledge workspace" />
        <div className="hero-overlay" aria-hidden="true" />
        <div className="hero-grid">
          <div className="hero-copy-block">
            <p className="hero-kicker"><Database size={13} /> Intelligent Retrieval System</p>
            <h1>
              Rag application <br />
              <span className="hero-your-word">
                for your <span key={currentWord} className="typed-word">{currentWord}</span>
              </span>
            </h1>
            <p className="hero-copy">
              Connect, process, and query your knowledge base in one seamless, high-performance interface.
              Engineered for speed, precision, and absolute clarity.
            </p>
            <div className="hero-actions">
              <Link to="/data" className="btn-primary-solid">Get Started <ArrowUp size={15} /></Link>
              <Link to="/chat" className="btn-glass">View Documentation</Link>
            </div>
          </div>
        </div>
      </section>

      <section className="prompt-band">
        <div className="prompt-dot-grid" aria-hidden="true">
          <DotGrid
            dotSize={3}
            gap={24}
            baseColor="#d8d0bf"
            activeColor="#a99c82"
            proximity={120}
            speedTrigger={90}
            shockRadius={210}
            shockStrength={1.8}
            resistance={760}
            returnDuration={1.35}
          />
        </div>
        <div className="prompt-card">
          <p>
            Crafting intelligent retrieval experiences that blend speed, clarity, and purpose.
            Every query is designed to surface your knowledge effectively while maintaining precision and simplicity.
          </p>
          <div className="prompt-tags">
            <span><Database size={12} /> Data Processing</span>
            <span><FolderSync size={12} /> Knowledge Base</span>
            <span><FileText size={12} /> Document Q&A</span>
            <span><ArrowUp size={12} /> Analytics</span>
          </div>
          <div className="prompt-actions">
            <button type="button" className="language-pill">English</button>
            <Link to="/data" className="generate-pill">Generate <ArrowUp size={14} /></Link>
          </div>
        </div>
        <p className="built-label">Built for your docs</p>
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
          <article className="feature-card">
            <span className="feature-icon"><FolderSync size={18} /></span>
            <h3>Universal Knowledge Mapping</h3>
            <p>Map relationships between documents, links, and drive assets with semantic precision.</p>
          </article>
          <article className="feature-card dark-card">
            <span className="feature-icon"><Shield size={18} /></span>
            <h3>Hardened Privacy</h3>
            <p>Enterprise-grade controls keep each query and source isolated and secure.</p>
          </article>
          <article className="feature-card">
            <span className="feature-icon"><ArrowUp size={18} /></span>
            <h3>Millisecond Latency</h3>
            <p>Retrieve top context chunks instantly from your indexed knowledge base.</p>
          </article>
        </div>
      </section>

      <section className="connect-section">
        <div className="connect-text">
          <h2>Connect Everything.</h2>
          <div className="connect-list">
            <article className="connect-item">
              <span className="connect-index">01</span>
              <div>
                <h3>Google Drive & Workspace</h3>
                <p>Sync your entire organizational memory in seconds.</p>
              </div>
            </article>
            <article className="connect-item">
              <span className="connect-index">02</span>
              <div>
                <h3>Complex PDFs & Tables</h3>
                <p>Extract structural data from static files with zero loss.</p>
              </div>
            </article>
            <article className="connect-item">
              <span className="connect-index">03</span>
              <div>
                <h3>Live Web Crawling</h3>
                <p>Maintain live indexes of external documentation and sites.</p>
              </div>
            </article>
          </div>
        </div>
        <div className="connect-tiles">
          <div className="tile large">
            <img src={serverRacksImage} alt="Server racks for connected knowledge infrastructure" />
          </div>
          <div className="tile">
            <img src={serverCablesImage} alt="Server cables representing data ingestion pipelines" />
          </div>
          <div className="tile dark-tile">
            <span>Optimized for</span>
            <span>enterprise</span>
            <span>deployment</span>
          </div>
        </div>
      </section>

      <section className="faq-section">
        <div className="faq-head">
          <h2>Frequently Asked Questions.</h2>
          <p>Everything you need to know before deploying your knowledge archive.</p>
        </div>

        <div className="faq-list">
          <details className="faq-item">
            <summary>What data sources can I connect?</summary>
            <p>
              You can connect Google Drive, upload local files like PDF/DOCX/TXT, and ingest public URLs.
              All sources are indexed into your retrieval pipeline.
            </p>
          </details>

          <details className="faq-item">
            <summary>How does the assistant cite answers?</summary>
            <p>
              Every generated answer can include source-backed citations linked to retrieved chunks, so you can
              verify where each claim came from.
            </p>
          </details>

          <details className="faq-item">
            <summary>Is chat memory isolated by session?</summary>
            <p>
              Yes. Chat context is tracked per session/thread, and you can clear memory anytime from the chat
              controls.
            </p>
          </details>

          <details className="faq-item">
            <summary>Can I evaluate retrieval quality?</summary>
            <p>
              Yes. The project includes RAGAS-based evaluation tooling to measure faithfulness, answer relevancy,
              and context precision on your test set.
            </p>
          </details>
        </div>
      </section>

      <section className="cta-section" style={{ '--cta-image': `url(${ctaTexture})` }}>
        <h2>Ready to archive the future?</h2>
        <div className="cta-actions">
          <Link className="btn-primary-solid" to="/data">Get Started Now</Link>
          <Link className="btn-outline" to="/chat">Request Demo</Link>
        </div>
      </section>
    </>
  )
}

function PixelMaskVisual() {
  return (
    <div className="login-visual">
      <img src={heroImage} alt="RagKno knowledge landscape" />
      <div className="pixel-mask" aria-hidden="true" />
    </div>
  )
}

function LoginPage({ onUserChange, onToast }) {
  const [loading, setLoading] = useState(false)

  async function startGoogleLogin() {
    setLoading(true)
    try {
      const { url } = await getGoogleLoginUrl()
      window.location.href = url
    } catch (error) {
      setLoading(false)
      onToast?.({ type: 'error', message: error.message || 'Unable to start Google login.' })
    }
  }

  useEffect(() => {
    let ignore = false
    async function checkExistingSession() {
      try {
        const result = await getCurrentUser()
        if (!ignore && result.authenticated) {
          onUserChange?.(result.user)
        }
      } catch {
        // Login page remains usable if the backend is not up yet.
      }
    }
    void checkExistingSession()
    return () => {
      ignore = true
    }
  }, [onUserChange])

  return (
    <section className="login-page">
      <div className="login-panel">
        <Link to="/" className="login-brand">
          <img src={brandLogo} alt="RAGKNO logo" />
          <span>RagKno</span>
        </Link>

        <div className="login-copy">
          <h1>Create your account</h1>
          <button className="login-google-btn" type="button" onClick={startGoogleLogin} disabled={loading}>
            <img className="google-mark" src={googleLogo} alt="" />
            {loading ? 'Opening Google...' : 'Continue with Google'}
          </button>
          <p className="login-terms">
            By continuing you agree to RagKno&apos;s terms and can connect data sources after sign in.
          </p>
        </div>

        <div className="login-footer">
          <span>© 2026 RagKno</span>
          <span>Privacy</span>
          <span>Terms</span>
        </div>
      </div>
      <PixelMaskVisual />
    </section>
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

function UserAvatar({ user, displayName, className = ' ' }) {
  const [attempt, setAttempt] = useState(0)
  const initial = (displayName || user?.name || user?.email || 'U').trim().slice(0, 1).toUpperCase()

  const sources = useMemo(() => {
    const list = []
    if (user?.avatar_url) {
      const url = user.avatar_url.startsWith('http') ? user.avatar_url : `${API_BASE}${user.avatar_url}`
      list.push(url)
    }
    if (user?.picture) {
      const proxyUrl = `${API_BASE}/auth/avatar?url=${encodeURIComponent(user.picture)}`
      if (!list.includes(proxyUrl)) list.push(proxyUrl)
      if (!list.includes(user.picture)) list.push(user.picture)
    }
    return list
  }, [user?.avatar_url, user?.picture])

  useEffect(() => {
    setAttempt(0)
  }, [sources])

  if (sources.length > 0 && attempt < sources.length) {
    return (
      <img
        src={sources[attempt]}
        alt={displayName || 'Profile'}
        referrerPolicy="no-referrer"
        loading="eager"
        className={className}
        onError={() => setAttempt((prev) => prev + 1)}
      />
    )
  }

  return <span className={className}>{initial}</span>
}

function ChatPage({ user, onUserChange, onToast }) {
  const location = useLocation()
  const navigate = useNavigate()
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
      action: message?.action || null,
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
  const { t } = useI18n()
  const [chatgptModalOpen, setChatgptModalOpen] = useState(false)
  const [chatgptModalTab, setChatgptModalTab] = useState('general')
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false)

  const openChatGPTModal = (tab = 'general') => {
    setChatgptModalTab(tab)
    setChatgptModalOpen(true)
  }

  useEffect(() => {
    function handleKeyDown(e) {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && (e.key === ',' || e.keyCode === 188)) {
        e.preventDefault()
        openChatGPTModal('general')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
  const [error, setError] = useState('')
  const [hoveredCitation, setHoveredCitation] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [expandedSourceMessages, setExpandedSourceMessages] = useState(new Set())
  const [hoveredSourceMessage, setHoveredSourceMessage] = useState(null)
  const [indexedSources, setIndexedSources] = useState([])
  const [sourcesLoading, setSourcesLoading] = useState(false)
  const [sourceMenuOpen, setSourceMenuOpen] = useState(false)
  const [sourceModalOpen, setSourceModalOpen] = useState(false)
  const [profileMenuOpen, setProfileMenuOpen] = useState(false)
  const [selectedSourceKeys, setSelectedSourceKeys] = useState(new Set())
  const [quickUrl, setQuickUrl] = useState('')
  const [quickUploading, setQuickUploading] = useState(false)
  const [quickAddingUrl, setQuickAddingUrl] = useState(false)
  const quickFileInputRef = useRef(null)
  const bottomRef = useRef(null)

  const appView = location.pathname.endsWith('/data')
    ? 'data'
    : location.pathname.endsWith('/apps')
      ? 'apps'
      : 'chat'

  const sortedThreads = useMemo(
    () => [...threads].sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0)),
    [threads],
  )

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) || sortedThreads[0],
    [threads, activeThreadId, sortedThreads],
  )

  const messages = activeThread?.messages || []
  const rawFirstName = user?.name?.split(' ')?.[0] || user?.email?.split('@')?.[0] || 'there'
  const firstName = rawFirstName ? `${rawFirstName.slice(0, 1).toUpperCase()}${rawFirstName.slice(1)}` : 'there'
  const displayName = firstName === 'there' ? 'RagKno user' : firstName

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

  async function refreshIndexedSources() {
    setSourcesLoading(true)
    try {
      const result = await getIndexedSources()
      const sources = result.sources || []
      setIndexedSources(sources)
      setSelectedSourceKeys((prev) => new Set([...prev].filter((key) => sources.some((item) => item.key === key))))
    } catch (error) {
      onToast?.({ type: 'error', message: error.message || 'Failed to load indexed sources.' })
    } finally {
      setSourcesLoading(false)
    }
  }

  useEffect(() => {
    void refreshIndexedSources()
  }, [])

  function addGuidedIndexMessage(userText) {
    if (!activeThread) return
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const userMessage = {
      id: makeMessageId('user'),
      role: 'user',
      text: userText,
      ts: now,
      sources: [],
      streaming: false,
    }
    const assistantMessage = {
      id: makeMessageId('assistant'),
      role: 'assistant',
      text: `Hi ${user?.name?.split(' ')?.[0] || 'there'}, I can answer from your documents once your knowledge base has data. Please index a file, Drive document, or URL first.`,
      ts: now,
      sources: [],
      streaming: false,
      action: { label: 'Open Data Center', to: '/chat/data' },
    }
    updateActiveThread((thread) => ({
      ...thread,
      title: thread.messages.some((msg) => msg.role === 'user') ? thread.title : makeThreadTitleFromMessage(userText),
      updatedAt: Date.now(),
      messages: [...thread.messages, userMessage, assistantMessage],
    }))
  }

  function toggleSourceSelection(key) {
    setSelectedSourceKeys((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  async function unindexSelectedSources() {
    const keys = [...selectedSourceKeys]
    if (keys.length === 0) return
    const confirmed = window.confirm(`Unindex ${keys.length} selected source(s)?`)
    if (!confirmed) return

    try {
      for (const key of keys) {
        await unindexSource(key)
      }
      onToast?.({ type: 'success', message: `Unindexed ${keys.length} source(s).` })
      setSelectedSourceKeys(new Set())
      await refreshIndexedSources()
    } catch (error) {
      onToast?.({ type: 'error', message: error.message || 'Failed to unindex selected sources.' })
    }
  }

  async function quickUploadFiles(files) {
    const picked = Array.from(files || [])
    if (picked.length === 0) return
    setQuickUploading(true)
    try {
      const result = await ingestFiles(picked)
      onToast?.({ type: 'success', message: result.message || 'Files indexed.' })
      await refreshIndexedSources()
      setSourceModalOpen(false)
    } catch (error) {
      onToast?.({ type: 'error', message: error.message || 'Upload failed.' })
    } finally {
      setQuickUploading(false)
    }
  }

  async function quickAddUrl() {
    if (!quickUrl.trim()) return
    setQuickAddingUrl(true)
    try {
      const result = await ingestUrl(quickUrl.trim())
      onToast?.({ type: 'success', message: result.message || 'URL indexed.' })
      setQuickUrl('')
      await refreshIndexedSources()
      setSourceModalOpen(false)
    } catch (error) {
      onToast?.({ type: 'error', message: error.message || 'URL indexing failed.' })
    } finally {
      setQuickAddingUrl(false)
    }
  }

  async function handleLogout() {
    try {
      await logoutUser()
    } catch {
      // Local logout still clears the UI session.
    }
    setProfileMenuOpen(false)
    onUserChange?.(null)
    navigate('/login', { replace: true })
  }

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

    if (indexedSources.length === 0) {
      addGuidedIndexMessage(text)
      setInput('')
      setError('')
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
        const noIndexedData = String(fallbackError.message || '').toLowerCase().includes('no documents indexed')
        updateMessageById(assistantMessageId, (message) => ({
          ...message,
          streaming: false,
          text: noIndexedData
            ? `Hi ${user?.name?.split(' ')?.[0] || 'there'}, I can answer from your documents once your knowledge base has data. Please index a file, Drive document, or URL first.`
            : message.text || 'The response failed. Please retry.',
          action: noIndexedData ? { label: 'Open Data Center', to: '/chat/data' } : null,
        }))
        if (noIndexedData) {
          setError('')
        } else {
          setError(fallbackError.message)
          onToast?.({
            type: 'error',
            message: fallbackError.message,
            actionLabel: 'Retry',
            onAction: () => { },
          })
        }
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
    setSourceMenuOpen(false)
    setSourceModalOpen(false)
    navigate('/chat')
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
      {!sidebarCollapsed && (
        <aside className="chat-sidebar">
          <div className="chat-sidebar-header">
            <Link to="/chat" className="chat-sidebar-brand">
              <img src={brandLogo} alt="RAGKNO logo" />
              <span>RagKno</span>
            </Link>
            <div className="chat-sidebar-header-actions">
              <button
                className="sidebar-action-icon-btn"
                type="button"
                onClick={() => onToast('Search chats')}
                aria-label="Search chats"
                title="Search chats"
              >
                <Search size={16} />
              </button>
              <button
                className="sidebar-toggle-btn"
                type="button"
                onClick={() => setSidebarCollapsed(true)}
                aria-label="Collapse sidebar"
                title="Collapse sidebar"
              >
                <PanelLeftClose size={17} />
              </button>
            </div>
          </div>

          <button className="history-btn new-chat-pill" onClick={startNewChat}>
            <SquarePen size={14} />
            <span>{t('newChat') || 'New chat'}</span>
          </button>

          <div className="sidebar-nav-group">
            <button className={`history-btn ${appView === 'chat' ? 'active' : ''}`} onClick={() => navigate('/chat')}>
              <MessageSquare size={14} /> <span>Chats</span>
            </button>
            <button className={`history-btn ${appView === 'data' ? 'active' : ''}`} onClick={() => navigate('/chat/data')}>
              <Database size={14} /> <span>Data Center</span>
            </button>
            <button className={`history-btn ${appView === 'apps' ? 'active' : ''}`} onClick={() => navigate('/chat/apps')}>
              <Cloud size={14} /> <span>{t('connectApps') || 'Connect Apps'}</span>
            </button>
          </div>

          <div className="chat-sidebar-section-title">
            <span>{t('yourChats') || 'Chats'}</span>
          </div>

          <div className="sidebar-threads-scroll">
            {sortedThreads.map((thread) => (
              <div key={thread.id} className={`history-item ${thread.id === activeThread?.id ? 'active' : ''}`}>
                <button
                  className={`history-btn ${thread.id === activeThread?.id ? 'active' : ''}`}
                  onClick={() => setActiveThreadId(thread.id)}
                  title={thread.title || 'New Chat'}
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
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>

          <div className="chat-sidebar-footer">
            {profileMenuOpen && (
              <div className="profile-dropup">
                <p>{user?.email || 'Signed in'}</p>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openChatGPTModal('general'); }}><Settings size={15} /> {t('settings')} <span>⇧⌘,</span></button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openChatGPTModal('help'); }}><CircleHelp size={15} /> {t('helpFaq')}</button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openChatGPTModal('about'); }}><Info size={15} /> {t('learnMore')} <ChevronRight size={14} /></button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); setFeedbackModalOpen(true); }}><MessageSquareHeart size={15} /> {t('giveFeedback')}</button>
                <hr />
                <button type="button" onClick={handleLogout}><LogOut size={15} /> {t('logOut')}</button>
              </div>
            )}
            <button className="user-pill" type="button" onClick={() => setProfileMenuOpen((open) => !open)} aria-expanded={profileMenuOpen}>
              <div className="user-avatar-badge">
                <UserAvatar user={user} displayName={displayName} />
              </div>
              <div className="user-info-text">
                <strong>{displayName}</strong>
                <small>Go</small>
              </div>
              <ChevronsUpDown size={13} className="user-pill-chevron" />
            </button>
          </div>
        </aside>
      )}

      <section className={`chat-canvas ${appView !== 'chat' ? 'utility-view' : ''} ${appView === 'chat' && messages.length === 0 ? 'is-empty' : ''}`}>
        {sidebarCollapsed && (
          <header className="chat-canvas-top-bar">
            <button
              className="canvas-toggle-btn"
              type="button"
              onClick={() => setSidebarCollapsed(false)}
              aria-label="Open sidebar"
              title="Open sidebar"
            >
              <PanelLeft size={18} />
            </button>
            <div className="canvas-model-selector" onClick={() => onToast('RagKno 1.0 (GPT-5.6-sol active)')}>
              <span>RagKno</span>
              <ChevronDown size={14} />
            </div>
            <button
              className="canvas-new-chat-btn"
              type="button"
              onClick={startNewChat}
              aria-label="New chat"
              title="New chat"
            >
              <SquarePen size={18} />
            </button>
          </header>
        )}

        {appView === 'data' && (
          <div className="embedded-data-center">
            <DataPage onToast={onToast} />
          </div>
        )}

        {appView === 'apps' && (
          <div className="connect-apps-view">
            <span className="view-kicker">Connect Apps</span>
            <h1>Bring more context into RagKno.</h1>
            <p>Google Drive is available now. Slack, Notion, GitHub, and Gmail connectors can be added here as the product grows.</p>
            <button className="btn-primary-solid" type="button" onClick={() => navigate('/chat/data')}>
              Open Data Center
            </button>
          </div>
        )}

        {appView === 'chat' && (
          <>
            <div className="message-stack">
              {!messages.length && !loading && (
                <div className="chat-empty-state">
                  <h1>How can I help, {firstName}?</h1>
                </div>
              )}

              {messages.map((message) => (
                <div key={message.id} className={`message ${message.role}`}>
                  <div className="bubble">
                    {message.role === 'assistant' ? renderAssistantText(message.text, message) : message.text}
                    {message.action?.to && (
                      <Link className="message-action-link" to={message.action.to}>
                        {message.action.label || 'Open'}
                      </Link>
                    )}
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

            <div className="chat-composer-container">
              <form className="chat-input-row" onSubmit={sendMessage}>
                <button
                  type="button"
                  className="prompt-tool-btn"
                  aria-label="Add source"
                  onClick={() => setSourceModalOpen(true)}
                  title="Add documents or links"
                >
                  <Plus size={19} />
                </button>
                <input
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={t('askAnything') || 'Ask anything...'}
                />
                <div className="chat-input-right-tools">
                  <div className="source-dropdown">
                    <button
                      type="button"
                      className="source-dropdown-btn"
                      onClick={() => {
                        setSourceMenuOpen((prev) => !prev)
                        if (!sourceMenuOpen) void refreshIndexedSources()
                      }}
                      title="Sources selector"
                    >
                      <Database size={13} />
                      <span>{t('sources') || 'Sources'}</span>
                      <span className="source-count-badge">{indexedSources.length}</span>
                      <ChevronDown size={12} />
                    </button>
                    {sourceMenuOpen && (
                      <div className="source-dropdown-menu">
                        <div className="source-dropdown-head">
                          <strong>Indexed sources</strong>
                          <button type="button" onClick={refreshIndexedSources} disabled={sourcesLoading}>
                            <RefreshCw size={13} className={sourcesLoading ? 'spin' : ''} />
                          </button>
                        </div>
                        {indexedSources.length === 0 && (
                          <p className="source-empty">No sources indexed yet.</p>
                        )}
                        {indexedSources.map((source) => (
                          <label key={source.key} className="source-option">
                            <input
                              type="checkbox"
                              checked={selectedSourceKeys.has(source.key)}
                              onChange={() => toggleSourceSelection(source.key)}
                            />
                            <span>
                              <strong>{source.source}</strong>
                              <small>{source.type}</small>
                            </span>
                          </label>
                        ))}
                        <div className="source-dropdown-actions">
                          <button type="button" onClick={() => navigate('/chat/data')}>Data Center</button>
                          <button type="button" onClick={unindexSelectedSources} disabled={selectedSourceKeys.size === 0}>
                            <Trash2 size={13} /> Unindex
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  <button
                    type="submit"
                    className="send-btn"
                    disabled={!input.trim() || loading}
                    title="Send message"
                  >
                    <ArrowUp size={16} strokeWidth={2.5} />
                  </button>
                </div>
              </form>

              <p className="chat-disclaimer">{t('disclaimer') || 'RagKno can make mistakes. Verify critical information.'}</p>
            </div>
          </>
        )}
      </section>

      {sourceModalOpen && (
        <div className="source-modal-backdrop" role="presentation" onMouseDown={() => setSourceModalOpen(false)}>
          <div className="source-modal" role="dialog" aria-modal="true" aria-label="Add source" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div>
                <span className="view-kicker">Add Source</span>
                <h2>Index new data without leaving chat.</h2>
              </div>
              <button type="button" onClick={() => setSourceModalOpen(false)} aria-label="Close add source">
                <X size={18} />
              </button>
            </header>

            <div className="source-modal-grid">
              <button className="source-modal-card" type="button" onClick={() => quickFileInputRef.current?.click()}>
                <Upload size={24} />
                <strong>{quickUploading ? 'Uploading...' : 'Upload files'}</strong>
                <span>PDF, DOCX, and TXT documents.</span>
              </button>
              <button className="source-modal-card" type="button" onClick={() => navigate('/chat/data')}>
                <Cloud size={24} />
                <strong>Connect Drive</strong>
                <span>Open Data Center for Google Drive sync.</span>
              </button>
            </div>

            <div className="source-url-row">
              <input
                type="url"
                value={quickUrl}
                onChange={(event) => setQuickUrl(event.target.value)}
                placeholder="https://example.com/docs"
              />
              <button type="button" onClick={quickAddUrl} disabled={!quickUrl.trim() || quickAddingUrl}>
                {quickAddingUrl ? 'Adding...' : 'Add URL'}
              </button>
            </div>

            <input
              ref={quickFileInputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              multiple
              hidden
              onChange={(event) => {
                void quickUploadFiles(event.target.files)
                event.target.value = ''
              }}
            />
          </div>
        </div>
      )}

      <ChatGPTSettingsModal
        isOpen={chatgptModalOpen}
        onClose={() => setChatgptModalOpen(false)}
        initialTab={chatgptModalTab}
        user={user}
        onClearHistory={() => {
          setThreads([])
          setActiveThreadId(null)
          try {
            localStorage.removeItem(CHAT_THREADS_KEY)
            localStorage.removeItem(ACTIVE_THREAD_KEY)
          } catch {}
          onToast?.({ type: 'info', message: t('historyCleared') || 'Chat history cleared.' })
        }}
      />

      <FeedbackModal
        isOpen={feedbackModalOpen}
        onClose={() => setFeedbackModalOpen(false)}
        onSubmitFeedback={async (feedbackData) => {
          try {
            await submitFeedback({
              rating: feedbackData.rating,
              comment: feedbackData.comment,
              user_id: user?.id,
              user_email: user?.email,
            })
            onToast?.({ type: 'success', message: 'Thank you for your feedback!' })
          } catch (err) {
            onToast?.({ type: 'error', message: 'Failed to submit feedback. Please try again.' })
          }
        }}
      />
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

function LenisScrollController() {
  const { pathname } = useLocation()

  useEffect(() => {
    if (pathname.startsWith('/chat')) {
      return undefined
    }

    const lenis = new Lenis({
      duration: 1.05,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    })

    function raf(time) {
      lenis.raf(time)
      window.requestAnimationFrame(raf)
    }

    const frame = window.requestAnimationFrame(raf)
    return () => {
      window.cancelAnimationFrame(frame)
      lenis.destroy()
    }
  }, [pathname])

  return null
}

function HandleOAuthRedirect() {
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    if (params.get('drive') === 'connected') {
      navigate('/chat/data?connected=1', { replace: true })
    }
  }, [location.search, navigate])

  return null
}

function ProtectedRoute({ user, checkingAuth, children }) {
  const location = useLocation()
  if (checkingAuth) {
    return (
      <div className="auth-loading">
        <span>Loading RagKno...</span>
      </div>
    )
  }
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }
  return children
}

export default function App() {
  const [toasts, setToasts] = useState([])
  const [user, setUser] = useState(null)
  const [checkingAuth, setCheckingAuth] = useState(true)

  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('dark')
    root.style.colorScheme = 'light'
  }, [])

  useEffect(() => {
    let ignore = false
    async function loadUser() {
      try {
        const result = await getCurrentUser()
        if (!ignore) {
          setUser(result.authenticated ? result.user : null)
        }
      } catch {
        if (!ignore) {
          setUser(null)
        }
      } finally {
        if (!ignore) {
          setCheckingAuth(false)
        }
      }
    }
    void loadUser()
    return () => {
      ignore = true
    }
  }, [])

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
      <LenisScrollController />
      <ScrollToTop />
      <HandleOAuthRedirect />
      <AppShell
        toasts={toasts}
        onDismissToast={dismissToast}
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/login"
            element={user ? <Navigate to="/chat" replace /> : <LoginPage onUserChange={setUser} onToast={pushToast} />}
          />
          <Route path="/data" element={<Navigate to="/chat/data" replace />} />
          <Route
            path="/chat/*"
            element={(
              <ProtectedRoute user={user} checkingAuth={checkingAuth}>
                <ChatPage user={user} onUserChange={setUser} onToast={pushToast} />
              </ProtectedRoute>
            )}
          />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}
