import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BrowserRouter, Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import Lenis from 'lenis'
import { HugeiconsIcon } from '@hugeicons/react'
import { LayoutAlignLeftIcon, LayoutAlignRightIcon } from '@hugeicons/core-free-icons'
import {
  ArrowUp,
  Check,
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
  MoreHorizontal,
  PanelLeft,
  PanelLeftClose,
  Plus,
  Pencil,
  RefreshCw,
  Search,
  Send,
  Shield,
  Settings,
  SquarePen,
  Trash2,
  Upload,
  X,
  ArrowRight,
  ArrowDown,
} from 'lucide-react'
import {
  disconnectDrive,
  getAuthStatus,
  getAuthUrl,
  getCurrentUser,
  getDriveFiles,
  getIndexedSources,
  ingestFiles,
  ingestUrl,
  logoutUser,
  queryRAG,
  queryRAGStream,
  syncDrive,
  unindexSource,
  submitFeedback,
  getThreads,
  getThreadMessages,
  createBackendThread,
  renameBackendThread,
  deleteBackendThread,
  API_BASE,
} from './api.js'
import FeedbackModal from './components/FeedbackModal.jsx'
import SettingsModal from './components/SettingsModal.jsx'
import DotGrid from './components/DotGrid.jsx'
import brandLogo from './assets/figma-logo-mark.svg'
import { useI18n } from './lib/i18n.jsx'
import { parseMarkdownTable } from './lib/markdownTable.js'
import useHomepageReveal from './lib/useHomepageReveal.js'
import heroImage from './assets/ragkno-hero.webp'
import serverRacksImage from './assets/server-racks.png'
import serverCablesImage from './assets/server-cables.png'
import ctaTexture from './assets/cta-texture.webp'
import KnowledgeSourcesMarquee from './components/landing/KnowledgeSourcesMarquee.jsx'
import BentoCapabilities from './components/landing/BentoCapabilities.jsx'
import HowRagknoWorks from './components/landing/HowRagknoWorks.jsx'
import ProductChatDemo from './components/landing/ProductChatDemo.jsx'
import UseCasesSection from './components/landing/UseCasesSection.jsx'
import FrequentlyAskedQuestions from './components/landing/FrequentlyAskedQuestions.jsx'
import AuthPage from './components/AuthPage.jsx'
import LegalPage from './components/LegalPage.jsx'
import DocsPage from './components/DocsPage.jsx'
import ChatErrorBoundary from './components/ChatErrorBoundary.jsx'
import SiteFooter from './components/SiteFooter.jsx'

function GitHubNavIcon({ size = 20 }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden="true">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  )
}

function AppShell({ children, toasts, onDismissToast }) {
  const location = useLocation()
  const isChatRoute = location.pathname === '/chat'
  const isAppRoute = location.pathname.startsWith('/chat')
  const isHomeRoute = location.pathname === '/'
  const isLoginRoute = location.pathname === '/login'
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)
  const [navTone, setNavTone] = useState(isHomeRoute ? 'dark' : 'light')

  useEffect(() => {
    let ticking = false
    const readSurfaceTone = () => {
      if (!isHomeRoute) {
        setNavTone('light')
        return
      }

      const sampleY = Math.min(54, window.innerHeight - 1)
      const themedSurface = document
        .elementsFromPoint(window.innerWidth / 2, sampleY)
        .filter((element) => !element.closest('.top-nav'))
        .map((element) => element.closest?.('[data-nav-theme]'))
        .find(Boolean)
      setNavTone(themedSurface?.dataset.navTheme === 'dark' ? 'dark' : 'light')
    }

    const onScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const currentY = window.scrollY
          setIsScrolled((prev) => {
            const next = currentY > 12
            return prev === next ? prev : next
          })
          readSurfaceTone()
          ticking = false
        })
        ticking = true
      }
    }

    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [isHomeRoute])

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  return (
    <div className="app-shell">
      {!isAppRoute && !isLoginRoute && (
        <header className={`top-nav ${isHomeRoute ? 'home-nav' : 'inner-nav'} ${isScrolled ? 'scrolled' : ''} ${mobileOpen ? 'mobile-open' : ''} nav-on-${navTone}`}>
          <div className="top-nav-track">
            <div className="top-nav-shell">
              <div className="top-nav-inner">
                <Link to="/" className="brand">
                  <img src={brandLogo} alt="RAGKNO logo" className="brand-logo" />
                  <span>RAGKNO</span>
                </Link>
                <nav className="top-links" aria-label="Primary navigation">
                  <Link to="/#capabilities" className="top-link">Features</Link>
                  <Link to="/#how-it-works" className="top-link">How it works</Link>
                  <Link to="/docs" className="top-link">Docs</Link>
                  <Link to="/#faq" className="top-link">FAQ</Link>
                </nav>
                <div className="top-actions">
                  <a
                    href="https://github.com/GitHpriyanshu23/Ragkno.git"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="navbar-github-link"
                    aria-label="GitHub Repository"
                    title="GitHub Repository"
                  >
                    <GitHubNavIcon size={18} />
                    <span>GitHub</span>
                  </a>
                  <Link to="/login?mode=signin" className="navbar-button secondary">Login</Link>
                  <Link to="/login?mode=signup" className="navbar-button primary">Get started</Link>
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
                <nav className="mobile-menu is-open" aria-label="Mobile navigation">
                  <div className="mobile-menu-links">
                    <Link to="/#capabilities" className="mobile-menu-link" onClick={() => setMobileOpen(false)}>Features</Link>
                    <Link to="/#how-it-works" className="mobile-menu-link" onClick={() => setMobileOpen(false)}>How it works</Link>
                    <Link to="/docs" className="mobile-menu-link" onClick={() => setMobileOpen(false)}>Docs</Link>
                    <Link to="/#faq" className="mobile-menu-link" onClick={() => setMobileOpen(false)}>FAQ</Link>
                  </div>
                  <div className="mobile-menu-actions">
                    <a
                      href="https://github.com/GitHpriyanshu23/Ragkno.git"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="navbar-github-link"
                      aria-label="GitHub Repository"
                      title="GitHub Repository"
                      onClick={() => setMobileOpen(false)}
                    >
                      <GitHubNavIcon size={18} />
                      <span>GitHub</span>
                    </a>
                    <Link to="/login?mode=signin" className="navbar-button secondary" onClick={() => setMobileOpen(false)}>Login</Link>
                    <Link to="/login?mode=signup" className="navbar-button primary" onClick={() => setMobileOpen(false)}>Get started</Link>
                  </div>
                </nav>
              )}
            </div>
          </div>
        </header>
      )}
      <main className={isAppRoute ? 'chat-main' : isHomeRoute || isLoginRoute ? 'home-main' : 'page-main'}>{children}</main>

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
  const homeRef = useRef(null)
  const words = useMemo(() => ['Docs', 'Links', 'PDFs'], [])
  const [wordIndex, setWordIndex] = useState(0)

  const currentWord = words[wordIndex]
  useHomepageReveal(homeRef)

  useEffect(() => {
    const timer = window.setInterval(() => {
      setWordIndex((prev) => (prev + 1) % words.length)
    }, 1800)

    return () => window.clearInterval(timer)
  }, [words.length])

  return (
    <div className="home-page" ref={homeRef}>
      <section className="hero-section" data-nav-theme="dark">
        <img className="hero-image" src={heroImage} alt="Open field landscape representing an accessible knowledge workspace" fetchpriority="high" />
        <div className="hero-overlay" aria-hidden="true" />
        <div className="hero-grid">
          <div className="hero-copy-block">
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
              <Link to="/data" className="btn-primary-solid">Get Started <ArrowRight size={15} /></Link>
              <Link to="/docs" className="btn-glass">View Documentation</Link>
            </div>
          </div>
        </div>
      </section>

      <section className="prompt-band" data-nav-theme="light">
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
            <Link to="/data" className="generate-pill">Generate <ArrowRight size={14} /></Link>
          </div>
        </div>
        <p className="built-label">Built for your</p>
        <KnowledgeSourcesMarquee />
      </section>

      {/* 2. Bento Capabilities (Dark #090909) */}
      <BentoCapabilities />

      {/* 4. How RAGKNO Works (Dark #090909) */}
      <HowRagknoWorks />

      {/* 5. Product / Chat Demo (Light #FAFAFA) */}
      <ProductChatDemo />

      {/* 6. Use Cases / Knowledge Types (Dark #090909) */}
      <UseCasesSection />

      {/* 7. Frequently Asked Questions */}
      <FrequentlyAskedQuestions />

      {/* 8. Final CTA (Dark Texture) */}
      <section className="cta-section" data-nav-theme="light" style={{ '--cta-image': `url(${ctaTexture})` }}>
        <h2>Turn your knowledge into answers</h2>
        <div className="cta-actions">
          <Link className="btn-primary-solid" to="/data">Get Started Now</Link>
          <Link className="btn-outline" to="/chat">Request Demo</Link>
        </div>
      </section>

      <SiteFooter />
    </div>
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
    } else if (params.get('error')) {
      const err = params.get('error')
      if (err === 'access_denied') {
        showError('Google Drive access was denied (access_denied). In Google Cloud Console, ensure your email is added under "OAuth Consent Screen" > "Test Users", and click "Advanced" > "Go to RagKno (unsafe)" on Google\'s consent prompt.')
      } else {
        showError(`Google Drive connection failed: ${err}`)
      }
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
      const sources = (result.sources || []).map(repairSourceMetadata)
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

function isSimpleGreeting(value) {
  const normalized = String(value || '')
    .toLowerCase()
    .replace(/[^a-z\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  return /^(hi|hello|hey|hiya|howdy|good morning|good afternoon|good evening|hi there|hello there|hey there)$/.test(normalized)
}

function repairDisplayMojibake(value) {
  return String(value || '')
    .replaceAll('\u00e2\u0082\u00b9', '₹')
    .replaceAll('\u00e2\u201a\u00b9', '₹')
    .replaceAll('\u00c2₹', '₹')
    .replaceAll('\u00e2\u0080\u0093', '–')
    .replaceAll('\u00e2\u20ac\u201c', '–')
    .replaceAll('\u00e2\u0080\u0094', '—')
    .replaceAll('\u00e2\u20ac\u201d', '—')
    .replaceAll('\u00e2\u0080\u00a6', '…')
    .replaceAll('\u00e2\u20ac\u00a6', '…')
    .replaceAll('\u00e2\u0080\u00a2', '•')
    .replaceAll('\u00e2\u20ac\u00a2', '•')
    .replaceAll('\u00e2\u0080\u0099', "'")
    .replaceAll('\u00e2\u20ac\u2122', "'")
    .replaceAll('\u00e2\u0080\u009c', '“')
    .replaceAll('\u00e2\u0080\u009d', '”')
}

function repairSourceMetadata(source) {
  if (!source || typeof source !== 'object') return source
  return {
    ...source,
    source: repairDisplayMojibake(source.source),
    title: repairDisplayMojibake(source.title),
    preview: repairDisplayMojibake(source.preview),
    text: repairDisplayMojibake(source.text),
  }
}

const RETRIEVAL_ACTIVITY_STEPS = [
  { label: 'Understanding your question', Icon: Search },
  { label: 'Embedding your question', Icon: FolderSync },
  { label: 'Searching user-scoped chunks', Icon: Database },
  { label: 'Reranking relevant evidence', Icon: FolderSync },
  { label: 'Writing a response', Icon: Send },
]

function RetrievalActivity({ activity }) {
  const completed = activity?.stage === 'complete'
  const failed = activity?.stage === 'failed'
  const targetIndex = activity?.stage === 'searching'
    ? 2
    : activity?.stage === 'writing'
      ? 4
      : RETRIEVAL_ACTIVITY_STEPS.length
  const [activeIndex, setActiveIndex] = useState(0)
  const [expanded, setExpanded] = useState(!completed)

  useEffect(() => {
    if (completed) {
      setActiveIndex(RETRIEVAL_ACTIVITY_STEPS.length)
      return undefined
    }
    if (failed || activeIndex >= targetIndex) return undefined
    const timer = window.setTimeout(() => {
      setActiveIndex((current) => Math.min(current + 1, targetIndex))
    }, 320)
    return () => window.clearTimeout(timer)
  }, [activeIndex, completed, failed, targetIndex])

  useEffect(() => {
    if (!completed) {
      setExpanded(true)
      return undefined
    }
    const timer = window.setTimeout(() => setExpanded(false), 900)
    return () => window.clearTimeout(timer)
  }, [completed])

  if (!activity) return null

  const label = failed
    ? 'Retrieval interrupted'
    : completed
      ? `Searched ${activity.sourceCount || 0} ${activity.sourceCount === 1 ? 'source' : 'sources'}`
      : RETRIEVAL_ACTIVITY_STEPS[Math.min(activeIndex, RETRIEVAL_ACTIVITY_STEPS.length - 1)].label

  return (
    <div className={`retrieval-activity ${failed ? 'is-failed' : ''}`}>
      <button type="button" className="retrieval-activity-trigger" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>
        {completed ? <Check size={15} /> : <RefreshCw className={failed ? '' : 'spin'} size={15} />}
        <span role="status">{label}</span>
        <ChevronDown size={14} className={expanded ? 'is-open' : ''} />
      </button>
      <div className={`retrieval-activity-panel ${expanded ? 'is-open' : ''}`}>
        <div className="retrieval-activity-steps">
          {RETRIEVAL_ACTIVITY_STEPS.map(({ label: stepLabel, Icon }, index) => {
            const done = completed || index < activeIndex
            const active = !completed && !failed && index === activeIndex
            return (
              <div className={`retrieval-activity-step ${done ? 'is-done' : ''} ${active ? 'is-active' : ''}`} key={stepLabel}>
                <span className="retrieval-step-icon">
                  {done ? <Check size={13} /> : active ? <span className="retrieval-step-spinner" /> : <Icon size={13} />}
                </span>
                <span>{stepLabel}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export function ChatPage({ user, onUserChange, onToast }) {
  const location = useLocation()
  const navigate = useNavigate()
  const abortRef = useRef(null)

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

  function normalizeMessage(message, index = 0) {
    const role = message?.role === 'assistant' ? 'assistant' : 'user'
    return {
      id: message?.id || makeMessageId(`legacy-${index}`),
      role,
      text: String(message?.text || ''),
      ts: String(message?.ts || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
      sources: Array.isArray(message?.sources) ? message.sources.map(repairSourceMetadata) : [],
      streaming: Boolean(message?.streaming),
      interrupted: Boolean(message?.interrupted || message?.action?.interrupted),
      action: message?.action || null,
      activity: null,
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
    const lines = repairDisplayMojibake(text).replace(/\r\n/g, '\n').split('\n')
    const output = []
    let index = 0

    while (index < lines.length) {
      const trimmed = lines[index].trim()
      if (!trimmed) {
        index += 1
        continue
      }

      const table = parseMarkdownTable(lines, index)
      if (table) {
        output.push(
          <div className="assistant-table-wrap" key={`table-${output.length}`}>
            <table className="assistant-table">
              <thead>
                <tr>
                  {table.headers.map((cell, cellIndex) => (
                    <th className={`align-${table.alignments[cellIndex]}`} key={`th-${cellIndex}`} scope="col">
                      {renderInlineText(cell, `table-${output.length}-head-${cellIndex}`, message)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, rowIndex) => (
                  <tr key={`tr-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <td className={`align-${table.alignments[cellIndex]}`} key={`td-${rowIndex}-${cellIndex}`}>
                        {renderInlineText(cell, `table-${output.length}-${rowIndex}-${cellIndex}`, message)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>,
        )
        index = table.nextIndex
        continue
      }

      if (/^[-*•]\s+/.test(trimmed)) {
        const items = []
        while (index < lines.length) {
          const match = lines[index].trim().match(/^[-*•]\s+(.+)/)
          if (!match) break
          items.push(match[1])
          index += 1
        }
        output.push(
          <ul key={`ul-${output.length}`}>
            {items.map((item, itemIndex) => (
              <li key={`uli-${itemIndex}`}>{renderInlineText(item, `ul-${output.length}-${itemIndex}`, message)}</li>
            ))}
          </ul>,
        )
        continue
      }

      if (/^\d+[.)]\s+/.test(trimmed)) {
        const items = []
        while (index < lines.length) {
          const match = lines[index].trim().match(/^\d+[.)]\s+(.+)/)
          if (!match) break
          items.push(match[1])
          index += 1
        }
        output.push(
          <ol key={`ol-${output.length}`}>
            {items.map((item, itemIndex) => (
              <li key={`oli-${itemIndex}`}>{renderInlineText(item, `ol-${output.length}-${itemIndex}`, message)}</li>
            ))}
          </ol>,
        )
        continue
      }

      const headingMatch = trimmed.match(/^#{1,4}\s+(.+)/)
      if (headingMatch) {
        output.push(<h3 className="assistant-response-heading" key={`heading-${output.length}`}>{renderInlineText(headingMatch[1], `heading-${output.length}`, message)}</h3>)
        index += 1
        continue
      }

      const paragraph = [trimmed]
      index += 1
      while (index < lines.length) {
        const next = lines[index].trim()
        if (!next || /^[-*•]\s+/.test(next) || /^\d+[.)]\s+/.test(next) || /^#{1,4}\s+/.test(next) || parseMarkdownTable(lines, index)) break
        paragraph.push(next)
        index += 1
      }
      output.push(
        <p key={`p-${output.length}`}>
          {paragraph.map((part, partIndex) => (
            <span key={`line-${partIndex}`}>{partIndex > 0 && <br />}{renderInlineText(part, `p-${output.length}-${partIndex}`, message)}</span>
          ))}
        </p>,
      )
    }

    return output
  }

  const chatCacheKey = `ragkno_chat_cache_v2:${user?.id || user?.email || 'user'}`
  const [threads, setThreads] = useState(() => {
    try {
      const cached = JSON.parse(localStorage.getItem(chatCacheKey) || '[]')
      return Array.isArray(cached)
        ? cached.map((thread) => ({ ...thread, messagesLoaded: true, pending: false }))
        : []
    } catch {
      return []
    }
  })
  const cachedThreadsRef = useRef(threads)
  const [activeThreadId, setActiveThreadId] = useState(
    () => new URLSearchParams(window.location.search).get('thread'),
  )
  const [threadStartup, setThreadStartup] = useState({ status: 'ready', error: '' })
  const [threadRetry, setThreadRetry] = useState(0)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const { t } = useI18n()
  const [settingsModalOpen, setSettingsModalOpen] = useState(false)
  const [settingsModalTab, setSettingsModalTab] = useState('general')
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [compactSidebarOpen, setCompactSidebarOpen] = useState(false)

  const openSettingsModal = (tab = 'general') => {
    setFeedbackModalOpen(false)
    setSettingsModalTab(tab)
    setSettingsModalOpen(true)
  }
  const openChatGPTModal = openSettingsModal
  const chatgptModalOpen = settingsModalOpen
  const setChatgptModalOpen = setSettingsModalOpen

  useEffect(() => {
    function handleKeyDown(e) {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && (e.key === ',' || e.keyCode === 188)) {
        e.preventDefault()
        openSettingsModal('general')
      }
      if (
        ((e.metaKey || e.ctrlKey) && e.shiftKey && (e.key === 's' || e.key === 'S')) ||
        ((e.metaKey || e.ctrlKey) && (e.key === 'b' || e.key === 'B')) ||
        ((e.metaKey || e.ctrlKey) && e.key === '\\')
      ) {
        e.preventDefault()
        if (window.matchMedia('(max-width: 920px)').matches) {
          setSidebarCollapsed(false)
          setCompactSidebarOpen((open) => !open)
        } else {
          setSidebarCollapsed((prev) => !prev)
        }
      }
      if (e.key === 'Escape') setCompactSidebarOpen(false)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
  const [error, setError] = useState('')
  const [hoveredCitation, setHoveredCitation] = useState(null)
  const [expandedSourceMessages, setExpandedSourceMessages] = useState(new Set())
  const [hoveredSourceMessage, setHoveredSourceMessage] = useState(null)
  const [indexedSources, setIndexedSources] = useState([])
  const [sourcesLoading, setSourcesLoading] = useState(false)
  const [sourceMenuOpen, setSourceMenuOpen] = useState(false)
  const [sourceModalOpen, setSourceModalOpen] = useState(false)
  const [profileMenuOpen, setProfileMenuOpen] = useState(false)
  const [threadMenu, setThreadMenu] = useState(null)
  const [renameDialog, setRenameDialog] = useState(null)
  const [deleteDialog, setDeleteDialog] = useState(null)
  const [threadActionPending, setThreadActionPending] = useState(false)
  const [selectedSourceKeys, setSelectedSourceKeys] = useState(new Set())
  const [quickUrl, setQuickUrl] = useState('')
  const [quickUploading, setQuickUploading] = useState(false)
  const [quickAddingUrl, setQuickAddingUrl] = useState(false)
  const quickFileInputRef = useRef(null)
  const bottomRef = useRef(null)
  const chatScrollRef = useRef(null)
  const sourceDropdownRef = useRef(null)
  const threadMenuRef = useRef(null)
  const renameInputRef = useRef(null)
  const [showScrollBottom, setShowScrollBottom] = useState(false)

  const handleChatScroll = () => {
    const el = chatScrollRef.current
    if (!el) return
    const isOverflowing = el.scrollHeight > el.clientHeight + 80
    const isScrolledUp = el.scrollHeight - el.scrollTop - el.clientHeight > 100
    setShowScrollBottom(isOverflowing && isScrolledUp)
  }

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const appView = location.pathname.endsWith('/data')
    ? 'data'
    : location.pathname.endsWith('/apps')
      ? 'apps'
      : 'chat'

  const requestedThreadId = useMemo(
    () => new URLSearchParams(location.search).get('thread'),
    [location.search],
  )

  const sortedThreads = useMemo(
    () => [...threads].sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0)),
    [threads],
  )

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) || null,
    [threads, activeThreadId],
  )

  const messages = activeThread?.messages || []
  const rawFirstName = user?.name?.split(' ')?.[0] || user?.email?.split('@')?.[0] || 'there'
  const firstName = rawFirstName ? `${rawFirstName.slice(0, 1).toUpperCase()}${rawFirstName.slice(1)}` : 'there'
  const displayName = firstName === 'there' ? 'RagKno user' : firstName

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const cacheable = sortedThreads.slice(0, 12).map((thread) => ({
          id: thread.id,
          sessionId: thread.id,
          title: thread.title,
          updatedAt: thread.updatedAt,
          messagesLoaded: true,
          messages: (thread.messages || []).slice(-60).map((message) => ({
            ...message,
            streaming: false,
            activity: null,
            sources: (message.sources || []).map(({ text, ...source }) => source),
          })),
        }))
        localStorage.setItem(chatCacheKey, JSON.stringify(cacheable))
      } catch {
        // A cache write must never block or break the live conversation.
      }
    }, 450)
    return () => window.clearTimeout(timer)
  }, [chatCacheKey, sortedThreads])

  useEffect(() => {
    let cancelled = false
    async function bootstrapThreads() {
      setThreadStartup({ status: 'ready', error: '' })
      try {
        localStorage.removeItem('ragkno_chat_threads_v1')
        localStorage.removeItem('ragkno_active_thread_v1')
        const [result, requestedMessages] = await Promise.all([
          getThreads(),
          requestedThreadId
            ? getThreadMessages(requestedThreadId).catch(() => null)
            : Promise.resolve(null),
        ])
        const cachedById = new Map(cachedThreadsRef.current.map((thread) => [thread.id, thread]))
        const nextThreads = Array.isArray(result?.threads)
          ? result.threads.map((thread) => ({
            ...thread,
            sessionId: thread.id,
            messages: thread.id === requestedThreadId && requestedMessages
              ? (requestedMessages.messages || []).map((message, index) => normalizeMessage(message, index))
              : (cachedById.get(thread.id)?.messages || []),
            messagesLoaded: thread.id === requestedThreadId && Boolean(requestedMessages),
          }))
          : []
        if (!cancelled) {
          setThreads(nextThreads)
          setThreadStartup({ status: 'ready', error: '' })

          // Warm recent conversations after the shell is usable. This keeps the
          // first paint fast and makes subsequent switches feel immediate.
          const threadsToWarm = nextThreads.slice(0, 12).filter((thread) => !thread.messagesLoaded)
          void Promise.allSettled(threadsToWarm.map(async (thread) => {
            const messagesResult = await getThreadMessages(thread.id)
            return {
              id: thread.id,
              messages: (messagesResult.messages || []).map((message, index) => normalizeMessage(message, index)),
            }
          })).then((settled) => {
            if (cancelled) return
            const warmed = new Map(settled
              .filter((item) => item.status === 'fulfilled')
              .map((item) => [item.value.id, item.value.messages]))
            if (!warmed.size) return
            setThreads((prev) => prev.map((thread) => warmed.has(thread.id)
              ? { ...thread, messages: warmed.get(thread.id), messagesLoaded: true }
              : thread))
          })
        }
      } catch (err) {
        if (!cancelled) {
          setThreadStartup({ status: 'ready', error: err.message || 'Failed to load conversations.' })
          onToast?.({ type: 'error', message: err.message || 'Failed to load conversations.' })
        }
      }
    }
    void bootstrapThreads()
    return () => { cancelled = true }
  }, [user?.id, threadRetry])

  useEffect(() => {
    if (appView !== 'chat') return
    if (!requestedThreadId) {
      setActiveThreadId(null)
      return
    }
    if (threads.some((thread) => thread.id === requestedThreadId)) {
      setActiveThreadId(requestedThreadId)
    }
  }, [appView, requestedThreadId, threads])

  useEffect(() => {
    if (!activeThreadId) return undefined
    const targetThread = threads.find((thread) => thread.id === activeThreadId)
    if (targetThread?.messagesLoaded) return undefined
    let cancelled = false
    async function hydrate() {
      try {
        const result = await getThreadMessages(activeThreadId)
        if (cancelled) return
        setThreads((prev) => prev.map((thread) => thread.id === activeThreadId
          ? { ...thread, messages: (result.messages || []).map((message, index) => normalizeMessage(message, index)), messagesLoaded: true }
          : thread))
      } catch (err) {
        if (!cancelled) onToast?.({ type: 'error', message: err.message || 'Failed to load this conversation.' })
      }
    }
    void hydrate()
    return () => { cancelled = true }
  }, [activeThreadId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    handleChatScroll()
  }, [messages, loading])

  async function refreshIndexedSources() {
    setSourcesLoading(true)
    try {
      const result = await getIndexedSources()
      const sources = (result.sources || []).map(repairSourceMetadata)
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

  useEffect(() => {
    function handleClickOutside(event) {
      if (sourceDropdownRef.current && !sourceDropdownRef.current.contains(event.target)) {
        setSourceMenuOpen(false)
      }
    }
    if (sourceMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [sourceMenuOpen])

  useEffect(() => {
    if (!threadMenu) return undefined
    function closeThreadMenu(event) {
      if (!threadMenuRef.current?.contains(event.target)) setThreadMenu(null)
    }
    function closeOnEscape(event) {
      if (event.key === 'Escape') setThreadMenu(null)
    }
    document.addEventListener('mousedown', closeThreadMenu)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('mousedown', closeThreadMenu)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [threadMenu])

  useEffect(() => {
    if (!renameDialog) return undefined
    const timer = window.setTimeout(() => renameInputRef.current?.select(), 0)
    function closeOnEscape(event) {
      if (event.key === 'Escape' && !threadActionPending) setRenameDialog(null)
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      window.clearTimeout(timer)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [renameDialog, threadActionPending])

  useEffect(() => {
    if (!deleteDialog) return undefined
    function closeOnEscape(event) {
      if (event.key === 'Escape' && !threadActionPending) setDeleteDialog(null)
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [deleteDialog, threadActionPending])

  function getSourceIcon(type) {
    const t = String(type || '').toLowerCase()
    if (t.includes('web') || t.includes('url') || t.includes('http')) {
      return <Globe size={13} />
    }
    if (t.includes('drive') || t.includes('google')) {
      return <Cloud size={13} />
    }
    return <FileText size={13} />
  }

  function addGuidedIndexMessage(userText, threadId) {
    if (!threadId) return
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
    updateThreadById(threadId, (thread) => ({
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
    if (!activeThreadId) return
    updateThreadById(activeThreadId, updater)
  }

  function updateThreadById(threadId, updater) {
    if (!threadId) return
    setThreads((prev) => prev.map((thread) => {
      if (thread.id !== threadId) {
        return thread
      }
      return updater(thread)
    }))
  }

  function updateMessageById(messageId, updater, threadId = activeThreadId) {
    updateThreadById(threadId, (thread) => ({
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

    let targetThread = activeThread
    let targetThreadId = activeThread?.id || null

    if (!targetThreadId) {
      targetThreadId = makeMessageId('thread')
      const title = makeThreadTitleFromMessage(text)
      targetThread = {
        id: targetThreadId,
        sessionId: targetThreadId,
        title,
        messages: [],
        messagesLoaded: true,
        updatedAt: Date.now(),
        pending: true,
      }
      setThreads((prev) => [targetThread, ...prev])
      setActiveThreadId(targetThreadId)
      navigate(`/chat?thread=${encodeURIComponent(targetThreadId)}`)

      try {
        const result = await createBackendThread(title, targetThreadId)
        updateThreadById(targetThreadId, (thread) => ({
          ...thread,
          ...(result?.thread || {}),
          id: targetThreadId,
          sessionId: targetThreadId,
          messages: thread.messages,
          messagesLoaded: true,
          pending: false,
        }))
      } catch (createError) {
        setThreads((prev) => prev.filter((thread) => thread.id !== targetThreadId))
        setActiveThreadId(null)
        navigate('/chat', { replace: true })
        onToast?.({ type: 'error', message: createError.message || 'Failed to create a new chat.' })
        return
      }
    }

    const greetingOnly = isSimpleGreeting(text)

    if (!greetingOnly && indexedSources.length === 0) {
      addGuidedIndexMessage(text, targetThreadId)
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
    const requestId = window.crypto?.randomUUID?.() || `request-${Date.now()}-${Math.random().toString(36).slice(2)}`
    const assistantMessage = {
      id: assistantMessageId,
      role: 'assistant',
      text: '',
      ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      sources: [],
      streaming: true,
      requestId,
      activity: greetingOnly ? null : { stage: 'searching', sourceCount: 0 },
    }

    updateThreadById(targetThreadId, (thread) => ({
      ...thread,
      title: thread.messages.some((msg) => msg.role === 'user') ? thread.title : makeThreadTitleFromMessage(text),
      updatedAt: Date.now(),
      messages: [...thread.messages, userMessage, assistantMessage],
    }))
    setInput('')
    setLoading(true)
    setError('')
    const controller = new AbortController()
    let requestTimedOut = false
    const requestTimeout = window.setTimeout(() => {
      requestTimedOut = true
      controller.abort()
    }, 120_000)
    abortRef.current = controller

    const requestOptions = {
      threadId: targetThreadId,
      requestId,
      topK: Math.max(1, Math.min(20, Number(localStorage.getItem('ragkno_top_k') || 5))),
      useReranker: localStorage.getItem('ragkno_reranker') !== 'false',
      language: localStorage.getItem('ragkno_lang') || 'auto',
      sourceIds: [...selectedSourceKeys],
      signal: controller.signal,
    }

    const revealQueue = []
    const revealWaiters = []
    let revealRunning = false

    const finishRevealWaiters = () => {
      while (revealWaiters.length > 0) revealWaiters.shift()()
    }

    const drainRevealQueue = async () => {
      if (revealRunning) return
      revealRunning = true
      while (revealQueue.length > 0 && !controller.signal.aborted) {
        const nextPart = revealQueue.shift()
        updateMessageById(assistantMessageId, (message) => ({
          ...message,
          text: `${message.text}${nextPart}`,
        }), targetThreadId)
        const delay = revealQueue.length > 90 ? 7 : revealQueue.length > 40 ? 11 : 18
        await new Promise((resolve) => window.setTimeout(resolve, delay))
      }
      revealRunning = false
      if (revealQueue.length === 0 || controller.signal.aborted) finishRevealWaiters()
    }

    const enqueueResponseText = (value) => {
      const parts = String(value || '').match(/\S+\s*|\s+/g) || []
      if (parts.length === 0) return
      revealQueue.push(...parts)
      void drainRevealQueue()
    }

    const appendStreamedText = (value) => {
      const part = String(value || '')
      if (!part) return
      updateMessageById(assistantMessageId, (message) => ({
        ...message,
        text: `${message.text}${part}`,
        activity: null,
      }), targetThreadId)
    }

    const waitForResponseReveal = () => {
      if (!revealRunning && revealQueue.length === 0) return Promise.resolve()
      return new Promise((resolve) => revealWaiters.push(resolve))
    }

    try {
      const shouldStream = localStorage.getItem('ragkno_streaming') !== 'false'
      const streamDonePayload = shouldStream
        ? await queryRAGStream(text, requestOptions, {
          signal: controller.signal,
          onMeta: (meta) => {
            const metaSources = Array.isArray(meta?.sources) ? meta.sources.map(repairSourceMetadata) : []
            updateMessageById(assistantMessageId, (message) => ({
              ...message,
              sources: metaSources,
              activity: meta?.local ? null : { stage: 'writing', sourceCount: metaSources.length },
            }), targetThreadId)
          },
          onToken: (token) => {
            appendStreamedText(token)
          },
        })
        : await queryRAG(text, requestOptions)

      if (!shouldStream) {
        enqueueResponseText(streamDonePayload?.answer)
        await waitForResponseReveal()
      }
      if (controller.signal.aborted) {
        const abortError = new Error('Response cancelled.')
        abortError.name = 'AbortError'
        throw abortError
      }

      updateMessageById(assistantMessageId, (message) => ({
        ...message,
        text: message.text || streamDonePayload?.answer || 'No relevant documents found.',
        sources: Array.isArray(streamDonePayload?.sources) && streamDonePayload.sources.length > 0
          ? streamDonePayload.sources.map(repairSourceMetadata)
          : message.sources,
        activity: null,
        streaming: false,
      }), targetThreadId)
    } catch (streamError) {
      const aborted = streamError?.name === 'AbortError'
      const timeoutMessage = 'The model took too long to respond. Please try again.'
      updateMessageById(assistantMessageId, (message) => ({
        ...message,
        streaming: false,
        interrupted: true,
        activity: message.activity ? { ...message.activity, stage: 'failed' } : null,
        text: message.text || (requestTimedOut ? timeoutMessage : aborted ? 'Response cancelled.' : 'The response failed before it completed.'),
      }), targetThreadId)
      setError(requestTimedOut ? timeoutMessage : aborted ? '' : streamError.message)
      if (requestTimedOut) onToast?.({ type: 'error', message: timeoutMessage })
      else if (!aborted) onToast?.({ type: 'error', message: streamError.message || 'Response interrupted.' })
    } finally {
      window.clearTimeout(requestTimeout)
      abortRef.current = null
      setLoading(false)
    }
  }

  function startNewChat() {
    abortRef.current?.abort()
    setActiveThreadId(null)
    setError('')
    setInput('')
    setSourceMenuOpen(false)
    setSourceModalOpen(false)
    setCompactSidebarOpen(false)
    navigate('/chat')
  }

  function openThread(threadId) {
    if (!threadId) return
    setActiveThreadId(threadId)
    setError('')
    setCompactSidebarOpen(false)
    navigate(`/chat?thread=${encodeURIComponent(threadId)}`)
  }

  function navigateFromSidebar(path) {
    setCompactSidebarOpen(false)
    navigate(path)
  }

  function collapseOrCloseSidebar() {
    if (window.matchMedia('(max-width: 920px)').matches) {
      setCompactSidebarOpen(false)
      return
    }
    setSidebarCollapsed(true)
  }

  function openThreadMenu(event, thread) {
    event.stopPropagation()
    if (threadMenu?.id === thread.id) {
      setThreadMenu(null)
      return
    }
    const rect = event.currentTarget.getBoundingClientRect()
    const menuHeight = 92
    const top = rect.bottom + menuHeight + 8 > window.innerHeight
      ? Math.max(8, rect.top - menuHeight - 4)
      : rect.bottom + 4
    setThreadMenu({
      id: thread.id,
      title: thread.title || 'New Chat',
      top,
      left: Math.max(8, rect.right - 156),
    })
  }

  async function submitThreadRename(event) {
    event.preventDefault()
    const title = String(renameDialog?.title || '').trim()
    if (!renameDialog?.id || !title || threadActionPending) return
    setThreadActionPending(true)
    try {
      await renameBackendThread(renameDialog.id, title)
      setThreads((prev) => prev.map((thread) => thread.id === renameDialog.id
        ? { ...thread, title, updatedAt: Date.now() }
        : thread))
      setRenameDialog(null)
    } catch (err) {
      onToast?.({ type: 'error', message: err.message || 'Failed to rename conversation.' })
    } finally {
      setThreadActionPending(false)
    }
  }

  async function confirmThreadDelete() {
    if (!deleteDialog?.id || threadActionPending) return
    setThreadActionPending(true)
    try {
      const deleted = await deleteThread(deleteDialog.id)
      if (deleted) setDeleteDialog(null)
    } finally {
      setThreadActionPending(false)
    }
  }

  async function resetCurrentChatMemory() {
    if (!activeThread?.id) return
    await deleteThread(activeThread.id)
    startNewChat()
  }

  async function deleteThread(threadId) {
    const target = threads.find((thread) => thread.id === threadId)
    if (!target) {
      return false
    }

    try {
      await deleteBackendThread(threadId)
      const remaining = threads.filter((thread) => thread.id !== threadId)
      setThreads(remaining)
      if (activeThreadId === threadId) {
        setActiveThreadId(null)
        navigate('/chat', { replace: true })
      }
      return true
    } catch (err) {
      onToast?.({ type: 'error', message: err.message || 'Failed to delete conversation.' })
      return false
    }
  }

  async function clearAllHistory() {
    try {
      await Promise.all(threads.map((thread) => deleteBackendThread(thread.id)))
      setThreads([])
      setActiveThreadId(null)
      startNewChat()
      onToast?.({ type: 'info', message: t('historyCleared') || 'Chat history cleared.' })
    } catch (err) {
      onToast?.({ type: 'error', message: err.message || 'Failed to clear conversation history.' })
    }
  }

  return (
    <section className={`chat-page ${sidebarCollapsed ? 'collapsed' : ''}`}>
      <button
        className={`compact-sidebar-open-btn ${compactSidebarOpen ? 'is-hidden' : ''}`}
        type="button"
        onClick={() => {
          setSidebarCollapsed(false)
          setCompactSidebarOpen(true)
        }}
        aria-label="Open sidebar"
        aria-expanded={compactSidebarOpen}
        aria-controls="chat-navigation-sidebar"
        title="Open sidebar"
      >
        <HugeiconsIcon icon={LayoutAlignRightIcon} size={20} />
      </button>

      {compactSidebarOpen && (
        <button
          className="compact-sidebar-backdrop"
          type="button"
          onClick={() => setCompactSidebarOpen(false)}
          aria-label="Close sidebar"
          tabIndex={-1}
        />
      )}

      {sidebarCollapsed && (
        <aside
          className="chat-collapsed-rail"
          onClick={(e) => {
            if (e.target === e.currentTarget || e.target.classList.contains('rail-spacer')) {
              setSidebarCollapsed(false)
            }
          }}
        >
          <button
            className="rail-logo"
            type="button"
            onClick={() => setSidebarCollapsed(false)}
            aria-label="Open sidebar"
            title="Open sidebar"
          >
            <img src={brandLogo} alt="RagKno logo" className="rail-logo-img" />
            <HugeiconsIcon icon={LayoutAlignRightIcon} size={18} className="rail-open-icon" />
          </button>

          <button
            className="rail-icon-btn"
            type="button"
            onClick={startNewChat}
            aria-label={t('newChat') || 'New chat'}
            title={t('newChat') || 'New chat'}
          >
            <SquarePen size={17} />
          </button>

          <button
            className="rail-icon-btn"
            type="button"
            onClick={() => onToast('Search chats')}
            aria-label="Search chats"
            title="Search chats"
          >
            <Search size={17} />
          </button>

          <button
            className={`rail-icon-btn ${appView === 'chat' ? 'active' : ''}`}
            type="button"
            onClick={() => navigate('/chat')}
            aria-label="Chats"
            title="Chats"
          >
            <MessageSquare size={17} />
          </button>

          <button
            className={`rail-icon-btn ${appView === 'data' ? 'active' : ''}`}
            type="button"
            onClick={() => navigate('/chat/data')}
            aria-label="Data Center"
            title="Data Center"
          >
            <Database size={17} />
          </button>

          <button
            className={`rail-icon-btn ${appView === 'apps' ? 'active' : ''}`}
            type="button"
            onClick={() => navigate('/chat/apps')}
            aria-label={t('connectApps') || 'Connect Apps'}
            title={t('connectApps') || 'Connect Apps'}
          >
            <Cloud size={17} />
          </button>

          <div
            className="rail-spacer"
            onClick={() => setSidebarCollapsed(false)}
            title="Click to open sidebar"
          />

          <div className="rail-profile-wrap">
            {profileMenuOpen && (
              <div className="profile-dropup rail-profile-dropup">
                <p>{user?.email || 'Signed in'}</p>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openSettingsModal('general'); }}><Settings size={15} /> {t('settings')} <span>⇧⌘,</span></button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openSettingsModal('help'); }}><CircleHelp size={15} /> {t('helpFaq')}</button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openSettingsModal('about'); }}><Info size={15} /> {t('learnMore')} <ChevronRight size={14} /></button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); setSettingsModalOpen(false); setFeedbackModalOpen(true); }}><MessageSquareHeart size={15} /> {t('giveFeedback')}</button>
                <hr />
                <button type="button" onClick={handleLogout}><LogOut size={15} /> {t('logOut')}</button>
                <div className="profile-product-version" aria-label="RagKno version 1.0">
                  <span>RagKno</span>
                  <strong>v1.0</strong>
                </div>
              </div>
            )}
            <button
              className="rail-profile-btn"
              type="button"
              onClick={() => setProfileMenuOpen((open) => !open)}
              aria-label="Account menu"
              title={displayName}
              aria-expanded={profileMenuOpen}
            >
              <UserAvatar user={user} displayName={displayName} />
            </button>
          </div>
        </aside>
      )}

      {!sidebarCollapsed && (
        <aside id="chat-navigation-sidebar" className={`chat-sidebar ${compactSidebarOpen ? 'compact-open' : ''}`}>
          <div className="chat-sidebar-header">
            <Link to="/chat" className="chat-sidebar-brand" onClick={() => setCompactSidebarOpen(false)}>
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
                onClick={collapseOrCloseSidebar}
                aria-label="Collapse sidebar"
                title="Collapse sidebar"
              >
                <HugeiconsIcon icon={LayoutAlignLeftIcon} size={18} />
              </button>
            </div>
          </div>

          <button className="history-btn new-chat-pill" onClick={startNewChat}>
            <SquarePen size={14} />
            <span>{t('newChat') || 'New chat'}</span>
          </button>

          <div className="sidebar-nav-group">
            <button className={`history-btn ${appView === 'chat' ? 'active' : ''}`} onClick={() => navigateFromSidebar('/chat')}>
              <MessageSquare size={14} /> <span>Chats</span>
            </button>
            <button className={`history-btn ${appView === 'data' ? 'active' : ''}`} onClick={() => navigateFromSidebar('/chat/data')}>
              <Database size={14} /> <span>Data Center</span>
            </button>
            <button className={`history-btn ${appView === 'apps' ? 'active' : ''}`} onClick={() => navigateFromSidebar('/chat/apps')}>
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
                  onClick={() => openThread(thread.id)}
                  title={thread.title || 'New Chat'}
                >
                  <span className="history-title-text">{thread.title || 'New Chat'}</span>
                </button>
                <button
                  className="history-options-btn"
                  aria-label={`Chat options for ${thread.title || 'New Chat'}`}
                  aria-haspopup="menu"
                  aria-expanded={threadMenu?.id === thread.id}
                  onClick={(event) => openThreadMenu(event, thread)}
                >
                  <MoreHorizontal size={16} />
                </button>
              </div>
            ))}
          </div>

          <div className="chat-sidebar-footer">
            {profileMenuOpen && (
              <div className="profile-dropup">
                <p>{user?.email || 'Signed in'}</p>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openSettingsModal('general'); }}><Settings size={15} /> {t('settings')} <span>⇧⌘,</span></button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openSettingsModal('help'); }}><CircleHelp size={15} /> {t('helpFaq')}</button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); openSettingsModal('about'); }}><Info size={15} /> {t('learnMore')} <ChevronRight size={14} /></button>
                <button type="button" onClick={() => { setProfileMenuOpen(false); setSettingsModalOpen(false); setFeedbackModalOpen(true); }}><MessageSquareHeart size={15} /> {t('giveFeedback')}</button>
                <hr />
                <button type="button" onClick={handleLogout}><LogOut size={15} /> {t('logOut')}</button>
                <div className="profile-product-version" aria-label="RagKno version 1.0">
                  <span>RagKno</span>
                  <strong>v1.0</strong>
                </div>
              </div>
            )}
            <button className="user-pill" type="button" onClick={() => setProfileMenuOpen((open) => !open)} aria-expanded={profileMenuOpen}>
              <div className="user-avatar-badge">
                <UserAvatar user={user} displayName={displayName} />
              </div>
              <div className="user-info-text">
                <strong>{displayName}</strong>
              </div>
              <ChevronsUpDown size={13} className="user-pill-chevron" />
            </button>
          </div>
        </aside>
      )}

      {threadMenu && (
        <div
          ref={threadMenuRef}
          className="thread-options-menu"
          role="menu"
          aria-label={`Options for ${threadMenu.title}`}
          style={{ top: threadMenu.top, left: threadMenu.left }}
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setRenameDialog({ id: threadMenu.id, title: threadMenu.title })
              setThreadMenu(null)
            }}
          >
            <Pencil size={16} /> Rename
          </button>
          <button
            type="button"
            role="menuitem"
            className="danger"
            onClick={() => {
              setDeleteDialog({ id: threadMenu.id, title: threadMenu.title })
              setThreadMenu(null)
            }}
          >
            <Trash2 size={16} /> Delete
          </button>
        </div>
      )}

      {renameDialog && (
        <div className="thread-dialog-backdrop" role="presentation" onMouseDown={() => !threadActionPending && setRenameDialog(null)}>
          <form className="thread-dialog" role="dialog" aria-modal="true" aria-labelledby="rename-chat-title" onSubmit={submitThreadRename} onMouseDown={(event) => event.stopPropagation()}>
            <button className="thread-dialog-close" type="button" aria-label="Close rename dialog" disabled={threadActionPending} onClick={() => setRenameDialog(null)}><X size={19} /></button>
            <h2 id="rename-chat-title">Rename chat</h2>
            <p>Keep it short and recognizable.</p>
            <input
              ref={renameInputRef}
              value={renameDialog.title}
              maxLength={120}
              aria-label="Chat name"
              onChange={(event) => setRenameDialog((current) => ({ ...current, title: event.target.value }))}
            />
            <div className="thread-dialog-actions">
              <button type="button" className="secondary" disabled={threadActionPending} onClick={() => setRenameDialog(null)}>Cancel</button>
              <button type="submit" className="primary" disabled={threadActionPending || !renameDialog.title.trim()}>{threadActionPending ? 'Saving…' : 'Save'}</button>
            </div>
          </form>
        </div>
      )}

      {deleteDialog && (
        <div className="thread-dialog-backdrop" role="presentation" onMouseDown={() => !threadActionPending && setDeleteDialog(null)}>
          <div className="thread-dialog delete-thread-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-chat-title" onMouseDown={(event) => event.stopPropagation()}>
            <button className="thread-dialog-close" type="button" aria-label="Close delete dialog" disabled={threadActionPending} onClick={() => setDeleteDialog(null)}><X size={19} /></button>
            <h2 id="delete-chat-title">Delete chat?</h2>
            <p>This will permanently delete <strong>{deleteDialog.title}</strong>. This can’t be undone.</p>
            <div className="thread-dialog-actions">
              <button type="button" className="secondary" disabled={threadActionPending} onClick={() => setDeleteDialog(null)}>Cancel</button>
              <button type="button" className="danger" disabled={threadActionPending} onClick={() => void confirmThreadDelete()}>{threadActionPending ? 'Deleting…' : 'Delete chat'}</button>
            </div>
          </div>
        </div>
      )}

      <section className={`chat-canvas ${appView !== 'chat' ? 'utility-view' : ''} ${appView === 'chat' && messages.length === 0 ? 'is-empty' : ''}`}>

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

        {appView === 'chat' && threadStartup.status === 'loading' && (
          <div className="chat-startup-quiet" role="status" aria-label="Loading conversations" aria-live="polite" />
        )}

        {appView === 'chat' && threadStartup.status === 'error' && (
          <div className="chat-startup-state chat-startup-error" role="alert"><div><h1>We couldn’t load your chats.</h1><p>{threadStartup.error}</p><button type="button" onClick={() => setThreadRetry((value) => value + 1)}><RefreshCw size={16} /> Try again</button></div></div>
        )}

        {appView === 'chat' && threadStartup.status === 'ready' && (
          <>
            <div className="chat-scroll-container" ref={chatScrollRef} onScroll={handleChatScroll}>
              <div className="message-stack">
                {!messages.length && !loading && (
                  <div className="chat-empty-state">
                    <h1>How can I help, {firstName}?</h1>
                  </div>
                )}

                {messages.map((message) => (
                  <div key={message.id} className={`message ${message.role}`}>
                    {message.role === 'assistant' && message.activity && (
                      <RetrievalActivity activity={message.activity} />
                    )}
                    {message.role === 'assistant' && !message.activity && Array.isArray(message.sources) && message.sources.length > 0 && (
                      <div className="message-sources">
                        <div
                          className="source-trigger-wrap"
                          onMouseEnter={() => setHoveredSourceMessage(message.id)}
                          onMouseLeave={() => setHoveredSourceMessage(null)}
                        >
                          <button
                            type="button"
                            className={`source-trigger ${expandedSourceMessages.has(message.id) ? 'open' : ''}`}
                            onClick={() => toggleMessageSources(message.id)}
                            aria-expanded={expandedSourceMessages.has(message.id)}
                          >
                            <Database size={14} aria-hidden="true" />
                            <span>Used {message.sources.length} {message.sources.length === 1 ? 'source' : 'sources'}</span>
                            <ChevronDown size={14} className="source-trigger-chevron" aria-hidden="true" />
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
                        </div>

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

                    <div className="bubble">
                      {message.role === 'assistant' ? renderAssistantText(message.text, message) : message.text}
                      {message.action?.to && (
                        <Link className="message-action-link" to={message.action.to}>
                          {message.action.label || 'Open'}
                        </Link>
                      )}
                      {message.role === 'assistant' && message.streaming && !message.activity && !message.text && <span className="typing-cursor" aria-hidden="true" />}
                      {message.role === 'assistant' && message.interrupted && (
                        <span className="response-interrupted" role="status">Response interrupted</span>
                      )}
                    </div>

                    <span>{message.role === 'user' ? message.ts : `Ragkno • ${message.ts}`}</span>
                  </div>
                ))}

                {error && <p className="notice">{error}</p>}
                <div ref={bottomRef} />
              </div>
            </div>

            <div className="chat-composer-container">
              {showScrollBottom && (
                <button
                  type="button"
                  className="scroll-to-bottom-btn"
                  onClick={scrollToBottom}
                  aria-label="Scroll to bottom"
                  title="Scroll to bottom"
                >
                  <ArrowDown size={15} strokeWidth={2.2} />
                </button>
              )}
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
                  <div className="source-dropdown" ref={sourceDropdownRef}>
                    <button
                      type="button"
                      className="source-dropdown-btn"
                      onClick={() => {
                        setSourceMenuOpen((prev) => !prev)
                        if (!sourceMenuOpen) void refreshIndexedSources()
                      }}
                      title="Sources selector"
                      aria-expanded={sourceMenuOpen}
                    >
                      <Database size={13} />
                      <span>{t('sources') || 'Sources'}</span>
                      <span className={`source-count-badge ${selectedSourceKeys.size > 0 ? 'active' : ''}`}>
                        {selectedSourceKeys.size > 0 ? selectedSourceKeys.size : indexedSources.length}
                      </span>
                      <ChevronDown size={13} className={`source-chevron ${sourceMenuOpen ? 'open' : ''}`} />
                    </button>
                    {sourceMenuOpen && (
                      <div className="source-dropdown-menu">
                        <div className="source-dropdown-head">
                          <div>
                            <strong>Sources</strong>
                            <small>
                              {selectedSourceKeys.size > 0
                                ? `${selectedSourceKeys.size} of ${indexedSources.length} selected`
                                : `${indexedSources.length} available`}
                            </small>
                          </div>
                          <div className="source-head-actions">
                            {indexedSources.length > 0 && (
                              <button
                                type="button"
                                className="source-action-link"
                                onClick={() => {
                                  if (selectedSourceKeys.size === indexedSources.length) {
                                    setSelectedSourceKeys(new Set())
                                  } else {
                                    setSelectedSourceKeys(new Set(indexedSources.map((s) => s.key)))
                                  }
                                }}
                              >
                                {selectedSourceKeys.size === indexedSources.length ? 'Clear' : 'Select all'}
                              </button>
                            )}
                            <button
                              type="button"
                              className="source-refresh-btn"
                              onClick={refreshIndexedSources}
                              disabled={sourcesLoading}
                              title="Refresh sources"
                            >
                              <RefreshCw size={12} className={sourcesLoading ? 'spin' : ''} />
                            </button>
                          </div>
                        </div>

                        <div className="source-dropdown-list">
                          {indexedSources.length === 0 && (
                            <p className="source-empty">No sources indexed yet.</p>
                          )}
                          {indexedSources.map((source) => {
                            const isSelected = selectedSourceKeys.has(source.key)
                            return (
                              <button
                                key={source.key}
                                type="button"
                                className={`source-option-btn ${isSelected ? 'selected' : ''}`}
                                onClick={() => toggleSourceSelection(source.key)}
                              >
                                <span className="source-option-icon">
                                  {getSourceIcon(source.type)}
                                </span>
                                <span className="source-option-text">
                                  <strong className="source-option-title">{source.source}</strong>
                                  <small className="source-option-type">{source.type}</small>
                                </span>
                                {isSelected && (
                                  <Check size={15} className="source-option-check" />
                                )}
                              </button>
                            )
                          })}
                        </div>

                        <div className="source-dropdown-actions">
                          <button
                            type="button"
                            className="source-footer-btn"
                            onClick={() => {
                              setSourceMenuOpen(false)
                              navigate('/chat/data')
                            }}
                          >
                            Data Center
                          </button>
                          <button
                            type="button"
                            className="source-footer-btn danger"
                            onClick={unindexSelectedSources}
                            disabled={selectedSourceKeys.size === 0}
                          >
                            <Trash2 size={12} /> Unindex {selectedSourceKeys.size > 0 ? `(${selectedSourceKeys.size})` : ''}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  <button
                    type={loading ? 'button' : 'submit'}
                    className={`send-btn ${loading ? 'is-stop' : ''}`}
                    disabled={!loading && !input.trim()}
                    onClick={loading ? () => abortRef.current?.abort() : undefined}
                    title={loading ? 'Stop response' : 'Send message'}
                    aria-label={loading ? 'Stop response' : 'Send message'}
                  >
                    {loading ? <span className="stop-square" /> : <ArrowUp size={16} strokeWidth={2.5} />}
                  </button>
                </div>
              </form>

              <p className="chat-disclaimer">{t('disclaimer') || 'Ragkno can make mistakes. verify important info.'}</p>
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

      <SettingsModal
        isOpen={settingsModalOpen}
        onClose={() => setSettingsModalOpen(false)}
        initialTab={settingsModalTab}
        user={user}
        conversationData={threads}
        onClearHistory={() => { void clearAllHistory() }}
        onToast={onToast}
        onNavigateData={() => navigate('/chat/data')}
        onSampleQuestion={(q) => {
          setInput(q)
          navigate('/chat')
        }}
        onLogout={handleLogout}
        onOpenFeedback={() => {
          setSettingsModalOpen(false)
          setFeedbackModalOpen(true)
        }}
      />

      <FeedbackModal
        isOpen={feedbackModalOpen}
        onClose={() => setFeedbackModalOpen(false)}
        onSubmitFeedback={async (feedbackData) => {
          try {
            await submitFeedback({
              rating: feedbackData?.rating || 'neutral',
              feedback: feedbackData?.feedback || feedbackData?.comment || '',
              comment: feedbackData?.comment || feedbackData?.feedback || '',
              user_id: user?.id,
              user_email: user?.email,
            })
            onToast?.({ type: 'success', message: 'Thank you for your feedback!' })
          } catch (err) {
            onToast?.({ type: 'error', message: err?.message || 'Failed to submit feedback. Please try again.' })
          }
        }}
      />
    </section>
  )
}

function ScrollToTop() {
  const { pathname, hash } = useLocation()

  useEffect(() => {
    let firstFrame = 0
    let secondFrame = 0

    if (hash) {
      firstFrame = window.requestAnimationFrame(() => {
        secondFrame = window.requestAnimationFrame(() => {
          const target = document.getElementById(decodeURIComponent(hash.slice(1)))
          target?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        })
      })
    } else {
      window.scrollTo(0, 0)
    }

    return () => {
      window.cancelAnimationFrame(firstFrame)
      window.cancelAnimationFrame(secondFrame)
    }
  }, [pathname, hash])

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
      allowNestedScroll: true,
    })

    let frame = 0
    function raf(time) {
      lenis.raf(time)
      frame = window.requestAnimationFrame(raf)
    }

    frame = window.requestAnimationFrame(raf)
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
            element={user ? <Navigate to="/chat" replace /> : <AuthPage onUserChange={setUser} onToast={pushToast} />}
          />
          <Route path="/privacy" element={<LegalPage kind="privacy" />} />
          <Route path="/docs" element={<DocsPage />} />
          <Route path="/terms" element={<LegalPage kind="terms" />} />
          <Route path="/cookies" element={<LegalPage kind="cookies" />} />
          <Route path="/data" element={<Navigate to="/chat/data" replace />} />
          <Route
            path="/chat/*"
            element={(
              <ProtectedRoute user={user} checkingAuth={checkingAuth}>
                <ChatErrorBoundary onRecover={() => window.location.assign('/chat')}>
                  <ChatPage user={user} onUserChange={setUser} onToast={pushToast} />
                </ChatErrorBoundary>
              </ProtectedRoute>
            )}
          />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}
