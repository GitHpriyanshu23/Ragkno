import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  BookOpen,
  Check,
  ChevronRight,
  Cloud,
  Code2,
  Copy,
  Database,
  ExternalLink,
  FileText,
  Menu,
  MessageSquare,
  Search,
  Settings,
  ShieldCheck,
  Upload,
  X,
} from 'lucide-react'
import SiteFooter from './SiteFooter.jsx'

const DOC_GROUPS = [
  {
    label: 'Overview',
    items: [
      ['introduction', 'Introduction'],
      ['how-ragkno-works', 'How RagKno works'],
      ['quickstart', 'Quickstart'],
    ],
  },
  {
    label: 'Knowledge sources',
    items: [
      ['upload-documents', 'Upload documents'],
      ['add-a-website', 'Add a website'],
      ['google-drive', 'Google Drive'],
    ],
  },
  {
    label: 'Using RagKno',
    items: [
      ['ask-questions', 'Ask questions'],
      ['sources-and-citations', 'Sources and citations'],
      ['chat-management', 'Manage chats'],
    ],
  },
  {
    label: 'Self-hosting',
    items: [
      ['local-setup', 'Local setup'],
      ['environment', 'Environment variables'],
      ['api-reference', 'API reference'],
    ],
  },
  {
    label: 'Help',
    items: [
      ['troubleshooting', 'Troubleshooting'],
      ['privacy-and-security', 'Privacy and security'],
    ],
  },
  {
    label: 'Contributing',
    items: [
      ['contributing', 'Contributing'],
      ['code-of-conduct', 'Code of Conduct'],
    ],
  },
]

const ON_THIS_PAGE = [
  ['how-ragkno-works', 'How RagKno works'],
  ['quickstart', 'Get started'],
  ['supported-content', 'Supported content'],
  ['sources-and-citations', 'Grounded answers'],
  ['self-hosting', 'Self-hosting'],
  ['contributing', 'Contributing'],
  ['code-of-conduct', 'Code of Conduct'],
  ['more-resources', 'More resources'],
]

function DocsSidebar({ query, onQueryChange, mobileOpen, onClose, searchRef }) {
  const filteredGroups = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return DOC_GROUPS
    return DOC_GROUPS
      .map((group) => ({
        ...group,
        items: group.items.filter(([, title]) => title.toLowerCase().includes(normalized)),
      }))
      .filter((group) => group.items.length)
  }, [query])

  return (
    <aside
      className={`docs-sidebar ${mobileOpen ? 'is-open' : ''}`}
      data-lenis-prevent
      aria-label="Documentation navigation"
    >
      <div className="docs-sidebar-mobile-head">
        <strong>Documentation</strong>
        <button type="button" onClick={onClose} aria-label="Close documentation menu"><X size={18} /></button>
      </div>
      <label className="docs-search">
        <Search size={16} aria-hidden="true" />
        <input
          ref={searchRef}
          type="search"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search the docs..."
          aria-label="Search documentation"
        />
        <kbd>⌘K</kbd>
      </label>
      <nav className="docs-nav-groups">
        {filteredGroups.map((group) => (
          <div className="docs-nav-group" key={group.label}>
            <p>{group.label}</p>
            {group.items.map(([id, title]) => (
              <a key={id} href={`#${id}`} onClick={onClose}>{title}</a>
            ))}
          </div>
        ))}
        {filteredGroups.length === 0 && <p className="docs-search-empty">No matching section.</p>}
      </nav>
    </aside>
  )
}

function CodeBlock({ children }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard?.writeText(String(children))
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="docs-code-block">
      <button type="button" onClick={copy} aria-label="Copy code">
        {copied ? <Check size={15} /> : <Copy size={15} />}
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre><code>{children}</code></pre>
    </div>
  )
}

function Step({ number, title, children }) {
  return (
    <li className="docs-step">
      <span>{number}</span>
      <div><strong>{title}</strong><p>{children}</p></div>
    </li>
  )
}

function ResourceCard({ icon: Icon, title, children, to, external = false }) {
  const content = (
    <>
      <span className="docs-resource-icon"><Icon size={18} /></span>
      <span><strong>{title}</strong><small>{children}</small></span>
      {external ? <ExternalLink size={15} /> : <ChevronRight size={16} />}
    </>
  )
  return external
    ? <a className="docs-resource-card" href={to} target="_blank" rel="noopener noreferrer">{content}</a>
    : <Link className="docs-resource-card" to={to}>{content}</Link>
}

export default function DocsPage() {
  const [query, setQuery] = useState('')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [activeSection, setActiveSection] = useState(ON_THIS_PAGE[0][0])
  const searchRef = useRef(null)

  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        searchRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  useEffect(() => {
    const nodes = ON_THIS_PAGE
      .map(([id]) => document.getElementById(id))
      .filter(Boolean)
    if (!nodes.length || !('IntersectionObserver' in window)) return undefined
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.find((entry) => entry.isIntersecting)
      if (visible) setActiveSection(visible.target.id)
    }, { rootMargin: '-18% 0px -68% 0px' })
    nodes.forEach((node) => observer.observe(node))
    return () => observer.disconnect()
  }, [])

  return (
    <div className="docs-page">
      <div className="docs-mobile-bar">
        <button type="button" onClick={() => setMobileOpen(true)} aria-label="Open documentation menu">
          <Menu size={18} /> Menu
        </button>
        <span>RagKno documentation</span>
      </div>

      {mobileOpen && <button className="docs-sidebar-scrim" type="button" onClick={() => setMobileOpen(false)} aria-label="Close documentation menu" />}
      <div className="docs-layout">
        <DocsSidebar
          query={query}
          onQueryChange={setQuery}
          mobileOpen={mobileOpen}
          onClose={() => setMobileOpen(false)}
          searchRef={searchRef}
        />

        <article className="docs-content">
          <header id="introduction" className="docs-intro">
            <div className="docs-eyebrow"><BookOpen size={15} /> RagKno documentation</div>
            <h1>Introduction</h1>
            <p className="docs-lead">
              RagKno turns your documents, web pages, and connected Drive files into a searchable knowledge workspace. Ask a question in natural language and receive a streamed answer grounded in the sources you selected.
            </p>
            <div className="docs-intro-actions">
              <Link to="/login?mode=signup" className="docs-primary-action">Create an account <ArrowRight size={15} /></Link>
              <a href="https://github.com/GitHpriyanshu23/Ragkno" target="_blank" rel="noopener noreferrer" className="docs-secondary-action">View on GitHub <ExternalLink size={14} /></a>
            </div>
          </header>

          <section id="how-ragkno-works" className="docs-section">
            <h2>Turn scattered knowledge into grounded answers</h2>
            <p>RagKno keeps the retrieval process visible so you can understand how an answer was produced.</p>
            <ul className="docs-feature-list">
              <li><Upload size={18} /><span><strong>Bring your sources</strong> — Upload PDF, DOCX, or TXT documents, add a public web page, or connect supported Google Drive files.</span></li>
              <li><Database size={18} /><span><strong>Build a private index</strong> — Content is extracted, divided into meaningful chunks, embedded, and stored with user-scoped metadata.</span></li>
              <li><Search size={18} /><span><strong>Retrieve relevant evidence</strong> — Dense and keyword retrieval find useful passages before semantic reranking orders the results.</span></li>
              <li><MessageSquare size={18} /><span><strong>Generate a cited response</strong> — The selected language model writes from retrieved evidence and streams the answer into the conversation.</span></li>
            </ul>
          </section>

          <section id="quickstart" className="docs-section">
            <h2>Get started</h2>
            <p>You can create your first grounded conversation in a few steps.</p>
            <ol className="docs-steps">
              <Step number="1" title="Create your account">Sign up with email and password, or continue with Google.</Step>
              <Step number="2" title="Add knowledge">Open Data Center and upload a document, add a website URL, or connect Google Drive.</Step>
              <Step number="3" title="Choose your sources">Keep the sources you want available for the current question selected in the composer.</Step>
              <Step number="4" title="Ask a focused question">RagKno retrieves relevant passages, reranks the evidence, and streams a source-grounded response.</Step>
            </ol>
            <div className="docs-callout">
              <strong>Tip</strong>
              <p>Ask specific questions that match the language used in your documents. Include a company, topic, date range, or metric when it helps narrow the search.</p>
            </div>
          </section>

          <section id="supported-content" className="docs-section">
            <h2>Supported content</h2>
            <p>Each source is indexed separately and remains scoped to the account that added it.</p>
            <div className="docs-source-grid">
              <div id="upload-documents" className="docs-source-card"><FileText size={20} /><strong>Document uploads</strong><p>PDF, DOCX, and UTF-8 TXT files. Upload up to 20 files per request, with a 50 MB limit per file.</p></div>
              <div id="add-a-website" className="docs-source-card"><Code2 size={20} /><strong>Web pages</strong><p>Public HTTP or HTTPS pages containing HTML or plain text. Private network addresses and credential-bearing URLs are rejected.</p></div>
              <div id="google-drive" className="docs-source-card"><Cloud size={20} /><strong>Google Drive</strong><p>Import selected PDF, TXT, and Google Docs content through read-only Google Drive access.</p></div>
            </div>
          </section>

          <section id="ask-questions" className="docs-section">
            <h2>Ask questions</h2>
            <p>When you send a knowledge question, the activity panel moves through understanding, embedding, retrieval, reranking, and response generation. Greetings are handled locally and do not start the retrieval pipeline.</p>
            <p>Answers stream into the chat as they are generated. Markdown headings, lists, emphasis, links, and tables are rendered as formatted content.</p>
          </section>

          <section id="sources-and-citations" className="docs-section">
            <h2>Understand sources and citations</h2>
            <p>The <strong>Used sources</strong> control appears above a grounded response. Open it to inspect the evidence RagKno retrieved for that answer.</p>
            <ul className="docs-bullets">
              <li>Inline markers such as <strong>[1]</strong> refer to the corresponding retrieved source.</li>
              <li>The source panel shows document names and the relevant extracted passage.</li>
              <li>If the available evidence is incomplete, the answer should say so instead of inventing details.</li>
            </ul>
            <div className="docs-callout neutral">
              <strong>Verify critical information</strong>
              <p>Retrieval improves traceability, but generated answers can still be incomplete or incorrect. Open the cited source before relying on financial, legal, medical, or similarly important information.</p>
            </div>
          </section>

          <section id="chat-management" className="docs-section">
            <h2>Manage conversations</h2>
            <p>A chat is created in the sidebar after its first prompt. You can switch between conversations, rename them from the three-dot menu, or permanently delete a chat and its messages. Reloading a chat URL restores the selected thread.</p>
          </section>

          <section id="self-hosting" className="docs-section">
            <h2>Self-host RagKno</h2>
            <p>RagKno includes a React and Vite frontend, a FastAPI backend, PostgreSQL-compatible metadata storage, and a persistent Chroma vector index.</p>

            <h3 id="local-setup">Local setup</h3>
            <CodeBlock>{`# Backend\nuv sync --dev\nuv run python backend/main.py\n\n# Frontend\ncd frontend\nnpm install\nnpm run dev`}</CodeBlock>

            <h3 id="environment">Environment variables</h3>
            <p>Copy <code>.env.example</code> to <code>.env</code> for local development. Never commit the resulting file.</p>
            <CodeBlock>{`DATABASE_URL=postgresql://...\nLLM_PROVIDER=agentrouter\nAGENTROUTER_API_KEY=...\nAGENTROUTER_MODEL=deepseek-v4-flash\nGOOGLE_CLIENT_ID=...\nGOOGLE_CLIENT_SECRET=...\nRAGKNO_SESSION_SECRET=...\nFRONTEND_URL=http://localhost:5173`}</CodeBlock>
          </section>

          <section id="api-reference" className="docs-section">
            <h2>API reference</h2>
            <p>The browser client uses authenticated JSON endpoints and server-sent events. Mutation requests require the session CSRF token.</p>
            <div className="docs-endpoint-list">
              <div><code>GET</code><span>/health/ready</span><small>Database, vector store, and LLM configuration readiness</small></div>
              <div><code>POST</code><span>/ingest/files</span><small>Extract and index uploaded documents</small></div>
              <div><code>POST</code><span>/ingest/url</span><small>Extract and index a public web page</small></div>
              <div><code>POST</code><span>/query/stream</span><small>Retrieve evidence and stream a grounded response</small></div>
              <div><code>GET</code><span>/threads</span><small>List the authenticated user’s conversations</small></div>
            </div>
          </section>

          <section id="troubleshooting" className="docs-section">
            <h2>Troubleshooting</h2>
            <div className="docs-troubleshooting">
              <details><summary>The response says no relevant documents were found</summary><p>Confirm that a source is indexed and selected, then ask a question that uses terms present in the source. Re-index documents after changing chunking or embedding settings.</p></details>
              <details><summary>The model response is interrupted</summary><p>Check the AgentRouter key, enabled model names, provider capacity, and request timeout. The backend health endpoint confirms configuration, while server logs contain the provider error.</p></details>
              <details><summary>Google Drive does not connect</summary><p>Verify both OAuth redirect URIs, the consent screen test users, and the Drive read-only scope in Google Cloud Console.</p></details>
              <details><summary>The backend runs out of memory</summary><p>Use CPU devices for production embedding and reranking, keep one worker initially, and provide enough memory for both models and document processing.</p></details>
            </div>
          </section>

          <section id="privacy-and-security" className="docs-section">
            <h2>Privacy and security</h2>
            <p>Sessions use HTTP-only cookies, mutation routes validate origin and CSRF tokens, and retrieval queries filter vector data by user ID. URL ingestion blocks private and reserved network addresses.</p>
            <p>Self-hosters remain responsible for secret management, database and vector-store backups, token encryption, TLS, retention rules, and access controls.</p>
          </section>

          <section id="contributing" className="docs-section">
            <h2>Contributing</h2>
            <p>Thanks for helping improve RagKno. Contributions to the interface, retrieval pipeline, documentation, tests, accessibility, and deployment tooling are welcome.</p>

            <h3>Choose an issue</h3>
            <ul className="docs-bullets">
              <li>Search existing GitHub issues before opening a new one.</li>
              <li>For bugs, include reproduction steps, expected behavior, actual behavior, logs with secrets removed, and your operating system and browser.</li>
              <li>For a feature request, explain the user problem and desired outcome before proposing an implementation.</li>
            </ul>

            <h3>Prepare your branch</h3>
            <p>Fork the repository, update your local <code>main</code> branch, and create a short branch name that describes one focused change.</p>
            <CodeBlock>{`git checkout main\ngit pull origin main\ngit checkout -b feat/short-description`}</CodeBlock>

            <h3>Run RagKno locally</h3>
            <p>Install both workspaces and copy <code>.env.example</code> to an ignored local <code>.env</code>. Use placeholder or personal development credentials and never add secrets to a commit.</p>
            <CodeBlock>{`# Backend\nuv sync --dev\nuv run python backend/main.py\n\n# Frontend (in another terminal)\ncd frontend\nnpm install\nnpm run dev`}</CodeBlock>

            <h3>Validate your changes</h3>
            <p>Add focused tests when behavior changes, then run the same core checks used by continuous integration.</p>
            <CodeBlock>{`# From the repository root\nuv run pytest -q\n\n# From frontend/\nnpm test -- --run\nnpm run build`}</CodeBlock>

            <h3>Submit a pull request</h3>
            <ol className="docs-contribution-list">
              <li>Keep the pull request limited to one logical change.</li>
              <li>Explain the problem, the resulting behavior, and any important implementation decision.</li>
              <li>Include screenshots or a short recording for visible interface changes.</li>
              <li>List the tests you ran and link the related issue when one exists.</li>
              <li>Confirm that generated data, uploaded documents, databases, and credentials are absent from the diff.</li>
            </ol>

            <h3>License</h3>
            <p>By contributing, you agree that your contribution may be distributed under RagKno’s Apache License 2.0.</p>
          </section>

          <section id="code-of-conduct" className="docs-section">
            <h2>Code of Conduct</h2>
            <p>RagKno should be a respectful place to learn, build, review code, and exchange ideas. Everyone participating in the project is expected to follow these standards.</p>

            <h3>Expected behavior</h3>
            <ul className="docs-bullets">
              <li>Be considerate, patient, and specific when giving feedback.</li>
              <li>Discuss ideas and code without attacking the person who proposed them.</li>
              <li>Respect different backgrounds, experience levels, identities, and communication styles.</li>
              <li>Accept correction, acknowledge mistakes, and help repair their impact.</li>
              <li>Protect private information found in bug reports, logs, documents, and security reports.</li>
            </ul>

            <h3>Unacceptable behavior</h3>
            <ul className="docs-bullets">
              <li>Harassment, discrimination, threats, intimidation, or unwanted sexual attention.</li>
              <li>Personal insults, inflammatory comments, sustained disruption, or deliberate humiliation.</li>
              <li>Publishing another person’s private information without explicit permission.</li>
              <li>Using project spaces to promote spam, scams, malware, or unrelated commercial material.</li>
            </ul>

            <h3>Reporting and enforcement</h3>
            <p>Report conduct concerns privately to the repository owner through their GitHub profile. Do not publish sensitive evidence in a public issue. Project maintainers may edit or remove contributions, lock discussions, issue warnings, or temporarily or permanently restrict participation when these standards are violated.</p>
            <div className="docs-callout neutral">
              <strong>Security vulnerabilities</strong>
              <p>Do not open a public issue containing an API key, user document, access token, exploitable vulnerability, or reproduction data belonging to another person. Use a private GitHub security report when available.</p>
            </div>
          </section>

          <section id="more-resources" className="docs-section docs-resources">
            <h2>More resources</h2>
            <div className="docs-resource-grid">
              <ResourceCard icon={Settings} title="Open your workspace" to="/chat">Start a conversation and manage sources.</ResourceCard>
              <ResourceCard icon={ShieldCheck} title="Privacy policy" to="/privacy">See how account and document data is handled.</ResourceCard>
              <ResourceCard icon={Code2} title="GitHub repository" to="https://github.com/GitHpriyanshu23/Ragkno" external>Inspect the source, report an issue, or contribute.</ResourceCard>
            </div>
          </section>

          <Link className="docs-next-link" to="/login?mode=signup">
            <span><small>Next</small><strong>Create your RagKno workspace</strong></span>
            <ArrowRight size={18} />
          </Link>
        </article>

        <aside className="docs-outline" data-lenis-prevent aria-label="On this page">
          <p>On this page</p>
          {ON_THIS_PAGE.map(([id, label]) => (
            <a key={id} href={`#${id}`} className={activeSection === id ? 'active' : ''}>{label}</a>
          ))}
        </aside>
      </div>
      <SiteFooter />
    </div>
  )
}
