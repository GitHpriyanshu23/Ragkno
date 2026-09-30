import { useEffect, useRef, useState } from 'react'
import brandLogo from '../../assets/figma-logo-mark.svg'
import notionLogo from '../../assets/notion-logo.png'
import './bento.css'

function GoogleDriveIcon() {
  return (
    <svg viewBox="0 0 87.3 78" fill="none" aria-hidden="true">
      <path d="m6.6 66.85 3.85 6.65c.8 1.4 1.95 2.5 3.3 3.3l13.75-23.8H0c0 1.55.4 3.1 1.2 4.5z" fill="#0066da" />
      <path d="M43.65 25 29.9 1.2c-1.35.8-2.5 1.9-3.3 3.3l-25.4 44A9.06 9.06 0 0 0 0 53h27.5z" fill="#00ac47" />
      <path d="M73.55 76.8c1.35-.8 2.5-1.9 3.3-3.3l1.6-2.75 7.65-13.25c.8-1.4 1.2-2.95 1.2-4.5H59.8l5.85 10.15z" fill="#ea4335" />
      <path d="M43.65 25 57.4 1.2C56.05.4 54.5 0 52.9 0H34.4c-1.6 0-3.15.45-4.5 1.2z" fill="#00832d" />
      <path d="M59.8 53H27.5L13.75 76.8c1.35.8 2.9 1.2 4.5 1.2h50.8c1.6 0 3.15-.45 4.5-1.2z" fill="#2684fc" />
      <path d="m73.4 26.5-12.7-22c-.8-1.4-1.95-2.5-3.3-3.3L43.65 25l16.15 28h27.5c0-1.55-.4-3.1-1.2-4.5z" fill="#ffba00" />
    </svg>
  )
}

function NotionIcon() {
  return (
    <img
      src={notionLogo}
      alt="Notion"
      loading="lazy"
      decoding="async"
      className="bx-notion-icon"
      width={22}
      height={22}
    />
  )
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  )
}

function SlackIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313z" fill="#E01E5A" />
      <path d="M8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312z" fill="#36C5F0" />
      <path d="M18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312z" fill="#2EB67D" />
      <path d="M15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" fill="#ECB22E" />
    </svg>
  )
}

function WebIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9.5" stroke="#3B82F6" strokeWidth="1.8" />
      <ellipse cx="12" cy="12" rx="4.5" ry="9.5" stroke="#3B82F6" strokeWidth="1.8" />
      <path d="M2.5 12h19M3.8 7.5h16.4M3.8 16.5h16.4" stroke="#3B82F6" strokeWidth="1.8" />
    </svg>
  )
}

function PdfIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="20" height="20" rx="4" fill="#E11D48" />
      <path d="M7 16V8h2.8c1.3 0 2.2.8 2.2 2s-.9 2-2.2 2H8.6v4H7zm1.6-5.4h1.1c.5 0 .9-.3.9-.7s-.4-.7-.9-.7H8.6v1.4zm4.8 5.4V8h2.6c2 0 3.2 1.4 3.2 4s-1.2 4-3.2 4h-2.6zm1.6-1.4h1c1.1 0 1.7-.8 1.7-2.6s-.6-2.6-1.7-2.6h-1v5.2z" fill="#fff" />
    </svg>
  )
}

function DocxIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="20" height="20" rx="4" fill="#2563EB" />
      <path d="M6.5 7.5h2.2l1.6 5.8 1.7-5.8h1.8l1.7 5.8 1.6-5.8h2.1l-2.4 9h-2l-1.7-5.8-1.7 5.8h-2l-2.9-9z" fill="#fff" />
    </svg>
  )
}

function XlsxIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="20" height="20" rx="4" fill="#16A34A" />
      <path d="M7.5 7.5h2.5l2 3.8 2-3.8h2.5l-3.2 5.2 3.4 5.3h-2.6l-2.1-3.9-2.1 3.9H7.3l3.4-5.3-3.2-5.2z" fill="#fff" />
    </svg>
  )
}

function CsvIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="20" height="20" rx="4" fill="#0D9488" />
      <path d="M6 7h12v2H6V7zm0 4h12v2H6v-2zm0 4h12v2H6v-2z" fill="#fff" />
    </svg>
  )
}

function MarkdownIcon() {
  return (
    <svg className="bx-md" viewBox="0 0 208 128" aria-hidden="true">
      <rect width="208" height="128" rx="18" fill="#1E293B" />
      <path d="M30 98V30h20l20 25 20-25h20v68H90V59L70 84 50 59v39zm125 0l-30-33h20V30h20v35h20z" fill="#94A3B8" />
    </svg>
  )
}

const OUTER_ORBIT = [
  { id: 'drive', name: 'Google Drive', icon: <GoogleDriveIcon /> },
  { id: 'web', name: 'Websites', icon: <WebIcon /> },
  { id: 'upload', name: 'Local uploads', icon: <DocxIcon /> },
]

const INNER_ORBIT = [
  { id: 'pdf', name: 'PDF', icon: <PdfIcon /> },
  { id: 'docx', name: 'DOCX', icon: <DocxIcon /> },
  { id: 'txt', name: 'TXT', icon: <MarkdownIcon /> },
]

const RETRIEVAL_DOCS = [
  { id: 'q3', name: 'Q3 Report.pdf', ext: 'PDF', x: 14, y: 28, rel: 'high', depth: 1.15, dur: '7.2s', delay: '0s' },
  { id: 'pricing', name: 'Pricing.docx', ext: 'DOC', x: 84, y: 22, rel: 'high', depth: 0.95, dur: '8.4s', delay: '-1.4s' },
  { id: 'notes', name: 'Release Notes.txt', ext: 'TXT', x: 82, y: 78, rel: 'low', depth: 0.55, dur: '9s', delay: '-2.2s' },
  { id: 'handbook', name: 'Handbook.pdf', ext: 'PDF', x: 14, y: 80, rel: 'low', depth: 0.4, dur: '8s', delay: '-0.6s' },
]

const INTEL_STEPS = [
  { id: 'meaning', label: 'Meaning' },
  { id: 'words', label: 'Exact words' },
  { id: 'best', label: 'Best match' },
]

const INTEL_PASSAGES = [
  { id: 'bill', text: 'Usage-based billing replaced seat pricing.', meaning: true, words: true },
  { id: 'plan', text: 'Enterprise plan kept its annual commitment.', meaning: true, words: false },
  { id: 'wifi', text: 'Office wifi rotation schedule.', meaning: false, words: false },
]

const GRAPH_NODES = [
  { id: 'hub', label: 'Thread', x: 230, y: 150, hub: true },
  { id: 'question', label: 'Question', x: 58, y: 42 },
  { id: 'answer', label: 'Answer', x: 392, y: 40 },
  { id: 'sources', label: 'Sources', x: 414, y: 168 },
  { id: 'citation', label: 'Citation', x: 52, y: 236 },
  { id: 'followup', label: 'Follow-up', x: 286, y: 272 },
]

const GRAPH_EDGES = [
  ['hub', 'question'],
  ['hub', 'answer'],
  ['hub', 'sources'],
  ['hub', 'citation'],
  ['hub', 'followup'],
  ['question', 'answer'],
  ['answer', 'sources'],
  ['sources', 'citation'],
  ['answer', 'followup'],
]

const WHISPERS = [
  { id: 'near', text: 'Enterprise plan kept its annual commitment.', level: 'mid', x: '28%', y: '28%', depth: 0.55 },
  { id: 'wifi', text: 'Wifi rotation schedule', level: 'low', x: '80%', y: '20%', depth: 1 },
  { id: 'brand', text: 'Brand color tokens', level: 'low', x: '78%', y: '82%', depth: 0.75 },
  { id: 'tax', text: 'FY24 tax appendix', level: 'low', x: '20%', y: '80%', depth: 0.85 },
]

const TRUST_SOURCES = [
  { id: 'q3', name: 'Q3 Report.pdf', ext: 'PDF', pos: 'q3' },
  { id: 'policy', name: 'Pricing Policy.docx', ext: 'DOC', pos: 'policy' },
  { id: 'notes', name: 'Release Notes.txt', ext: 'TXT', pos: 'finance' },
  { id: 'web', name: 'Pricing page', ext: 'WEB', pos: 'product' },
]

function useReducedMotion() {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(mq.matches)
    update()
    mq.addEventListener('change', update)
    return () => mq.removeEventListener('change', update)
  }, [])
  return reduced
}

function useParallax(reduced) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return undefined
    if (reduced) {
      el.style.setProperty('--px', '0')
      el.style.setProperty('--py', '0')
      return undefined
    }
    let frame = 0
    const move = (event) => {
      const rect = el.getBoundingClientRect()
      const x = (event.clientX - rect.left) / rect.width - 0.5
      const y = (event.clientY - rect.top) / rect.height - 0.5
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        el.style.setProperty('--px', x.toFixed(3))
        el.style.setProperty('--py', y.toFixed(3))
      })
    }
    const leave = () => {
      el.style.setProperty('--px', '0')
      el.style.setProperty('--py', '0')
    }
    el.addEventListener('mousemove', move)
    el.addEventListener('mouseleave', leave)
    return () => {
      cancelAnimationFrame(frame)
      el.removeEventListener('mousemove', move)
      el.removeEventListener('mouseleave', leave)
    }
  }, [reduced])
  return ref
}

function curveBetween(doc) {
  const x1 = doc.x
  const y1 = doc.y
  const x2 = 50
  const y2 = 58
  const mx = (x1 + x2) / 2
  const my = (y1 + y2) / 2
  const dx = x2 - x1
  const dy = y2 - y1
  return `M ${x1} ${y1} Q ${mx - dy * 0.15} ${my + dx * 0.12} ${x2} ${y2}`
}

function RetrievalCard({ reduced }) {
  const ref = useParallax(reduced)
  const [active, setActive] = useState(null)

  return (
    <article
      ref={ref}
      className="bx-card bx-card-retrieval"
      aria-labelledby="bx-retrieval-title"
      onMouseLeave={() => setActive(null)}
    >
      <div className="bx-copy">
        <h3 id="bx-retrieval-title">Find what matters.</h3>
        <p>Retrieve the right context from everything you know.</p>
      </div>
      <div className="bx-visual">
        <div className="bx-retrieve">
          <div className="bx-retrieve-glow" />
          <svg className="bx-paths" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {RETRIEVAL_DOCS.map((doc) => (
              <path
                key={doc.id}
                className={`bx-path ${doc.rel} ${active === doc.id ? 'is-on' : ''}`}
                d={curveBetween(doc)}
                pathLength="1"
              />
            ))}
          </svg>
          <div className="bx-focal" aria-hidden="true">
            <span className="bx-focal-ring bx-focal-ring-b" />
            <span className="bx-focal-ring bx-focal-ring-a" />
            <span className="bx-focal-core" />
          </div>
          <p className="bx-query">
            What changed in our Q3 pricing?
            <span className="bx-caret" aria-hidden="true" />
          </p>
          {RETRIEVAL_DOCS.map((doc) => {
            const toward = doc.rel === 'high' ? 0.42 : 0.28
            const away = doc.rel === 'low' ? -0.18 : 0
            return (
              <div
                key={doc.id}
                className={`bx-doc-slot ${active === doc.id ? 'is-on' : ''}`}
                data-rel={doc.rel}
                style={{
                  left: `${doc.x}%`,
                  top: `${doc.y}%`,
                  '--pull-x': `${(50 - doc.x) * toward}px`,
                  '--pull-y': `${(58 - doc.y) * toward}px`,
                  '--away-x': `${(50 - doc.x) * away}px`,
                  '--away-y': `${(58 - doc.y) * away}px`,
                  '--depth': doc.depth,
                  '--dur': doc.dur,
                  '--delay': doc.delay,
                }}
              >
                <div className="bx-doc-float">
                  <div className="bx-doc-shift">
                    <button
                      type="button"
                      className="bx-doc"
                      onMouseEnter={() => setActive(doc.id)}
                      onFocus={() => setActive(doc.id)}
                      onBlur={() => setActive(null)}
                    >
                      <span className="bx-ext">{doc.ext}</span>
                      <span className="bx-doc-name">{doc.name}</span>
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </article>
  )
}

function OrbitRing({ items, variant, active, setActive }) {
  return (
    <div className={`bx-ring bx-ring-${variant}`}>
      {items.map((item, index) => {
        const angle = (360 / items.length) * index + (variant === 'inner' ? 36 : 0)
        return (
          <div key={item.id} className="bx-spoke" style={{ transform: `rotate(${angle}deg)` }}>
            <div className="bx-orbit-pos">
              <div className="bx-counter">
                <div className="bx-level" style={{ transform: `rotate(${-angle}deg)` }}>
                  <button
                    type="button"
                    className={`bx-node ${active === item.id ? 'is-on' : ''}`}
                    aria-label={item.name}
                    onMouseEnter={() => setActive(item.id)}
                    onMouseLeave={() => setActive(null)}
                    onFocus={() => setActive(item.id)}
                    onBlur={() => setActive(null)}
                  >
                    {item.icon}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function OrbitCard() {
  const [active, setActive] = useState(null)

  return (
    <article className="bx-card bx-card-orbit" aria-labelledby="bx-orbit-title">
      <div className="bx-visual">
        <div className={`bx-orbit-stage ${active ? 'is-live' : ''}`}>
          <div className="bx-track bx-track-outer" />
          <div className="bx-track bx-track-inner" />
          <OrbitRing items={OUTER_ORBIT} variant="outer" active={active} setActive={setActive} />
          <OrbitRing items={INNER_ORBIT} variant="inner" active={active} setActive={setActive} />
          <div className="bx-hub">
            <img src={brandLogo} alt="" />
          </div>
        </div>
      </div>
      <div className="bx-copy">
        <h3 id="bx-orbit-title">Supported sources, one index.</h3>
        <p>Upload PDF, DOCX, or TXT files, add a website URL, or import supported Drive documents.</p>
      </div>
    </article>
  )
}

function IntelCard({ reduced }) {
  const [step, setStep] = useState(0)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    if (reduced || paused) return undefined
    const id = window.setInterval(() => setStep((current) => (current + 1) % INTEL_STEPS.length), 2400)
    return () => window.clearInterval(id)
  }, [reduced, paused])

  const phase = INTEL_STEPS[reduced ? 2 : step].id

  return (
    <article
      className="bx-card bx-card-intel"
      aria-labelledby="bx-intel-title"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <div className="bx-copy">
        <h3 id="bx-intel-title">Intelligent retrieval.</h3>
        <p>Meaning and exact words meet, then the closest passage stays.</p>
      </div>
      <div className="bx-visual">
        <div className={`bx-intel is-${phase}`}>
          <p className="bx-intel-query">
            What changed in our Q3 pricing?
            <span className="bx-caret" aria-hidden="true" />
          </p>
          <div className="bx-intel-phases">
            {INTEL_STEPS.map((item, index) => (
              <button
                key={item.id}
                type="button"
                className={phase === item.id ? 'is-on' : ''}
                onClick={() => setStep(index)}
              >
                {item.label}
              </button>
            ))}
            <i style={{ transform: `translateX(${(reduced ? 2 : step) * 100}%)` }} />
          </div>
          <div className="bx-intel-list">
            {INTEL_PASSAGES.map((row) => {
              const on = phase === 'best' ? row.meaning && row.words : phase === 'words' ? row.words : row.meaning
              return (
                <p key={row.id} className={`bx-intel-row ${on ? 'is-on' : ''} ${phase === 'best' && !on ? 'is-out' : ''}`}>
                  <i aria-hidden="true" />
                  {row.text}
                </p>
              )
            })}
          </div>
        </div>
      </div>
    </article>
  )
}

function GraphCard({ reduced }) {
  const ref = useParallax(reduced)
  const [active, setActive] = useState(null)
  const nodeById = Object.fromEntries(GRAPH_NODES.map((node) => [node.id, node]))
  const related = new Set()
  if (active) {
    related.add(active)
    GRAPH_EDGES.forEach(([from, to]) => {
      if (from === active) related.add(to)
      if (to === active) related.add(from)
    })
  }
  const focus = active ? nodeById[active] : null
  const shift = focus
    ? `translate(${(focus.x - 230) / 16}px, ${(focus.y - 148) / 16}px)`
    : 'translate(0px, 0px)'

  return (
    <article ref={ref} className="bx-card bx-card-graph" aria-labelledby="bx-graph-title">
      <div className="bx-copy">
        <h3 id="bx-graph-title">Keep the conversation in context.</h3>
        <p>Questions, answers, citations, and follow-ups remain attached to your thread.</p>
      </div>
      <div className="bx-visual">
        <div className="bx-graph-visual">
          <div className="bx-graph-parallax">
            <div className="bx-graph-shift" style={{ transform: shift }}>
              <svg className="bx-graph-svg" viewBox="0 0 460 300" role="img" aria-label="Conversation context">
                {GRAPH_EDGES.map(([from, to]) => {
                  const a = nodeById[from]
                  const b = nodeById[to]
                  const hot = active && related.has(from) && related.has(to)
                  const dim = active && !hot
                  return (
                    <line
                      key={`${from}-${to}`}
                      x1={a.x}
                      y1={a.y}
                      x2={b.x}
                      y2={b.y}
                      className={`bx-edge ${hot ? 'is-hot' : ''} ${dim ? 'is-dim' : ''}`}
                    />
                  )
                })}
                {GRAPH_NODES.map((node) => {
                  const on = active === node.id
                  const dim = active && !related.has(node.id)
                  const fill = dim ? 'rgba(255,255,255,0.28)' : '#f4f4f4'
                  return (
                    <g
                      key={node.id}
                      className="bx-g-node"
                      tabIndex={0}
                      role="button"
                      aria-label={node.label}
                      onMouseEnter={() => setActive(node.id)}
                      onMouseLeave={() => setActive(null)}
                      onFocus={() => setActive(node.id)}
                      onBlur={() => setActive(null)}
                    >
                      {node.hub && (
                        <circle cx={node.x} cy={node.y} r="34" fill="rgba(255,255,255,0.035)" />
                      )}
                      {node.hub && (
                        <circle cx={node.x} cy={node.y} r="24" fill="none" stroke={on || !dim ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.08)'} />
                      )}
                      <circle cx={node.x} cy={node.y} r="16" fill="transparent" />
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={node.hub ? 15 : on ? 6 : 4.5}
                        fill={node.hub ? '#101010' : fill}
                        stroke={node.hub ? 'rgba(255,255,255,0.55)' : 'transparent'}
                        strokeWidth={node.hub ? 1.2 : 0}
                      />
                      <text
                        className={`bx-g-label ${node.hub ? 'hub' : ''}`}
                        x={node.x}
                        y={node.hub ? node.y + 36 : node.y - 14}
                        textAnchor="middle"
                        fill={dim ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.86)'}
                      >
                        {node.label}
                      </text>
                    </g>
                  )
                })}
              </svg>
            </div>
          </div>
        </div>
      </div>
    </article>
  )
}

function SearchCard({ reduced }) {
  const ref = useParallax(reduced)
  return (
    <article ref={ref} className="bx-card bx-card-search" aria-labelledby="bx-search-title">
      <div className="bx-copy">
        <h3 id="bx-search-title">Search by meaning.</h3>
        <p>Find useful knowledge instead of matching words blindly.</p>
      </div>
      <div className="bx-visual">
        <div className="bx-search-stage">
          <p className="bx-search-query">enterprise pricing</p>
          <div className="bx-search-field">
            <span className="bx-search-link" aria-hidden="true" />
            {WHISPERS.map((item) => (
              <p
                key={item.id}
                className="bx-whisper"
                data-level={item.level}
                style={{ '--x': item.x, '--y': item.y, '--depth': item.depth }}
              >
                <span className="bx-whisper-shift">{item.text}</span>
              </p>
            ))}
            <p className="bx-hit">
              <span className="bx-hit-em">Usage-based billing</span> replaced seat pricing for new enterprise accounts.
            </p>
          </div>
        </div>
      </div>
    </article>
  )
}

function trustCurve(line) {
  const mx = (line.x1 + line.x2) / 2
  const my = (line.y1 + line.y2) / 2
  const dx = line.x2 - line.x1
  const dy = line.y2 - line.y1
  const len = Math.hypot(dx, dy) || 1
  const cx = mx - (dy / len) * 16
  const cy = my + (dx / len) * 16
  return `M ${line.x1} ${line.y1} Q ${cx} ${cy} ${line.x2} ${line.y2}`
}

const TRUST_TARGET = {
  q3: 'billing',
  notes: 'billing',
  policy: 'plan',
  web: 'remained',
}

const TRUST_ORDER = ['q3', 'policy', 'notes', 'web']

function TrustCard({ reduced }) {
  const fieldRef = useRef(null)
  const pathRef = useRef(null)
  const sourceRefs = useRef({})
  const markRefs = useRef({})
  const [step, setStep] = useState(0)
  const [hovered, setHovered] = useState(null)
  const active = hovered || TRUST_ORDER[reduced ? 0 : step]

  useEffect(() => {
    if (reduced || hovered) return undefined
    const id = window.setInterval(() => {
      setStep((current) => (current + 1) % TRUST_ORDER.length)
    }, 2200)
    return () => window.clearInterval(id)
  }, [reduced, hovered])

  useEffect(() => {
    const path = pathRef.current
    const field = fieldRef.current
    if (!path) return undefined
    if (!active || !field) {
      path.setAttribute('d', '')
      path.style.opacity = '0'
      return undefined
    }
    let frame = 0
    const draw = () => {
      const from = sourceRefs.current[active]
      const to = markRefs.current[TRUST_TARGET[active]]
      const stage = field.getBoundingClientRect()
      if (from && to && stage.width) {
        const a = from.getBoundingClientRect()
        const b = to.getBoundingClientRect()
        path.setAttribute('d', trustCurve({
          x1: a.left + a.width / 2 - stage.left,
          y1: a.top + a.height / 2 - stage.top,
          x2: b.left + b.width / 2 - stage.left,
          y2: b.top + b.height / 2 - stage.top,
        }))
        path.style.opacity = '1'
      }
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frame)
  }, [active])

  const mark = (id, ...sources) => sources.includes(active)

  return (
    <article className="bx-card bx-card-trust" aria-labelledby="bx-trust-title">
      <div className="bx-copy">
        <h3 id="bx-trust-title">Answers with inspectable evidence.</h3>
        <p>Open the retrieved passages used to support each response.</p>
      </div>
      <div className="bx-visual">
        <div ref={fieldRef} className={`bx-trust-field ${active ? 'is-live' : ''}`}>
          <svg className="bx-trust-svg" aria-hidden="true">
            <path ref={pathRef} className="bx-trust-path" style={{ opacity: 0 }} />
          </svg>
          {TRUST_SOURCES.map((source) => (
            <div key={source.id} className={`bx-source-pos ${source.pos}`}>
              <button
                type="button"
                ref={(node) => { sourceRefs.current[source.id] = node }}
                className={`bx-source ${active === source.id ? 'is-on' : ''}`}
                onMouseEnter={() => setHovered(source.id)}
                onMouseLeave={() => setHovered(null)}
                onFocus={() => setHovered(source.id)}
                onBlur={() => setHovered(null)}
              >
                <span className="bx-source-ext">{source.ext}</span>
                {source.name}
                <span className="bx-check" aria-hidden="true">
                  <svg width="10" height="10" viewBox="0 0 10 10">
                    <path d="M2 5.2 4 7.2 8 2.8" fill="none" stroke="#141414" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
              </button>
            </div>
          ))}
          <div className="bx-answer-wrap">
            <p className="bx-question">What changed in our Q3 pricing strategy?</p>
            <p className="bx-answer">
              <span ref={(node) => { markRefs.current.billing = node }} className={`bx-em ${mark('billing', 'q3', 'notes') ? 'is-on' : ''}`}>
                Usage-based billing
              </span>
              {' '}was introduced for{' '}
              <span ref={(node) => { markRefs.current.q3 = node }} className={`bx-em ${mark('q3', 'q3') ? 'is-on' : ''}`}>
                Q3
              </span>
              {' '}while the{' '}
              <span ref={(node) => { markRefs.current.plan = node }} className={`bx-em ${mark('plan', 'policy') ? 'is-on' : ''}`}>
                existing enterprise plan
              </span>
              {' '}
              <span ref={(node) => { markRefs.current.remained = node }} className={`bx-em ${mark('remained', 'web') ? 'is-on' : ''}`}>
                remained
              </span>
              .
            </p>
          </div>
        </div>
      </div>
    </article>
  )
}

export default function BentoCapabilities() {
  const reduced = useReducedMotion()

  return (
    <section id="capabilities" className="bento-section" data-nav-theme="dark" aria-label="Core capabilities">
      <div className="bento-inner">
        <div className="section-head-dark">
          <h2>
            Core capabilities.
          </h2>
          <p>Six working parts of the document question-and-answer workflow.</p>
        </div>

        <div className="bx-grid">
          <RetrievalCard reduced={reduced} />
          <OrbitCard />
          <IntelCard reduced={reduced} />
          <GraphCard reduced={reduced} />
          <SearchCard reduced={reduced} />
          <TrustCard reduced={reduced} />
        </div>
      </div>
    </section>
  )
}
