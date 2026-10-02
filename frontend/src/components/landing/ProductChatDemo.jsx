import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ArrowUp, ChevronDown, Database, FileText, Globe, Sparkles } from 'lucide-react'
import brandLogo from '../../assets/figma-logo-mark.svg'

const conversations = [
  {
    id: 'leave',
    label: 'Parental leave policy',
    question: 'What is our parental leave policy and when should I notify my manager?',
    answer: 'The employee handbook provides 16 weeks of paid parental leave for eligible full-time employees. It asks employees to notify their manager and People Operations at least 30 days before the expected start date when possible.',
    sources: [
      { index: 1, title: 'Employee_Handbook.pdf', detail: 'Page 42 · Leave benefits', excerpt: 'Eligible full-time employees may take up to 16 weeks of paid parental leave.' },
      { index: 2, title: 'Leave_Request_Guide.txt', detail: 'Planning and notice', excerpt: 'Give your manager and People Operations 30 days notice when the timing is foreseeable.' },
    ],
  },
  {
    id: 'release',
    label: 'Release process',
    question: 'What checks are required before a production release?',
    answer: 'The release guide requires passing automated tests, a completed peer review, and an approved rollback plan before deployment. The runbook also asks the release owner to verify monitoring after rollout.',
    sources: [
      { index: 1, title: 'Engineering_Release_Guide.docx', detail: 'Pre-release checklist', excerpt: 'Tests, peer review, and a rollback plan must be complete before production deployment.' },
      { index: 2, title: 'Production runbook', detail: 'Indexed website', excerpt: 'The release owner verifies service health and alerting immediately after rollout.' },
    ],
  },
  {
    id: 'onboarding',
    label: 'New hire onboarding',
    question: 'Which documents should a new engineer read in their first week?',
    answer: 'The onboarding plan starts with the architecture overview, local development guide, and incident-response handbook. These documents are available in the connected Google Drive folder.',
    sources: [
      { index: 1, title: 'Engineering_Onboarding.pdf', detail: 'Google Drive · Week one', excerpt: 'Read the architecture overview, development setup, and incident-response handbook.' },
      { index: 2, title: 'Local_Development.txt', detail: 'Google Drive · Setup', excerpt: 'Complete the local setup and verify the sample service before taking a starter task.' },
    ],
  },
]

function CitedAnswer({ item }) {
  return (
    <p>
      {item.answer} <button type="button" className="demo-inline-citation">[1]</button>{' '}
      <button type="button" className="demo-inline-citation">[2]</button>
    </p>
  )
}

export default function ProductChatDemo() {
  const [activeId, setActiveId] = useState(conversations[0].id)
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [draft, setDraft] = useState('')
  const reduceMotion = useReducedMotion()
  const active = conversations.find((item) => item.id === activeId) || conversations[0]

  const selectConversation = (id) => {
    setActiveId(id)
    setSourcesOpen(false)
    setDraft('')
  }

  const submitDemo = (event) => {
    event.preventDefault()
    if (!draft.trim()) return
    const normalized = draft.toLowerCase()
    const next = conversations.find((item) => item.question.toLowerCase().includes(normalized) || item.label.toLowerCase().includes(normalized)) || conversations[(conversations.findIndex((item) => item.id === activeId) + 1) % conversations.length]
    selectConversation(next.id)
  }

  return (
    <section id="chat-demo" className="chat-demo-section" data-nav-theme="light" aria-labelledby="chat-demo-title">
      <div className="chat-demo-wave" aria-hidden="true">
        <div className="chat-demo-wave-glow" />
        <svg viewBox="0 0 1440 980" preserveAspectRatio="none">
          <defs>
            <linearGradient id="chat-wave-top" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#0ea5e9" />
              <stop offset="0.58" stopColor="#7dd3fc" />
              <stop offset="1" stopColor="#e0f2fe" />
            </linearGradient>
            <linearGradient id="chat-wave-center" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stopColor="#e0f2fe" />
              <stop offset="0.48" stopColor="#ffffff" />
              <stop offset="1" stopColor="#dbeafe" />
            </linearGradient>
            <linearGradient id="chat-wave-bottom" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#bae6fd" />
              <stop offset="0.55" stopColor="#38bdf8" />
              <stop offset="1" stopColor="#0284c7" />
            </linearGradient>
          </defs>
          <path className="chat-wave-layer chat-wave-layer-top" fill="url(#chat-wave-top)" d="M-120 190C170 22 384-56 664 56c305 122 404 296 896 246V-80H-120Z" />
          <path className="chat-wave-layer chat-wave-layer-center" fill="url(#chat-wave-center)" d="M-120 258c264 30 390 194 667 224 335 36 476-224 1013-155v354c-335-79-525 103-842 106-319 4-509-185-838-206Z" />
          <path className="chat-wave-layer chat-wave-layer-bottom" fill="url(#chat-wave-bottom)" d="M-120 601c293 25 417 218 720 222 332 4 481-221 960-153v390H-120Z" />
        </svg>
      </div>
      <div className="chat-demo-inner">
        <div className="section-head-light">
          <span className="section-badge-light"><Sparkles size={12} /> DOCUMENT Q&amp;A</span>
          <h2 id="chat-demo-title">Ask naturally.<br />Verify every answer.</h2>
          <p>RagKno retrieves relevant passages from your indexed documents and keeps the supporting text one click away.</p>
        </div>

        <motion.div className="premium-chat-demo" initial={reduceMotion ? false : { opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .25 }} transition={{ duration: .5 }}>
          <div className="premium-chat-status"><span><i /> RagKno workspace</span><span>Hybrid retrieval · citations</span></div>

          <div className="premium-chat-body">
            <AnimatePresence mode="wait">
              <motion.div key={active.id} className="premium-chat-exchange" initial={reduceMotion ? false : { opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={reduceMotion ? undefined : { opacity: 0, y: -8 }} transition={{ duration: .24 }}>
                <div className="premium-user-message">{active.question}</div>
                <article className="premium-assistant-answer">
                  <div className="premium-answer-mark">
                    <img src={brandLogo} alt="" aria-hidden="true" />
                  </div>
                  <div>
                    <button type="button" className="premium-source-toggle" onClick={() => setSourcesOpen((value) => !value)} aria-expanded={sourcesOpen}>
                      <Database size={14} /> Used {active.sources.length} sources <ChevronDown size={14} className={sourcesOpen ? 'open' : ''} />
                    </button>
                    <CitedAnswer item={active} />
                    <AnimatePresence initial={false}>
                      {sourcesOpen && (
                        <motion.div className="premium-source-grid" initial={reduceMotion ? false : { opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
                          {active.sources.map((source) => (
                            <article key={source.index} className="premium-source-card">
                              <header><FileText size={14} /><strong>[{source.index}] {source.title}</strong></header>
                              <small>{source.detail}</small>
                              <p>{source.excerpt}</p>
                            </article>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </article>
              </motion.div>
            </AnimatePresence>
          </div>

          <div className="premium-chat-controls">
            <div className="premium-suggestions" aria-label="Sample questions">
              {conversations.map((item) => (
                <button key={item.id} type="button" className={item.id === active.id ? 'active' : ''} onClick={() => selectConversation(item.id)}>{item.label}</button>
              ))}
            </div>
            <form className="premium-demo-composer" onSubmit={submitDemo}>
              <Globe size={17} />
              <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask a question about your documents…" aria-label="Try a sample document question" />
              <button type="submit" aria-label="Submit demo question"><ArrowUp size={17} /></button>
            </form>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
