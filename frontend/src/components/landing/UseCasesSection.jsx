import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { BookOpen, Check, Cloud, FileSearch, Files, Terminal } from 'lucide-react'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../ui/accordion.jsx'

const workflows = [
  {
    id: 'policies',
    number: '01',
    title: 'Policies and handbooks',
    icon: FileSearch,
    prompt: '“What changed in our leave policy, and where is it written?”',
    description: 'Retrieve policy passages from indexed PDF, DOCX, TXT, and website content, then inspect the excerpts behind the answer.',
    outcome: ['Hybrid semantic and keyword retrieval', 'Inline citations to retrieved passages', 'Source text available for verification'],
    formats: ['PDF', 'DOCX', 'TXT', 'Web pages'],
  },
  {
    id: 'engineering',
    number: '02',
    title: 'Engineering documentation',
    icon: Terminal,
    prompt: '“Which checks does the deployment runbook require?”',
    description: 'Search architecture notes, runbooks, release guides, and other text-based technical documents in one indexed workspace.',
    outcome: ['CrossEncoder reranking', 'Conversation-aware follow-ups', 'Selectable document sources'],
    formats: ['Runbooks', 'ADRs', 'Guides', 'Static docs'],
  },
  {
    id: 'drive',
    number: '03',
    title: 'Google Drive knowledge',
    icon: Cloud,
    prompt: '“Summarize the onboarding material in the selected Drive files.”',
    description: 'Connect Google Drive with read-only access and manually sync supported files into a private user-scoped index.',
    outcome: ['Read-only Google Drive connection', 'Choose which files to sync', 'Drive links in source citations'],
    formats: ['Google Docs', 'Drive PDFs', 'Drive TXT'],
  },
  {
    id: 'research',
    number: '04',
    title: 'Multi-document research',
    icon: Files,
    prompt: '“Compare how these documents define the same requirement.”',
    description: 'Ask questions across an indexed collection and review the strongest retrieved passages before relying on the synthesis.',
    outcome: ['Multiple indexed documents', 'Source-diverse retrieval', 'Persistent conversation history'],
    formats: ['Reports', 'Notes', 'References', 'Web sources'],
  },
]

function WorkflowDetail({ workflow }) {
  const Icon = workflow.icon
  return (
    <div className="workflow-detail-inner">
      <div className="workflow-detail-icon"><Icon size={21} /></div>
      <p className="workflow-prompt">{workflow.prompt}</p>
      <h3>{workflow.title}</h3>
      <p className="workflow-description">{workflow.description}</p>
      <div className="workflow-proof-list">
        {workflow.outcome.map((item) => <span key={item}><Check size={14} /> {item}</span>)}
      </div>
      <div className="workflow-formats">{workflow.formats.map((item) => <span key={item}>{item}</span>)}</div>
    </div>
  )
}

export default function UseCasesSection() {
  const [activeId, setActiveId] = useState(workflows[0].id)
  const reduceMotion = useReducedMotion()
  const active = workflows.find((item) => item.id === activeId) || workflows[0]

  return (
    <section className="use-cases-section" data-nav-theme="dark" aria-labelledby="workflows-title">
      <div className="use-cases-inner">
        <div className="section-head-dark editorial-workflow-head">
          <span className="section-badge-dark"><BookOpen size={12} /> WORKING WITH YOUR KNOWLEDGE</span>
          <h2 id="workflows-title">One retrieval workflow.<br />Many kinds of questions.</h2>
          <p>Use RagKno where the supporting document matters as much as the generated answer.</p>
        </div>

        <div className="workflow-desktop">
          <div className="workflow-tabs" role="tablist" aria-label="Document workflows">
            {workflows.map((workflow) => {
              const Icon = workflow.icon
              const selected = workflow.id === active.id
              return (
                <button key={workflow.id} type="button" role="tab" aria-selected={selected} aria-controls={`workflow-panel-${workflow.id}`} className={selected ? 'active' : ''} onClick={() => setActiveId(workflow.id)}>
                  <span>{workflow.number}</span><Icon size={18} /><strong>{workflow.title}</strong>
                </button>
              )
            })}
          </div>
          <div className="workflow-panel" role="tabpanel" id={`workflow-panel-${active.id}`}>
            <AnimatePresence mode="wait">
              <motion.div key={active.id} initial={reduceMotion ? false : { opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} exit={reduceMotion ? undefined : { opacity: 0, x: -12 }} transition={{ duration: .25 }}>
                <WorkflowDetail workflow={active} />
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        <Accordion type="single" defaultValue={workflows[0].id} className="workflow-mobile">
          {workflows.map((workflow) => (
            <AccordionItem key={workflow.id} value={workflow.id} className="workflow-mobile-item">
              <AccordionTrigger className="workflow-mobile-trigger"><span>{workflow.number}</span>{workflow.title}</AccordionTrigger>
              <AccordionContent className="workflow-mobile-content"><WorkflowDetail workflow={workflow} /></AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </section>
  )
}
