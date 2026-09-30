import { useEffect, useState } from 'react'
import pdfIcon from '../../assets/pdf-icon-clean.png'
import docxIcon from '../../assets/docx-icon-clean.png'
import mdIcon from '../../assets/md-icon-clean.png'

const STEPS = [
  {
    number: '01',
    title: 'Bring it in.',
    text: 'Uploaded documents, website pages, and supported Drive files enter a user-scoped index.',
  },
  {
    number: '02',
    title: 'Find the passage.',
    text: 'Meaning and exact words meet, and the closest passage stays.',
  },
  {
    number: '03',
    title: 'Show the proof.',
    text: 'The answer includes citations that open the retrieved evidence for inspection.',
  },
]

function BringScene({ on }) {
  return (
    <div className={`hw-scene hw-bring ${on ? 'is-on' : ''}`} aria-hidden="true">
      <img
        src={pdfIcon}
        alt="PDF"
        className="hw-bring-doc"
        style={{ '--r': '-10deg', '--x': '0px' }}
      />
      <img
        src={docxIcon}
        alt="DOCX"
        className="hw-bring-doc"
        style={{ '--r': '7deg', '--x': '18px' }}
      />
      <img
        src={mdIcon}
        alt="MD"
        className="hw-bring-doc"
        style={{ '--r': '-3deg', '--x': '8px' }}
      />
    </div>
  )
}

function FindScene({ on }) {
  return (
    <div className={`hw-scene hw-find ${on ? 'is-on' : ''}`} aria-hidden="true">
      <p className="hw-find-q">Q3 pricing</p>
      <p className="hw-find-line">Wifi rotation schedule</p>
      <p className="hw-find-line is-hit">Usage-based billing for Q3</p>
      <p className="hw-find-line">Brand color tokens</p>
    </div>
  )
}

function ProofScene({ on }) {
  return (
    <div className={`hw-scene hw-proof ${on ? 'is-on' : ''}`} aria-hidden="true">
      <p>Usage-based billing was introduced for Q3.</p>
      <span>Q3 Report.pdf</span>
    </div>
  )
}

const SCENES = [BringScene, FindScene, ProofScene]

export default function HowRagknoWorks() {
  const [active, setActive] = useState(0)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    if (motion.matches || paused) return undefined
    const id = window.setInterval(() => {
      setActive((current) => (current + 1) % STEPS.length)
    }, 3400)
    return () => window.clearInterval(id)
  }, [paused])

  return (
    <section id="how-it-works" className="how-it-works-section" data-nav-theme="light" aria-label="How RAGKNO Works">
      <div className="how-it-works-inner">
        <header className="hw-head">
          <h2>How it works.</h2>
          <p>Three steps, from the file you already have to an answer you can check.</p>
        </header>

        <div
          className="hw-grid"
          onMouseEnter={() => setPaused(true)}
          onMouseLeave={() => setPaused(false)}
        >
          {STEPS.map((step, index) => {
            const Scene = SCENES[index]
            const on = active === index
            return (
              <article
                key={step.number}
                className={`hw-card ${on ? 'is-on' : ''}`}
                onMouseEnter={() => setActive(index)}
                onFocus={() => setActive(index)}
                tabIndex={0}
              >
                <Scene on={on} />
                <div className="hw-copy">
                  <span>{step.number}</span>
                  <h3>{step.title}</h3>
                  <p>{step.text}</p>
                </div>
              </article>
            )
          })}
        </div>
      </div>
    </section>
  )
}
