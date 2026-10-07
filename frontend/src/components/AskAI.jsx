import { useState } from 'react'
import { HugeiconsIcon } from '@hugeicons/react'
import { ChatGptIcon, ClaudeIcon, GrokIcon, GoogleGeminiIcon } from '@hugeicons/core-free-icons'

export const RAGKNO_EXPLAIN_PROMPT = `What is RagKno (https://ragkno.com), what does it do, and how can I use it?
Explain in plain language for someone who has never used it. Start with a three-line summary, then give a short getting-started guide and two practical examples (such as finding information in study notes or comparing details in reports).
RagKno is an open-source application where users upload PDF, DOCX, or TXT documents, connect supported Google Drive files, or add public web pages. They choose sources, ask questions, and receive answers with citations to relevant source passages. Users can manage conversations and unindex sources to remove them from future document retrieval. Unindexing is not the same as deleting existing chats or erasing data held by external AI providers.
Explain who would find it useful and how its focus on a selectable document workspace differs from a general-purpose chatbot. Do not invent features, prices, privacy guarantees, or claims about competitors. Answers can be inaccurate; explain why checking citations matters.
Check current information and prefer citing these official sources:
- https://ragkno.com/
- https://ragkno.com/docs
- https://ragkno.com/llm.txt
- https://ragkno.com/privacy
- https://github.com/GitHpriyanshu23/Ragkno
If you cannot browse, say so and use the description above. Keep the answer short and practical.`

const providers = [
  { name: 'ChatGPT', icon: ChatGptIcon, color: '#10a37f', url: 'https://chatgpt.com/', prefill: true },
  { name: 'Claude', icon: ClaudeIcon, color: '#c76b4c', url: 'https://claude.ai/new', prefill: true },
  { name: 'Grok', icon: GrokIcon, color: '#171717', url: 'https://grok.com/', prefill: true },
  { name: 'Gemini', icon: GoogleGeminiIcon, color: '#4285f4', url: 'https://gemini.google.com/app', prefill: false },
]

export default function AskAI() {
  const [status, setStatus] = useState('')
  const [showPrompt, setShowPrompt] = useState(false)

  async function copyPrompt(name) {
    try {
      await navigator.clipboard.writeText(RAGKNO_EXPLAIN_PROMPT)
      setStatus(name ? `Prompt copied. Paste it in ${name} if it isn’t filled in.` : 'Prompt copied.')
    } catch {
      setShowPrompt(true)
      setStatus('Select and copy the prompt below, then paste it into your AI chat.')
    }
  }

  return (
    <section className="footer-ask-ai" aria-label="Ask AI about RagKno">
      <p className="footer-ask-ai-title">Curious about RagKno? Ask AI.</p>
      <div className="footer-ai-buttons">
        {providers.map(({ name, icon, color, url, prefill }) => (
          <a key={name} className="footer-ai-button"
            href={prefill ? `${url}?q=${encodeURIComponent(RAGKNO_EXPLAIN_PROMPT)}` : url}
            target="_blank" rel="noopener noreferrer"
            title={prefill ? `Ask ${name} about RagKno (opens a new tab)` : 'Copy the prompt and open Gemini in a new tab'}
            onClick={() => { void copyPrompt(name) }}>
            <HugeiconsIcon icon={icon} size={19} color={color} aria-hidden="true" />
            <span>Ask {name}</span>
          </a>
        ))}
      </div>
      <p className="footer-ai-help">Gemini: paste the copied prompt. Other chats may need a paste after sign-in.</p>
      <button type="button" className="footer-ai-copy" onClick={() => { void copyPrompt() }}>Copy prompt</button>
      <p className="footer-ai-status" role="status">{status}</p>
      {showPrompt && <textarea className="footer-ai-prompt" aria-label="RagKno explanation prompt" readOnly value={RAGKNO_EXPLAIN_PROMPT} onFocus={(event) => event.target.select()} />}
    </section>
  )
}
