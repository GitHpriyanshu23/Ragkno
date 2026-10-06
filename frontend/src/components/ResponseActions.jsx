import { useEffect, useState } from 'react'
import { Check, Copy, ThumbsDown, ThumbsUp } from 'lucide-react'

export default function ResponseActions({ text, rating = null, canRate = true, onRate, onError }) {
  const [saving, setSaving] = useState(false)
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return undefined
    const timer = setTimeout(() => setCopied(false), 2000)
    return () => clearTimeout(timer)
  }, [copied])

  async function rate(value) {
    if (saving) return
    setSaving(true)
    try { await onRate(rating === value ? null : value) }
    catch (error) { onError?.(error.message || 'Could not save your rating. Please try again.') }
    finally { setSaving(false) }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch { onError?.('Could not copy this response. Please try again.') }
  }

  return (
    <div className="response-actions" role="group" aria-label="Response actions" aria-busy={saving}>
      <button type="button" aria-label="Like response" title="Like" aria-pressed={rating === 'like'} disabled={saving || !canRate} onClick={() => void rate('like')}><ThumbsUp size={15} /></button>
      <button type="button" aria-label="Dislike response" title="Dislike" aria-pressed={rating === 'dislike'} disabled={saving || !canRate} onClick={() => void rate('dislike')}><ThumbsDown size={15} /></button>
      <button type="button" aria-label={copied ? 'Response copied' : 'Copy response'} title={copied ? 'Copied!' : 'Copy'} onClick={() => void copy()}>{copied ? <Check size={15} /> : <Copy size={15} />}</button>
      {copied && <span role="status">Copied!</span>}
    </div>
  )
}
