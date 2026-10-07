// @vitest-environment jsdom
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import AskAI, { RAGKNO_EXPLAIN_PROMPT } from '../../../frontend/src/components/AskAI.jsx'

afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('Ask AI footer links', () => {
  it('encodes the complete prompt for prefill links and uses a copy fallback for Gemini', () => {
    render(<AskAI />)
    for (const name of ['ChatGPT', 'Claude', 'Grok']) {
      const link = screen.getByRole('link', { name: `Ask ${name}` })
      expect(new URL(link.href).searchParams.get('q')).toBe(RAGKNO_EXPLAIN_PROMPT)
      expect(link.target).toBe('_blank')
      expect(link.rel).toContain('noopener')
    }
    expect(screen.getByRole('link', { name: 'Ask Gemini' }).href).toBe('https://gemini.google.com/app')
  })

  it('copies the prompt for a visitor to paste into Gemini', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    render(<AskAI />)
    // Prevent jsdom navigation while preserving the actual browser link behavior.
    const link = screen.getByRole('link', { name: 'Ask Gemini' })
    link.addEventListener('click', (event) => event.preventDefault())
    fireEvent.click(link)
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(RAGKNO_EXPLAIN_PROMPT))
    expect(screen.getByRole('status').textContent).toContain('Paste it in Gemini')
  })

  it('provides selectable text when clipboard access is denied', async () => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: vi.fn().mockRejectedValue(new Error('denied')) } })
    render(<AskAI />)
    fireEvent.click(screen.getByRole('button', { name: 'Copy prompt' }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'RagKno explanation prompt' }).value).toBe(RAGKNO_EXPLAIN_PROMPT))
  })
})
