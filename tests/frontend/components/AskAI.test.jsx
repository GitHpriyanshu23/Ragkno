// @vitest-environment jsdom
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import AskAI, { RAGKNO_EXPLAIN_PROMPT } from '../../../frontend/src/components/AskAI.jsx'

afterEach(cleanup)

describe('Ask AI footer links', () => {
  it('opens four providers with the complete encoded prompt', () => {
    render(<AskAI />)
    const hosts = ['chatgpt.com', 'claude.ai', 'grok.com', 'www.perplexity.ai']
    for (const [index, name] of ['ChatGPT', 'Claude', 'Grok', 'Perplexity'].entries()) {
      const link = screen.getByRole('link', { name: `Ask ${name} about Ragkno` })
      const url = new URL(link.href)
      expect(url.hostname).toBe(hosts[index])
      expect(url.searchParams.get('q')).toBe(RAGKNO_EXPLAIN_PROMPT)
      expect(link.target).toBe('_blank')
      expect(link.rel).toContain('noopener')
      expect(link.textContent).toBe('Ask')
      expect(link.querySelector('img')).toBeTruthy()
    }
    expect(screen.queryByRole('link', { name: /Gemini/ })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Copy prompt' })).toBeNull()
    expect(screen.queryByRole('status')).toBeNull()
    expect(screen.getByText('Ask AI about Ragkno')).toBeTruthy()
  })
})
