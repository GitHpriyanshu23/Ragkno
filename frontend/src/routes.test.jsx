// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import LegalPage from './components/LegalPage.jsx'
import ChatErrorBoundary from './components/ChatErrorBoundary.jsx'

describe('route recovery and legal surfaces', () => {
  it('keeps the chat route recoverable instead of blank on a rendering failure', () => {
    expect(ChatErrorBoundary.getDerivedStateFromError(new Error('render failure'))).toEqual({ failed: true })
    const boundary = new ChatErrorBoundary({ onRecover: vi.fn() })
    boundary.state = { failed: true }
    render(boundary.render())
    expect(screen.getByRole('heading', { name: 'We couldn’t open this conversation.' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Reload workspace' })).toBeTruthy()
  })

  it('renders linked privacy, terms, and cookie policies', () => {
    render(<MemoryRouter><LegalPage kind="privacy" /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: 'Privacy Policy' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Terms' }).getAttribute('href')).toBe('/terms')
    expect(screen.getByRole('link', { name: 'Cookies' }).getAttribute('href')).toBe('/cookies')
  })

  it('uses Times New Roman only for primary homepage headings', () => {
    const css = readFileSync(resolve(process.cwd(), 'src/styles/headings.css'), 'utf8')
    expect(css).toContain('--heading-font:"Times New Roman",Times,serif')
    expect(css).toContain('.hero-section h1')
    expect(css).toContain('.faq-root-section .faq-headline')
    expect(css).not.toContain(':where(h1,h2,h3,h4,h5,h6)')
  })
})
