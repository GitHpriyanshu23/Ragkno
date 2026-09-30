// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import DocsPage from './DocsPage.jsx'

describe('documentation page', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('renders product documentation and filters its navigation', () => {
    class Observer {
      observe() {}
      disconnect() {}
    }
    vi.stubGlobal('IntersectionObserver', Observer)

    render(<MemoryRouter><DocsPage /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: 'Introduction' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Turn scattered knowledge into grounded answers' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Contributing' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Code of Conduct' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Documentation' })).toBeTruthy()

    fireEvent.change(screen.getByRole('searchbox', { name: 'Search documentation' }), { target: { value: 'Google Drive' } })
    expect(screen.getByRole('link', { name: 'Google Drive' })).toBeTruthy()
    expect(screen.queryByRole('link', { name: 'Quickstart' })).toBeNull()
  })
})
