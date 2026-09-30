// @vitest-environment jsdom
import { useRef } from 'react'
import { render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import useHomepageReveal from './useHomepageReveal.js'

function Fixture() {
  const rootRef = useRef(null)
  useHomepageReveal(rootRef)
  return <div ref={rootRef}><article className="bx-card">Card</article></div>
}

describe('homepage scroll reveal', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('reveals an observed card once it enters the viewport', () => {
    class ImmediateObserver {
      constructor(callback) { this.callback = callback }
      observe(node) { this.callback([{ target: node, isIntersecting: true }]) }
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal('IntersectionObserver', ImmediateObserver)
    vi.stubGlobal('matchMedia', () => ({ matches: false }))

    const { getByText } = render(<Fixture />)
    expect(getByText('Card').classList.contains('scroll-reveal')).toBe(true)
    expect(getByText('Card').classList.contains('is-visible')).toBe(true)
  })
})
