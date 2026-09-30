import { useLayoutEffect } from 'react'

export const HOMEPAGE_REVEAL_SELECTOR = [
  '.prompt-card',
  '.built-label',
  '.marquee-strip-wrapper',
  '.bento-section .section-head-dark',
  '.bx-card',
  '.hw-head',
  '.hw-card',
  '.chat-demo-section .section-head-light',
  '.use-cases-section .section-head-dark',
  '.workflow-tabs',
  '.workflow-panel',
  '.workflow-mobile-item',
  '.faq-left-col',
  '.faq-right-col',
  '.cta-section > h2',
  '.cta-actions',
  '.site-footer .footer-brand',
  '.site-footer .footer-links',
].join(',')

export default function useHomepageReveal(rootRef) {
  useLayoutEffect(() => {
    const root = rootRef.current
    if (!root) return undefined

    const nodes = [...root.querySelectorAll(HOMEPAGE_REVEAL_SELECTOR)]
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (reduceMotion || !('IntersectionObserver' in window)) {
      return undefined
    }

    const siblingCounts = new Map()
    nodes.forEach((node) => {
      const group = node.parentElement
      const position = siblingCounts.get(group) || 0
      siblingCounts.set(group, position + 1)
      node.classList.add('scroll-reveal')
      node.style.setProperty('--reveal-delay', `${Math.min(position, 5) * 55}ms`)
    })

    const cleanupTimers = []
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return
        const node = entry.target
        const finish = () => {
          node.classList.remove('scroll-reveal', 'is-visible')
          node.style.removeProperty('--reveal-delay')
        }
        node.classList.add('is-visible')
        node.addEventListener('transitionend', finish, { once: true })
        cleanupTimers.push(window.setTimeout(finish, 1000))
        observer.unobserve(node)
      })
    }, {
      rootMargin: '0px 0px -8% 0px',
      threshold: 0.08,
    })

    nodes.forEach((node) => observer.observe(node))
    return () => {
      observer.disconnect()
      cleanupTimers.forEach((timer) => window.clearTimeout(timer))
    }
  }, [rootRef])
}
