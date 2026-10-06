// @vitest-environment jsdom
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import ResponseActions from '../../../frontend/src/components/ResponseActions.jsx'

afterEach(() => { cleanup(); vi.restoreAllMocks() })
describe('Response actions', () => {
  it('switches or clears the saved rating', async () => {
    const onRate = vi.fn().mockResolvedValue(undefined)
    const { rerender } = render(<ResponseActions text="Answer" rating="like" onRate={onRate} />)
    expect(screen.getByRole('button', { name: 'Like response' }).getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: 'Dislike response' }))
    await waitFor(() => expect(onRate).toHaveBeenCalledWith('dislike'))
    await waitFor(() => expect(screen.getByRole('group').getAttribute('aria-busy')).toBe('false'))
    rerender(<ResponseActions text="Answer" rating="dislike" onRate={onRate} />)
    fireEvent.click(screen.getByRole('button', { name: 'Dislike response' }))
    await waitFor(() => expect(onRate).toHaveBeenLastCalledWith(null))
  })
  it('copies the whole response and confirms success', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    render(<ResponseActions text="Answer with citation [1]." />)
    fireEvent.click(screen.getByRole('button', { name: 'Copy response' }))
    await waitFor(() => expect(screen.getByRole('status').textContent).toBe('Copied!'))
    expect(writeText).toHaveBeenCalledWith('Answer with citation [1].')
  })
  it('reports failed saves without changing the selected rating', async () => {
    const onError = vi.fn()
    render(<ResponseActions text="Answer" rating="like" onRate={() => Promise.reject(new Error('Save failed'))} onError={onError} />)
    fireEvent.click(screen.getByRole('button', { name: 'Dislike response' }))
    await waitFor(() => expect(onError).toHaveBeenCalledWith('Save failed'))
    expect(screen.getByRole('button', { name: 'Like response' }).getAttribute('aria-pressed')).toBe('true')
  })
})
