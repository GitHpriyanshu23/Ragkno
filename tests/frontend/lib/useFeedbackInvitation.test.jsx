// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import useFeedbackInvitation from '../../../frontend/src/lib/useFeedbackInvitation.js'

beforeEach(() => { localStorage.clear(); vi.useFakeTimers() })
afterEach(() => { cleanup(); vi.useRealTimers() })

it('waits one minute after the first successful answer, without resetting', () => {
  const onOpen = vi.fn()
  const { result } = renderHook(() => useFeedbackInvitation('alice', { open: false, blocked: false, onOpen }))
  act(() => vi.advanceTimersByTime(60000))
  expect(onOpen).not.toHaveBeenCalled()
  act(() => result.current())
  act(() => vi.advanceTimersByTime(30000))
  act(() => result.current())
  act(() => vi.advanceTimersByTime(29999))
  expect(onOpen).not.toHaveBeenCalled()
  act(() => vi.advanceTimersByTime(1))
  expect(onOpen).toHaveBeenCalledTimes(1)
})

it('resumes on remount and only invites once per account in the browser', () => {
  const onOpen = vi.fn()
  const first = renderHook(() => useFeedbackInvitation('alice', { open: false, blocked: false, onOpen }))
  act(() => first.result.current())
  act(() => vi.advanceTimersByTime(20000))
  first.unmount()
  const second = renderHook(() => useFeedbackInvitation('alice', { open: false, blocked: false, onOpen }))
  act(() => vi.advanceTimersByTime(40000))
  expect(onOpen).toHaveBeenCalledTimes(1)
  second.unmount()
  const third = renderHook(() => useFeedbackInvitation('alice', { open: false, blocked: false, onOpen }))
  act(() => third.result.current())
  act(() => vi.advanceTimersByTime(60000))
  expect(onOpen).toHaveBeenCalledTimes(1)
})

it('defers while busy and keeps accounts separate', () => {
  const onOpen = vi.fn()
  const { result, rerender } = renderHook(({ blocked, user }) => useFeedbackInvitation(user, { open: false, blocked, onOpen }), { initialProps: { blocked: true, user: 'alice' } })
  act(() => result.current())
  act(() => vi.advanceTimersByTime(60000))
  expect(onOpen).not.toHaveBeenCalled()
  rerender({ blocked: false, user: 'alice' })
  act(() => vi.advanceTimersByTime(1))
  expect(onOpen).toHaveBeenCalledTimes(1)
  rerender({ blocked: false, user: 'bob' })
  act(() => result.current())
  act(() => vi.advanceTimersByTime(60000))
  expect(onOpen).toHaveBeenCalledTimes(2)
})

it('suppresses the invitation if feedback was already opened manually', () => {
  const onOpen = vi.fn()
  const { result, rerender } = renderHook(({ open }) => useFeedbackInvitation('alice', { open, blocked: false, onOpen }), { initialProps: { open: false } })
  act(() => result.current())
  rerender({ open: true })
  rerender({ open: false })
  act(() => vi.advanceTimersByTime(60000))
  expect(onOpen).not.toHaveBeenCalled()
})
