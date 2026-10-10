import { useCallback, useEffect, useRef, useState } from 'react'

// Account-scoped browser storage keeps navigation/reloads from resetting the wait.
export default function useFeedbackInvitation(userId, { open, blocked, onOpen }) {
  const key = userId ? `ragkno_feedback_invitation_v1:${userId}` : null
  const [revision, setRevision] = useState(0)
  const memory = useRef({})
  const callback = useRef(onOpen)
  callback.current = onOpen
  const read = useCallback(() => {
    try { return JSON.parse(localStorage.getItem(key) || 'null') || memory.current[key] || {} }
    catch { return memory.current[key] || {} }
  }, [key])
  const write = useCallback((value) => {
    memory.current[key] = value
    try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* Session fallback. */ }
  }, [key])

  const recordSuccessfulAnswer = useCallback(() => {
    if (!key) return
    const state = read()
    if (state.shown || state.dueAt) return
    write({ dueAt: Date.now() + 60000 })
    setRevision((value) => value + 1)
  }, [key, read, write])

  useEffect(() => {
    if (!key) return undefined
    if (open) {
      write({ shown: true })
      return undefined
    }
    const state = read()
    if (state.shown || !state.dueAt || blocked) return undefined
    const timer = setTimeout(() => {
      // Recheck storage in case another tab already displayed the invitation.
      if (read().shown) return
      write({ shown: true })
      callback.current()
    }, Math.max(0, state.dueAt - Date.now()))
    return () => clearTimeout(timer)
  }, [key, open, blocked, revision, read, write])

  return recordSuccessfulAnswer
}
