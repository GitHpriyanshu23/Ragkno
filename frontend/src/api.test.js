import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearApiSession, getCurrentUser, ingestUrl, loginWithPassword, queryRAG, queryRAGStream, registerUser } from './api.js'

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } })
}

describe('API client security and streaming', () => {
  beforeEach(() => {
    clearApiSession()
    global.fetch = vi.fn()
  })

  afterEach(() => vi.restoreAllMocks())

  it('sends cookies and the session CSRF token on mutations', async () => {
    fetch
      .mockResolvedValueOnce(jsonResponse({ authenticated: true, user: { id: 'alice' }, csrf_token: 'csrf-alice' }))
      .mockResolvedValueOnce(jsonResponse({ ok: true }))

    await getCurrentUser()
    await ingestUrl('https://example.com/docs')

    expect(fetch).toHaveBeenNthCalledWith(2, '/ingest/url', expect.objectContaining({ credentials: 'include', method: 'POST' }))
    const headers = fetch.mock.calls[1][1].headers
    expect(headers.get('X-CSRF-Token')).toBe('csrf-alice')
  })

  it('does not automatically retry non-idempotent queries', async () => {
    fetch.mockResolvedValue(jsonResponse({ detail: 'failed' }, 500))
    await expect(queryRAG('question', { threadId: 'thread-1', requestId: 'request-1' })).rejects.toThrow('failed')
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('explains an empty proxy 500 as an unavailable backend', async () => {
    fetch.mockResolvedValue(new Response('', { status: 500 }))

    await expect(loginWithPassword({ email: 'ada@example.com', password: 'StrongPassword1!' }))
      .rejects.toThrow('RagKno server is unavailable. Start the backend and try again.')
  })

  it('rejects a stream that ends without a done event', async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('event: token\ndata: {"token":"partial"}\n\n'))
        controller.close()
      },
    })
    fetch.mockResolvedValue(new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }))
    await expect(queryRAGStream('question', { threadId: 'thread-1', requestId: 'request-1' })).rejects.toMatchObject({ interrupted: true })
  })

  it('finishes as soon as the done event arrives without waiting for connection close', async () => {
    const cancelled = vi.fn()
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(
          'event: token\ndata: {"token":"Finished"}\n\n' +
          'event: done\ndata: {"answer":"Finished","sources":[]}\n\n',
        ))
        // Intentionally leave the connection open, as a keep-alive server or
        // proxy may do after it has sent the terminal SSE event.
      },
      cancel: cancelled,
    })
    fetch.mockResolvedValue(new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }))

    await expect(queryRAGStream('question', { threadId: 'thread-1', requestId: 'request-1' }))
      .resolves.toMatchObject({ answer: 'Finished' })
    expect(cancelled).toHaveBeenCalledOnce()
  })

  it('submits the database-backed registration and password login payloads', async () => {
    fetch
      .mockResolvedValueOnce(jsonResponse({ authenticated: true, user: { id: 'local-1' } }))
      .mockResolvedValueOnce(jsonResponse({ authenticated: true, user: { id: 'local-1' } }))

    await registerUser({ name: 'Ada', email: 'ada@example.com', password: 'StrongPassword1!', termsAccepted: true })
    await loginWithPassword({ email: 'ada@example.com', password: 'StrongPassword1!' })

    expect(fetch.mock.calls[0][0]).toBe('/auth/register')
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toMatchObject({ name: 'Ada', terms_accepted: true })
    expect(fetch.mock.calls[1][0]).toBe('/auth/login')
    expect(fetch.mock.calls[1][1]).toMatchObject({ credentials: 'include', method: 'POST' })
  })
})
