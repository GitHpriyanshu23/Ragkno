import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearApiSession, getCurrentUser, ingestFiles, ingestUrl, loginWithPassword, queryRAG, queryRAGStream, registerUser } from './api.js'

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } })
}

describe('API client security and streaming', () => {
  beforeEach(() => {
    clearApiSession()
    global.fetch = vi.fn()
  })

  afterEach(() => vi.restoreAllMocks())

  it('ends the session check when the server never responds', async () => {
    vi.useFakeTimers()
    try {
      fetch.mockImplementation((_url, { signal }) => new Promise((_resolve, reject) => {
        signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
      }))
      const request = getCurrentUser()
      const assertion = expect(request).rejects.toMatchObject({ name: 'AbortError' })
      await vi.advanceTimersByTimeAsync(15000)
      await assertion
      expect(fetch).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('shows file extraction errors instead of reporting a successful upload', async () => {
    fetch.mockResolvedValue(jsonResponse({ ok: false, source_count: 0, sources: [
      { display_name: 'scan.pdf', error: 'File is empty or contains no extractable text' },
    ] }))
    await expect(ingestFiles([new File(['scan'], 'scan.pdf', { type: 'application/pdf' })]))
      .rejects.toThrow('scan.pdf: File is empty or contains no extractable text')
  })

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

  it('tracks uploaded bytes then indexing while preserving authenticated requests', async () => {
    fetch.mockResolvedValue(jsonResponse({ csrf_token: 'upload-csrf' }))
    await getCurrentUser()
    const xhr = { upload: {}, open: vi.fn(), setRequestHeader: vi.fn(), send: vi.fn(), status: 200, responseText: JSON.stringify({ ok: true, source_count: 1 }) }
    const original = globalThis.XMLHttpRequest
    globalThis.XMLHttpRequest = class { constructor() { return xhr } }
    try {
      const onProgress = vi.fn()
      const request = ingestFiles([new File(['text'], 'notes.txt')], { onProgress })
      expect(xhr.withCredentials).toBe(true)
      expect(xhr.setRequestHeader).toHaveBeenCalledWith('X-CSRF-Token', 'upload-csrf')
      xhr.upload.onprogress({ lengthComputable: true, loaded: 88, total: 100 })
      expect(onProgress).toHaveBeenLastCalledWith({ stage: 'uploading', percent: 88 })
      xhr.upload.onload()
      expect(onProgress).toHaveBeenLastCalledWith({ stage: 'indexing', percent: 100 })
      xhr.onload()
      await expect(request).resolves.toMatchObject({ source_count: 1 })
    } finally { globalThis.XMLHttpRequest = original }
  })

  it.each([
    [524, 'request timed out (HTTP 524)'],
    [502, 'backend or proxy failed (HTTP 502)'],
    [413, 'request too large (HTTP 413)'],
    [200, 'unexpected response (HTTP 200)'],
  ])('preserves HTTP %s when the upload response is HTML instead of JSON', async (status, message) => {
    const xhr = { upload: {}, open: vi.fn(), setRequestHeader: vi.fn(), send: vi.fn(), status, responseText: '<html>Error</html>' }
    vi.stubGlobal('XMLHttpRequest', class { constructor() { return xhr } })
    try {
      const request = ingestFiles([new File(['text'], 'notes.txt')], { onProgress: vi.fn() })
      xhr.onload()
      await expect(request).rejects.toThrow(message)
      expect(xhr.send).toHaveBeenCalledTimes(1)
    } finally { vi.unstubAllGlobals() }
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
