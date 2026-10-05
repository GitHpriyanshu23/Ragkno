export const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')
const BASE = API_BASE
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])
let csrfToken = ''

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

async function parseError(response, fallback) {
  const payload = await response.json().catch(() => ({}))
  const detail = payload?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') return detail.message || JSON.stringify(detail)
  if (response.status === 500) {
    return 'RagKno server is unavailable. Start the backend and try again.'
  }
  return fallback
}

async function apiFetch(path, options = {}, config = {}) {
  const method = String(options.method || 'GET').toUpperCase()
  const headers = new Headers(options.headers || {})
  if (!SAFE_METHODS.has(method) && csrfToken) headers.set('X-CSRF-Token', csrfToken)
  const retries = SAFE_METHODS.has(method) ? Number(config.retries ?? 2) : Number(config.retries ?? 0)
  let lastError

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetch(`${BASE}${path}`, {
        ...options,
        method,
        headers,
        credentials: 'include',
      })
      if (response.ok) return response
      if (response.status >= 500 && attempt < retries) {
        await sleep(300 * (2 ** attempt))
        continue
      }
      throw new Error(await parseError(response, `Request failed (${response.status})`))
    } catch (error) {
      lastError = error
      if (error?.name === 'AbortError' || attempt >= retries) throw error
      await sleep(300 * (2 ** attempt))
    }
  }
  throw lastError || new Error('Request failed')
}

async function jsonRequest(path, options = {}, config = {}) {
  const response = await apiFetch(path, options, config)
  return response.json()
}

export function clearApiSession() {
  csrfToken = ''
}

export function getGoogleLoginUrl() {
  return jsonRequest('/app-auth/google/url', {}, { retries: 0 })
}

export async function registerUser({ name, email, password, termsAccepted }) {
  const payload = await jsonRequest('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password, terms_accepted: termsAccepted }),
  })
  return payload
}

export async function loginWithPassword({ email, password }) {
  return jsonRequest('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
}

export async function getCurrentUser() {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 15000)
  try {
    const payload = await jsonRequest('/auth/me', { signal: controller.signal }, { retries: 0 })
    csrfToken = payload?.csrf_token || ''
    return payload
  } finally {
    clearTimeout(timer)
  }
}

export async function logoutUser() {
  const payload = await jsonRequest('/auth/logout', { method: 'POST' })
  clearApiSession()
  return payload
}

export function getAuthUrl() { return jsonRequest('/auth/url') }
export function getAuthStatus() { return jsonRequest('/auth/status') }
export function getDriveFiles() { return jsonRequest('/drive/files') }

export function syncDrive(fileIds = null) {
  return jsonRequest('/drive/sync', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_ids: fileIds }),
  })
}

export function disconnectDrive() { return jsonRequest('/drive/disconnect', { method: 'DELETE' }) }

export async function ingestFiles(files, { onProgress, signal } = {}) {
  const form = new FormData()
  files.forEach((file) => form.append('files', file))
  const result = onProgress ? await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/ingest/files`)
    xhr.withCredentials = true
    if (csrfToken) xhr.setRequestHeader('X-CSRF-Token', csrfToken)
    const abort = () => xhr.abort()
    const cleanup = () => signal?.removeEventListener('abort', abort)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress({ stage: 'uploading', percent: Math.round(event.loaded / event.total * 100) })
    }
    xhr.upload.onload = () => onProgress({ stage: 'indexing', percent: 100 })
    xhr.onload = () => {
      cleanup()
      let payload
      try { payload = JSON.parse(xhr.responseText) } catch { reject(new Error('Invalid response from the upload server.')); return }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload)
      else reject(new Error(typeof payload.detail === 'string' ? payload.detail : payload.detail?.message || `Upload failed (${xhr.status}).`))
    }
    xhr.onerror = () => { cleanup(); reject(new Error('Upload failed. Check your connection and try again.')) }
    xhr.onabort = () => { cleanup(); reject(new DOMException('Upload cancelled', 'AbortError')) }
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) { cleanup(); reject(new DOMException('Upload cancelled', 'AbortError')); return }
    xhr.send(form)
  }) : await jsonRequest('/ingest/files', { method: 'POST', body: form })
  if (result.ok === false || result.source_count === 0) {
    const failures = (result.sources || []).filter((source) => source.error)
      .map((source) => `${source.display_name}: ${source.error}`)
    throw new Error(failures.join('; ') || result.message || 'No files were indexed.')
  }
  return result
}

export function ingestUrl(url) {
  return jsonRequest('/ingest/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
}

export function getIndexedSources() { return jsonRequest('/ingest/sources') }

export function unindexSource(sourceId) {
  return jsonRequest('/ingest/unindex', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_id: sourceId }),
  })
}

export function getThreads() { return jsonRequest('/threads') }

export function createBackendThread(title = 'New Chat', id = null) {
  return jsonRequest('/threads', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, id }),
  })
}

export function getThreadMessages(threadId) {
  return jsonRequest(`/threads/${encodeURIComponent(threadId)}/messages`)
}

export function renameBackendThread(threadId, title) {
  return jsonRequest(`/threads/${encodeURIComponent(threadId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
}

export function deleteBackendThread(threadId) {
  return jsonRequest(`/threads/${encodeURIComponent(threadId)}`, { method: 'DELETE' })
}

function queryBody(query, options = {}) {
  return {
    query,
    thread_id: options.threadId,
    request_id: options.requestId,
    top_k: options.topK ?? 5,
    model: options.model || undefined,
    use_reranker: options.useReranker ?? true,
    language: options.language || 'auto',
    source_ids: options.sourceIds?.length ? options.sourceIds : undefined,
  }
}

export function queryRAG(query, options = {}) {
  return jsonRequest('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(queryBody(query, options)),
    signal: options.signal,
  })
}

function parseSSEBlock(block) {
  const lines = block.split('\n')
  let event = 'message'
  const dataLines = []
  for (const line of lines) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return null
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    throw new Error('The server returned an invalid stream event.')
  }
}

export async function queryRAGStream(query, options = {}, handlers = {}) {
  const response = await apiFetch('/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(queryBody(query, options)),
    signal: handlers.signal || options.signal,
  })
  if (!response.body) throw new Error('Streaming is unavailable.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let donePayload = null
  let receivedText = false

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''
    for (const rawEvent of events) {
      const parsed = parseSSEBlock(rawEvent.replace(/\r/g, ''))
      if (!parsed) continue
      if (parsed.event === 'meta') handlers.onMeta?.(parsed.data)
      if (parsed.event === 'token') {
        receivedText = receivedText || Boolean(parsed.data?.token)
        handlers.onToken?.(parsed.data?.token || '')
      }
      if (parsed.event === 'done') {
        donePayload = parsed.data
        handlers.onDone?.(parsed.data)
        // The SSE event is the protocol-level completion signal. Do not keep
        // the composer locked while waiting for a proxy/server to close its
        // keep-alive connection after the answer is already complete.
        await reader.cancel().catch(() => {})
        return donePayload
      }
      if (parsed.event === 'error') {
        const error = new Error(parsed.data?.message || 'Streaming failed')
        error.interrupted = Boolean(parsed.data?.interrupted || receivedText)
        handlers.onError?.(error)
        throw error
      }
    }
  }
  if (!donePayload) {
    const error = new Error('The response stream ended before completion.')
    error.interrupted = receivedText
    throw error
  }
  return donePayload
}

export function resetChatMemory(sessionId) {
  return jsonRequest('/memory/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
}

export function submitFeedback({ rating, comment, feedback }) {
  return jsonRequest('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rating: rating || 'neutral', feedback: (feedback || comment || '').trim() }),
  })
}

export function fetchFeedbacks(limit = 50) { return jsonRequest(`/feedback?limit=${limit}`) }
