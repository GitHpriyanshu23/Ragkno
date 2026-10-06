export const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')
const BASE = API_BASE
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])
let csrfToken = ''

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
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
      const error = new Error(await parseError(response, `Request failed (${response.status})`))
      error.status = response.status
      throw error
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

export async function syncDrive(fileIds = null, options = {}) {
  const job = await jsonRequest('/ingest/jobs/drive', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_ids: fileIds }),
  })
  return watchIngestionJob(job.job_id, options)
}

export function disconnectDrive() { return jsonRequest('/drive/disconnect', { method: 'DELETE' }) }

export async function ingestFiles(files, { onProgress, signal } = {}) {
  const form = new FormData()
  files.forEach((file) => form.append('files', file))
  let result = onProgress ? await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/ingest/jobs/files`)
    xhr.withCredentials = true
    if (csrfToken) xhr.setRequestHeader('X-CSRF-Token', csrfToken)
    const abort = () => xhr.abort()
    const cleanup = () => signal?.removeEventListener('abort', abort)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress({ stage: 'uploading', percent: Math.round(event.loaded / event.total * 100) })
    }
    xhr.upload.onload = () => onProgress({ stage: 'indexing', phase: 'Waiting to index', percent: 0 })
    xhr.onload = () => {
      cleanup()
      let payload
      try { payload = JSON.parse(xhr.responseText) } catch {
        const status = xhr.status
        let message = `Upload server returned an unexpected response (HTTP ${status}).`
        if ([408, 504, 524].includes(status)) {
          message = `Upload/indexing request timed out (HTTP ${status}). The server may still be indexing. Check Data Center before uploading this file again.`
        } else if (status === 413) {
          message = 'Upload rejected by the server or proxy: request too large (HTTP 413). Try uploading fewer files at once.'
        } else if ([500, 502, 503, 520, 521, 522, 523].includes(status)) {
          message = `Upload backend or proxy failed (HTTP ${status}). Check the backend logs for the cause.`
        }
        reject(new Error(message))
        return
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(payload)
      else reject(new Error(typeof payload.detail === 'string' ? payload.detail : payload.detail?.message || `Upload failed (${xhr.status}).`))
    }
    xhr.onerror = () => { cleanup(); reject(new Error('Upload failed. Check your connection and try again.')) }
    xhr.onabort = () => { cleanup(); reject(new DOMException('Upload cancelled', 'AbortError')) }
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) { cleanup(); reject(new DOMException('Upload cancelled', 'AbortError')); return }
    xhr.send(form)
  }) : await jsonRequest('/ingest/jobs/files', { method: 'POST', body: form, signal })
  if (result.job_id) result = await watchIngestionJob(result.job_id, { onProgress, signal })
  if (result.ok === false || result.source_count === 0) {
    const failures = (result.sources || []).filter((source) => source.error)
      .map((source) => `${source.display_name}: ${source.error}`)
    throw new Error(failures.join('; ') || result.message || 'No files were indexed.')
  }
  return result
}

export function getActiveIngestionJobs() { return jsonRequest('/ingest/jobs') }

export async function watchIngestionJob(jobId, { onProgress, signal } = {}) {
  // Short status requests replace the one long request through the proxy.
  // A temporary loss of connectivity does not turn server-side work into a failure.
  while (!signal?.aborted) {
    const controller = new AbortController()
    const abort = () => controller.abort()
    signal?.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(abort, 15000)
    let job
    try {
      job = await jsonRequest(`/ingest/jobs/${encodeURIComponent(jobId)}`, { signal: controller.signal })
    } catch (error) {
      if (signal?.aborted || [401, 403, 404].includes(error.status)) throw error
      onProgress?.({ stage: 'indexing', phase: 'Reconnecting to indexing progress…' })
    } finally {
      clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
    }
    if (job) {
      onProgress?.({ stage: 'indexing', phase: job.phase, percent: job.percent, jobId })
      if (job.status === 'completed') return job.result
      if (job.status === 'failed') {
        const failures = (job.result?.sources || []).filter((source) => source.error)
        throw new Error(job.error || failures.map((source) => `${source.display_name}: ${source.error}`).join('; ') || job.result?.message || 'Indexing failed.')
      }
    }
    await sleep(2000)
  }
  throw new DOMException('Stopped watching indexing; server processing continues.', 'AbortError')
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
