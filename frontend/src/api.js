// src/api.js — all backend calls go through here

const BASE = ''   // dev proxy via vite.config.js forwards to :8000

const DEFAULT_RETRIES = 2
const DEFAULT_RETRY_DELAY_MS = 350

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

async function parseError(res, fallbackMessage) {
  const err = await res.json().catch(() => ({}))
  return err.detail || fallbackMessage
}

async function fetchWithRetry(url, options = {}, config = {}) {
  const retries = Number(config.retries ?? DEFAULT_RETRIES)
  const retryDelayMs = Number(config.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS)

  let lastError = null
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const res = await fetch(url, options)

      if (res.ok) {
        return res
      }

      const shouldRetryStatus = res.status >= 500
      if (shouldRetryStatus && attempt < retries) {
        const delay = retryDelayMs * (2 ** attempt)
        await sleep(delay)
        continue
      }

      lastError = new Error(await parseError(res, `Request failed (${res.status})`))
      throw lastError
    } catch (error) {
      lastError = error
      if (attempt >= retries) {
        throw lastError
      }
      const delay = retryDelayMs * (2 ** attempt)
      await sleep(delay)
    }
  }

  throw lastError || new Error('Request failed')
}

export async function getAuthUrl() {
  const res = await fetchWithRetry(`${BASE}/auth/url`)
  if (!res.ok) throw new Error('Failed to get auth URL')
  return res.json()   // { url: string }
}

export async function getAuthStatus() {
  const res = await fetchWithRetry(`${BASE}/auth/status`)
  if (!res.ok) throw new Error('Failed to check auth status')
  return res.json()   // { connected: bool }
}

export async function getDriveFiles() {
  const res = await fetchWithRetry(`${BASE}/drive/files`)
  if (!res.ok) {
    throw new Error(await parseError(res, 'Failed to list Drive files'))
  }
  return res.json()   // { files: [{id, name, mimeType}] }
}

export async function syncDrive(fileIds = null) {
  const body = fileIds ? JSON.stringify({ file_ids: fileIds }) : undefined
  const res = await fetchWithRetry(`${BASE}/drive/sync`, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body,
  })
  if (!res.ok) {
    throw new Error(await parseError(res, 'Sync failed'))
  }
  return res.json()   // { message, count }
}

export async function disconnectDrive() {
  const res = await fetchWithRetry(`${BASE}/drive/disconnect`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Disconnect failed')
  return res.json()
}

export async function ingestFiles(files) {
  const form = new FormData()
  files.forEach((file) => form.append('files', file))

  const res = await fetchWithRetry(`${BASE}/ingest/files`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    throw new Error(await parseError(res, 'File upload failed'))
  }
  return res.json()
}

export async function ingestUrl(url) {
  const res = await fetchWithRetry(`${BASE}/ingest/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res, 'URL ingestion failed'))
  }
  return res.json()
}

export async function getIndexedSources() {
  const res = await fetchWithRetry(`${BASE}/ingest/sources`)
  if (!res.ok) {
    throw new Error(await parseError(res, 'Failed to load indexed sources'))
  }
  return res.json()   // { count, sources: [{key, source, type}] }
}

export async function unindexSource(key) {
  const res = await fetchWithRetry(`${BASE}/ingest/unindex`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res, 'Failed to unindex source'))
  }
  return res.json()   // { ok, removed, message }
}

export async function queryRAG(query, top_k = 3, sessionId = null) {
  const res = await fetchWithRetry(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k, session_id: sessionId }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res, 'Query failed'))
  }
  return res.json()   // { answer, query, sources }
}

function parseSSEBlock(block) {
  const lines = block.split('\n')
  let event = 'message'
  const dataLines = []

  for (const line of lines) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
      continue
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  if (dataLines.length === 0) {
    return null
  }

  const rawData = dataLines.join('\n')
  let data = rawData
  try {
    data = JSON.parse(rawData)
  } catch {
    // Keep raw string payload if not JSON.
  }

  return { event, data }
}

export async function queryRAGStream(
  query,
  top_k = 3,
  sessionId = null,
  handlers = {},
) {
  const { onMeta, onToken, onDone, onError, signal } = handlers

  const res = await fetchWithRetry(`${BASE}/query/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ query, top_k, session_id: sessionId }),
    signal,
  })

  if (!res.ok || !res.body) {
    throw new Error('Streaming is unavailable.')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let donePayload = null

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''

    for (const rawEvent of events) {
      const parsed = parseSSEBlock(rawEvent.replace(/\r/g, ''))
      if (!parsed) continue

      const { event, data } = parsed
      if (event === 'meta') {
        onMeta?.(data)
      } else if (event === 'token') {
        onToken?.(data?.token || '')
      } else if (event === 'done') {
        donePayload = data
        onDone?.(data)
      } else if (event === 'error') {
        const message = data?.message || 'Streaming failed'
        onError?.(message)
        throw new Error(message)
      }
    }
  }

  return donePayload
}

export async function resetChatMemory(sessionId) {
  const res = await fetchWithRetry(`${BASE}/memory/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  if (!res.ok) {
    throw new Error(await parseError(res, 'Failed to reset chat memory'))
  }
  return res.json()
}
