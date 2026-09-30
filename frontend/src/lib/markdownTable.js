function splitTableRow(line) {
  const normalized = String(line || '').trim().replace(/\\\|/g, '|')
  if (!normalized.includes('|')) return null
  const withoutOuterPipes = normalized.replace(/^\|/, '').replace(/\|$/, '')
  return withoutOuterPipes.split('|').map((cell) => cell.trim())
}

function alignmentFromMarker(marker) {
  const value = String(marker || '').trim()
  if (/^:-{3,}:$/.test(value)) return 'center'
  if (/^-{3,}:$/.test(value)) return 'right'
  return 'left'
}

export function parseMarkdownTable(lines, startIndex = 0) {
  const headers = splitTableRow(lines[startIndex])
  const separators = splitTableRow(lines[startIndex + 1])
  if (!headers || headers.length < 2 || !separators || separators.length !== headers.length) return null
  if (!separators.every((cell) => /^:?-{3,}:?$/.test(cell))) return null

  const rows = []
  let nextIndex = startIndex + 2
  while (nextIndex < lines.length) {
    if (!String(lines[nextIndex] || '').trim()) break
    const cells = splitTableRow(lines[nextIndex])
    if (!cells || cells.length < 2) break
    rows.push(headers.map((_, cellIndex) => cells[cellIndex] || ''))
    nextIndex += 1
  }

  return {
    headers,
    alignments: separators.map(alignmentFromMarker),
    rows,
    nextIndex,
  }
}
