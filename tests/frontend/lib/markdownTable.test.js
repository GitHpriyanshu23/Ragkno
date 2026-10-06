import { describe, expect, it } from 'vitest'
import { parseMarkdownTable } from '../../../frontend/src/lib/markdownTable.js'

describe('Markdown table parsing', () => {
  it('parses financial tables and column alignment', () => {
    const table = parseMarkdownTable([
      '| Particulars | FY2024 | FY2025 |',
      '|---|---:|---:|',
      '| Total Income | 45,441.72 | 116,027.54 |',
    ])

    expect(table).toMatchObject({
      headers: ['Particulars', 'FY2024', 'FY2025'],
      alignments: ['left', 'right', 'right'],
      rows: [['Total Income', '45,441.72', '116,027.54']],
      nextIndex: 3,
    })
  })

  it('accepts table pipes escaped by a model', () => {
    const table = parseMarkdownTable([
      '\\| Name \\| Value \\|',
      '\\|---\\|---:\\|',
      '\\| Cash \\| ₹10 \\|',
    ])

    expect(table?.headers).toEqual(['Name', 'Value'])
    expect(table?.rows).toEqual([['Cash', '₹10']])
  })
})
