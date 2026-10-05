// @vitest-environment jsdom
import React from 'react'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import UploadProgress from './UploadProgress.jsx'

afterEach(cleanup)
describe('Upload progress', () => {
  it('shows measured upload progress and indeterminate indexing separately', () => {
    const upload = { name: 'Resume.pdf', size: 6081740, stage: 'uploading', percent: 88 }
    const { rerender } = render(<UploadProgress upload={upload} />)
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('88')
    expect(screen.getByText('88%')).toBeTruthy()
    rerender(<UploadProgress upload={{ ...upload, stage: 'indexing', percent: 100 }} />)
    expect(screen.getByRole('progressbar').hasAttribute('aria-valuenow')).toBe(false)
    expect(screen.getByText('Indexing documents…')).toBeTruthy()
    expect(screen.queryByRole('button')).toBeNull()
    rerender(<UploadProgress upload={{ ...upload, stage: 'done', percent: 100 }} />)
    expect(screen.getByText('Ready to ask questions')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Dismiss upload progress' })).toBeTruthy()
  })
})
