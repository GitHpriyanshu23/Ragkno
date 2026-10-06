// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import UserAvatar from '../../../frontend/src/components/UserAvatar.jsx'

afterEach(cleanup)
const alice = { id: 'alice', name: 'Alice', picture: 'https://lh3.googleusercontent.com/alice', avatar_url: '/auth/avatar' }

it('uses the displayed account picture rather than the shared session avatar URL', () => {
  const { rerender } = render(<UserAvatar user={alice} displayName="Alice" />)
  expect(screen.getByRole('img').getAttribute('src')).toContain(`?url=${encodeURIComponent(alice.picture)}`)
  rerender(<UserAvatar user={{ id: 'bob', avatar_url: '/auth/avatar' }} displayName="Bob" />)
  expect(screen.queryByRole('img')).toBeNull()
  expect(screen.getByText('B')).toBeTruthy()
})

it('resets failed-image fallback when the account changes', () => {
  const { rerender } = render(<UserAvatar user={alice} displayName="Alice" />)
  fireEvent.error(screen.getByRole('img'))
  expect(screen.getByRole('img').getAttribute('src')).toBe(alice.picture)
  fireEvent.error(screen.getByRole('img'))
  expect(screen.getByText('A')).toBeTruthy()
  const bob = { id: 'bob', picture: 'https://lh3.googleusercontent.com/bob' }
  rerender(<UserAvatar user={bob} displayName="Bob" />)
  expect(screen.queryByText('A')).toBeNull()
  expect(screen.getByRole('img').getAttribute('src')).toContain(`?url=${encodeURIComponent(bob.picture)}`)
})
