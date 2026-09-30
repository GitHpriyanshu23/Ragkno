// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import AuthPage from './AuthPage.jsx'

vi.mock('../api.js', () => ({
  getCurrentUser: vi.fn().mockResolvedValue({ authenticated: false }),
  getGoogleLoginUrl: vi.fn().mockResolvedValue({ url: 'https://example.test/oauth' }),
  loginWithPassword: vi.fn(),
  registerUser: vi.fn(),
}))

afterEach(cleanup)

function RouterState() {
  const location = useLocation()
  const navigate = useNavigate()
  return (
    <>
      <output data-testid="location">{location.pathname}{location.search}</output>
      <button type="button" onClick={() => navigate(-1)}>History back</button>
    </>
  )
}

function renderAuth(initialEntry = '/login?mode=signup') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/login" element={<><AuthPage /><RouterState /></>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('account page modes', () => {
  it('keeps the shell mounted while switching modes and follows browser history', async () => {
    renderAuth()

    expect(screen.getByRole('heading', { name: 'Create your account' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'RagKno' })).toBeTruthy()
    expect(screen.getByTestId('location').textContent).toBe('/login?mode=signup')

    fireEvent.click(screen.getAllByRole('button', { name: 'Sign in' })[0])

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Welcome back' })).toBeTruthy())
    expect(screen.getByTestId('location').textContent).toBe('/login?mode=signin')
    expect(screen.queryByRole('textbox', { name: 'Name' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'History back' }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Create your account' })).toBeTruthy())
    expect(screen.getByTestId('location').textContent).toBe('/login?mode=signup')
  })

  it('keeps accessible labels and legal destinations in signup mode', () => {
    renderAuth()

    expect(screen.getByRole('textbox', { name: 'Name' })).toBeTruthy()
    expect(screen.getByRole('textbox', { name: 'Email' })).toBeTruthy()
    expect(screen.getByLabelText('Password')).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Terms of Service' }).getAttribute('href')).toBe('/terms')
    expect(screen.getByRole('link', { name: 'Privacy Policy' }).getAttribute('href')).toBe('/privacy')
  })
})
