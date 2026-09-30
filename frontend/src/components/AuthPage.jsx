import { useEffect, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { ArrowRight, Eye, EyeOff, LoaderCircle } from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { getCurrentUser, getGoogleLoginUrl, loginWithPassword, registerUser } from '../api.js'
import brandLogoDark from '../assets/figma-logo-mark-dark.svg'
import googleLogo from '../assets/google-logo-2025.webp'
import heroImage from '../assets/ragkno-hero.webp'

const initialForm = { name: '', email: '', password: '', termsAccepted: false }

export default function AuthPage({ onUserChange, onToast }) {
  const location = useLocation()
  const navigate = useNavigate()
  const reduceMotion = useReducedMotion()
  const [mode, setMode] = useState(new URLSearchParams(location.search).get('mode') === 'signin' ? 'signin' : 'signup')
  const [form, setForm] = useState(initialForm)
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const signingUp = mode === 'signup'

  useEffect(() => {
    let cancelled = false
    void getCurrentUser().then((result) => {
      if (!cancelled && result.authenticated) onUserChange?.(result.user)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [onUserChange])

  useEffect(() => {
    const nextMode = new URLSearchParams(location.search).get('mode') === 'signin' ? 'signin' : 'signup'
    if (nextMode === mode) return
    setMode(nextMode)
    setError('')
    setForm(initialForm)
    setShowPassword(false)
  }, [location.search, mode])

  function switchMode(next) {
    setMode(next)
    setError('')
    setForm(initialForm)
    setShowPassword(false)
    navigate(`/login?mode=${next === 'signin' ? 'signin' : 'signup'}`)
  }

  async function startGoogleLogin() {
    setLoading(true)
    setError('')
    try {
      const { url } = await getGoogleLoginUrl()
      window.location.assign(url)
    } catch (requestError) {
      setLoading(false)
      setError(requestError.message || 'Unable to start Google sign in.')
    }
  }

  async function submit(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const result = signingUp
        ? await registerUser(form)
        : await loginWithPassword(form)
      const current = await getCurrentUser().catch(() => result)
      onUserChange?.(current.user || result.user)
      onToast?.({ type: 'success', message: signingUp ? 'Your account is ready.' : 'Welcome back.' })
      navigate('/chat', { replace: true })
    } catch (requestError) {
      setError(requestError.message || 'Unable to continue. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="account-page">
      <section className="account-form-panel" aria-labelledby="account-title">
        <header className="account-topbar">
          <Link to="/" className="account-brand"><img src={brandLogoDark} alt="" /><span>RagKno</span></Link>
          <button className="account-top-action" type="button" onClick={() => switchMode(signingUp ? 'signin' : 'signup')}>{signingUp ? 'Sign in' : 'Sign up'}</button>
        </header>
        <div className="account-form-wrap">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              className="account-mode-content"
              key={mode}
              initial={reduceMotion ? false : { opacity: 0, y: 7 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduceMotion ? undefined : { opacity: 0, y: -7 }}
              transition={{ duration: reduceMotion ? 0 : 0.18, ease: [0.22, 1, 0.36, 1] }}
            >
              <h1 id="account-title">{signingUp ? 'Create your account' : 'Welcome back'}</h1>
              <button className="account-google" type="button" onClick={startGoogleLogin} disabled={loading}>
                <img src={googleLogo} alt="" /><span>{signingUp ? 'Sign up with Google' : 'Sign in with Google'}</span>
              </button>
              <div className="account-divider"><span>or</span></div>
              <form className="account-form" onSubmit={submit}>
                {signingUp && <label className="account-field"><span>Name</span><input required autoComplete="name" value={form.name} onChange={(e) => setForm((value) => ({ ...value, name: e.target.value }))} placeholder="Name" /></label>}
                <label className="account-field"><span>Email</span><input required type="email" autoComplete="email" value={form.email} onChange={(e) => setForm((value) => ({ ...value, email: e.target.value }))} placeholder="Email" /></label>
                <label className="account-field"><span>Password</span>
                  <span className="account-password-field"><input required type={showPassword ? 'text' : 'password'} autoComplete={signingUp ? 'new-password' : 'current-password'} value={form.password} onChange={(e) => setForm((value) => ({ ...value, password: e.target.value }))} placeholder={signingUp ? 'Password' : 'Password'} /><button type="button" onClick={() => setShowPassword((value) => !value)} aria-label={showPassword ? 'Hide password' : 'Show password'}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></span>
                </label>
                {signingUp && <label className="account-check"><input required type="checkbox" checked={form.termsAccepted} onChange={(e) => setForm((value) => ({ ...value, termsAccepted: e.target.checked }))} /><span>I agree to the <Link to="/terms">Terms of Service</Link> and <Link to="/privacy">Privacy Policy</Link>.</span></label>}
                {error && <p className="account-error" role="alert">{error}</p>}
                <button className="account-submit" type="submit" disabled={loading}>{loading ? <LoaderCircle className="spin" size={18} /> : <>{signingUp ? 'Create account' : 'Sign in'} <ArrowRight size={18} /></>}</button>
              </form>
              <p className="account-toggle">{signingUp ? 'Already have an account?' : 'New to RagKno?'} <button type="button" onClick={() => switchMode(signingUp ? 'signin' : 'signup')}>{signingUp ? 'Sign in' : 'Create an account'}</button></p>
            </motion.div>
          </AnimatePresence>
        </div>
        <footer className="account-footer"><span>© 2026 RagKno</span><Link to="/privacy">Privacy</Link><Link to="/terms">Terms</Link><Link to="/cookies">Cookies</Link></footer>
      </section>
      <aside className="account-visual" aria-label="RagKno knowledge workspace"><img src={heroImage} alt="" /></aside>
    </main>
  )
}
