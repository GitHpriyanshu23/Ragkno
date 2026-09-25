import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { I18nProvider } from './lib/i18n.jsx'

// Remove any lingering dark theme classes/attributes and clear theme preference
try {
  document.documentElement.classList.remove('dark')
  document.documentElement.removeAttribute('data-theme')
  localStorage.removeItem('ragkno_appearance')
} catch {}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <I18nProvider>
      <App />
    </I18nProvider>
  </React.StrictMode>,
)
