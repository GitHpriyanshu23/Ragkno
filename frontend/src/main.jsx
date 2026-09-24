import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { applyTheme } from './components/ChatGPTSettingsModal.jsx'
import { I18nProvider } from './lib/i18n.jsx'

try {
  const savedTheme = localStorage.getItem('ragkno_appearance') || 'system'
  applyTheme(savedTheme)
} catch {}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <I18nProvider>
      <App />
    </I18nProvider>
  </React.StrictMode>,
)
