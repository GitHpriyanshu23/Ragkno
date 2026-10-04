import { useState } from 'react'
import { API_BASE } from '../api.js'

function AvatarImage({ sources, initial, displayName, className }) {
  const [attempt, setAttempt] = useState(0)
  if (attempt >= sources.length) return <span className={className}>{initial}</span>
  return (
    <img
      key={sources[attempt]}
      src={sources[attempt]}
      alt={displayName || 'Profile'}
      referrerPolicy="no-referrer"
      loading="eager"
      className={className}
      onError={() => setAttempt((prev) => prev + 1)}
    />
  )
}

export default function UserAvatar({ user, displayName, className = ' ' }) {
  const initial = (displayName || user?.name || user?.email || 'U').trim().slice(0, 1).toUpperCase()
  const sources = []
  if (user?.picture) {
    // Bind the request to this displayed profile, even if another window has
    // since changed the browser's shared session cookie.
    sources.push(`${API_BASE}/auth/avatar?url=${encodeURIComponent(user.picture)}`, user.picture)
  } else if (user?.avatar_url && !/\/auth\/avatar(?:[?#]|$)/.test(user.avatar_url)) {
    sources.push(user.avatar_url.startsWith('http') ? user.avatar_url : `${API_BASE}${user.avatar_url}`)
  }
  return (
    <AvatarImage
      key={JSON.stringify([user?.id || user?.email, ...sources])}
      sources={sources}
      initial={initial}
      displayName={displayName}
      className={className}
    />
  )
}
