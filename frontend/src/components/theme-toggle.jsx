import { Moon, Sun } from 'lucide-react'
import { Switch } from './ui/switch'

export default function ThemeToggle({ checked, onCheckedChange }) {
  return (
    <div className="theme-toggle" role="group" aria-label="Toggle color theme">
      <Sun size={14} className={!checked ? 'active' : ''} />
      <Switch checked={checked} onCheckedChange={onCheckedChange} aria-label="Theme toggle" />
      <Moon size={14} className={checked ? 'active' : ''} />
    </div>
  )
}
