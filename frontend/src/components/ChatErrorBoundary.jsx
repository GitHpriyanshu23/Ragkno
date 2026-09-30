import { Component } from 'react'
import { RefreshCw } from 'lucide-react'

export default class ChatErrorBoundary extends Component {
  state = { failed: false }

  static getDerivedStateFromError() { return { failed: true } }

  componentDidCatch(error) { console.error('Chat workspace failed to render', error) }

  recover = () => {
    this.setState({ failed: false })
    this.props.onRecover?.()
  }

  render() {
    if (this.state.failed) return <main className="chat-recovery" role="alert"><div><span className="chat-recovery-kicker">Workspace recovery</span><h1>We couldn’t open this conversation.</h1><p>Your chats and sources are still safe. Reload the workspace to try again.</p><button type="button" onClick={this.recover}><RefreshCw size={17} /> Reload workspace</button></div></main>
    return this.props.children
  }
}
