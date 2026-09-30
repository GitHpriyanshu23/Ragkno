import { Link } from 'react-router-dom'
import brandLogoDark from '../assets/figma-logo-mark-dark.svg'

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <Link to="/" className="footer-logo-link">
            <img src={brandLogoDark} alt="" className="footer-logo-img" />
            <span>RAGKNO</span>
          </Link>
          <p className="footer-copyright">© RagKno 2026. All rights reserved.</p>
        </div>

        <div className="footer-links">
          <div className="footer-col">
            <h4>Product</h4>
            <Link to="/#capabilities">Features</Link>
            <Link to="/login?mode=signin">Chat Assistant</Link>
            <Link to="/login?mode=signup">Get started</Link>
          </div>
          <div className="footer-col">
            <h4>Resources</h4>
            <Link to="/docs">Documentation</Link>
            <Link to="/#how-it-works">How it works</Link>
            <Link to="/#faq">FAQ</Link>
            <a href="https://github.com/GitHpriyanshu23/Ragkno" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
          <div className="footer-col">
            <h4>Legal</h4>
            <Link to="/privacy">Privacy Policy</Link>
            <Link to="/terms">Terms of Service</Link>
            <Link to="/cookies">Cookie Policy</Link>
          </div>
          <div className="footer-col">
            <h4>Account</h4>
            <Link to="/login?mode=signup">Create account</Link>
            <Link to="/login?mode=signin">Login</Link>
          </div>
        </div>
      </div>
      <div className="footer-watermark" aria-hidden="true">RagKno</div>
    </footer>
  )
}
