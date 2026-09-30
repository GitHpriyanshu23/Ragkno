import { Link } from 'react-router-dom'
import SiteFooter from './SiteFooter.jsx'

const updated = 'September 29, 2026'

const content = {
  privacy: {
    title: 'Privacy Policy',
    intro: 'Your privacy matters to us. This policy explains what RagKno collects, how that information is used, and the choices available to you.',
    sections: [
      {
        title: 'Information we collect',
        body: ['We collect the information needed to create and secure your account and to provide a private document-question-answering workspace.'],
        items: ['Account details such as your name, normalized email address, authentication provider, and securely hashed password when local login is used.', 'Documents, static URLs, Google Drive selections, extracted passages, source metadata, and ingestion status that you add to your workspace.', 'Questions, conversation history, retrieved source references, feedback, and non-sensitive interface preferences.', 'Basic request and diagnostic information used to protect, operate, and troubleshoot the service.'],
      },
      {
        title: 'How we use information',
        body: ['We use this information to authenticate sessions, ingest sources, retrieve relevant passages, generate cited answers, restore conversation history, prevent abuse, and improve reliability. RagKno does not sell personal information.'],
      },
      {
        title: 'Models and service providers',
        body: ['Relevant questions and source excerpts may be sent to the model and infrastructure providers configured for the deployment. Those providers process information to deliver the requested service under their own terms and privacy commitments.'],
      },
      {
        title: 'Google Drive',
        body: ['If you connect Google Drive, RagKno uses the permission you grant to list and ingest the documents you select. Drive credentials and file identifiers are kept for that connection and are not used to present unrelated workspace integrations. You can disconnect Drive from the product.'],
      },
      {
        title: 'Retention, deletion, and security',
        body: ['Chats and indexed sources remain available until you delete them or the account is removed. Operational records may be retained for a limited period where required for security, recovery, or legal obligations.', 'We use reasonable access controls, scoped storage, secure session cookies, CSRF protection, and transport security. No online service can guarantee absolute security.'],
      },
      {
        title: 'Your choices and contact',
        body: ['You may request access, correction, export, or deletion of your account information where applicable. Until self-service account deletion is available, contact your RagKno administrator. A production contact address will be added before launch.'],
      },
    ],
  },
  terms: {
    title: 'Terms of Service',
    intro: 'These terms describe the rules for using RagKno’s document-grounded answer workspace and the responsibilities that come with an account.',
    sections: [
      {
        title: 'Using RagKno',
        body: ['You may use RagKno only in accordance with applicable law and these terms. You must be able to authorize every document, URL, or connected Drive file that you add to the service.'],
      },
      {
        title: 'Accounts and access',
        body: ['Provide accurate registration information, protect your credentials, and promptly report suspected unauthorized access. You are responsible for activity performed through your account.'],
      },
      {
        title: 'Your content',
        body: ['You retain ownership of your source material and questions. You grant RagKno the limited permission needed to store, process, retrieve, and present that material back to you as part of the service.'],
      },
      {
        title: 'Generated answers',
        body: ['RagKno retrieves source passages and uses configured models to generate responses with citations. Answers may still be incomplete or incorrect. Review the cited source before relying on an answer, especially for legal, medical, financial, or other consequential decisions.'],
      },
      {
        title: 'Acceptable use',
        body: ['Do not probe other users’ data, bypass security controls, upload malicious content, disrupt the service, automate abusive traffic, or use RagKno to violate another person’s rights.'],
      },
      {
        title: 'Availability, suspension, and changes',
        body: ['Features may change as the product develops. We may limit or suspend access where needed to protect users or the service. Material updates to these terms will be posted here with a revised date.'],
      },
    ],
  },
  cookies: {
    title: 'Cookie Policy',
    intro: 'This policy explains the limited browser storage RagKno uses to keep accounts secure and remember non-sensitive interface choices.',
    sections: [
      {
        title: 'Essential session cookies',
        body: ['After authentication, RagKno sets a signed, HTTP-only session cookie. It keeps you signed in and allows protected workspace requests to be associated with the correct account.'],
      },
      {
        title: 'Security tokens',
        body: ['Authenticated sessions receive CSRF protection for account and workspace changes. These controls are required for the service to operate safely and cannot be disabled while remaining signed in.'],
      },
      {
        title: 'Local preferences',
        body: ['The interface may store non-sensitive display preferences in your browser. Chat messages, documents, source metadata, and answers are stored in the account-scoped backend rather than shared browser storage.'],
      },
      {
        title: 'Third-party authentication',
        body: ['Choosing Google sign-in takes you to Google’s authentication service. Google may use its own cookies under its policies; RagKno does not control those cookies.'],
      },
      {
        title: 'Your choices',
        body: ['Signing out clears the active RagKno session. You can remove local preferences through your browser settings. Blocking essential cookies prevents authenticated workspace features from working.'],
      },
      {
        title: 'Updates and contact',
        body: ['We will update this page if the browser storage used by RagKno materially changes. A production contact address will be added before launch.'],
      },
    ],
  },
}

export default function LegalPage({ kind }) {
  const page = content[kind] || content.privacy

  return (
    <div className="legal-page">
      <article className="legal-article">
        <header className="legal-hero">
          <p className="legal-kicker">RagKno legal</p>
          <h1>{page.title}</h1>
          <p className="legal-meta">Last updated {updated} · Starter policy for legal review before production launch</p>
          <p className="legal-intro">{page.intro}</p>
        </header>

        <div className="legal-sections">
          {page.sections.map((section, index) => (
            <section key={section.title}>
              <h2>{index + 1}. {section.title}</h2>
              {section.body.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
              {section.items && <ul>{section.items.map((item) => <li key={item}>{item}</li>)}</ul>}
            </section>
          ))}
        </div>

        <aside className="legal-contact">
          <div>
            <p className="legal-contact-label">Questions about this page?</p>
            <p>Contact your RagKno product administrator.</p>
          </div>
          <nav aria-label="Legal policies">
            <Link to="/privacy">Privacy</Link>
            <Link to="/terms">Terms</Link>
            <Link to="/cookies">Cookies</Link>
          </nav>
        </aside>
      </article>
      <SiteFooter />
    </div>
  )
}
