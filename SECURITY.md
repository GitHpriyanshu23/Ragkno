# Security Policy

## Supported version

Security fixes target the current `main` branch and the latest published RagKno release.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **Security → Report a vulnerability** flow for this repository. Include the affected route or component, reproduction steps, impact, and any suggested mitigation. Do not include real API keys, user documents, OAuth tokens, or database contents.

Please allow the maintainers time to confirm and fix the issue before public disclosure. Public issues and discussions are appropriate for ordinary bugs that do not expose data, credentials, authentication, authorization, or service availability.

## Deployment responsibility

Self-hosters are responsible for TLS, secret rotation, database and vector-store backups, access controls, retention rules, OAuth configuration, and keeping dependencies current. Use a unique `RAGKNO_SESSION_SECRET`, private database credentials, and persistent storage that is not publicly readable.
