# Contributing to RagKno

Thank you for improving RagKno. Contributions are welcome as focused issues, documentation fixes, bug fixes, tests, and product changes.

## Development setup

1. Fork and clone the repository.
2. Create a branch from `main`, for example `feat/source-filters` or `fix/stream-timeout`.
3. Copy `.env.example` to `.env` and use development credentials only.
4. Install dependencies with `uv sync --dev` and `npm --prefix frontend ci`.
5. Run FastAPI on port `8000` and Vite on port `5173` as described in the README.

Never commit secrets, uploaded documents, database files, or vector indexes. Use synthetic or public fixtures in tests.

## Changes and commits

Keep each pull request small enough to review. Use clear imperative commit messages, preferably Conventional Commits:

```text
feat(chat): render structured answers as tables
fix(retrieval): stop the stream after the final event
docs(deploy): document Cloudflare proxy variables
test(auth): cover expired session cookies
```

## Validation

Run the checks related to your change. Before opening a pull request, run the complete baseline:

```bash
uv lock --check
uv run pytest -q
uv sync --group evaluation
uv run python evaluation/ragas_eval.py --help
npm --prefix frontend test
npm --prefix frontend run build
git diff --check
```

Include meaningful tests for behavior changes. UI pull requests should include before/after screenshots for visible changes.

## Pull requests

Explain the problem, the resulting behavior, and the checks you ran. Link the related issue when one exists. A maintainer may ask for a smaller scope or changes before merge. By contributing, you agree that your contribution is licensed under Apache License 2.0.

For vulnerabilities, do not open a public issue; follow [SECURITY.md](SECURITY.md).
