# Tests

All project-owned test files live here:

- `backend/`: Python API, security, ingestion and retrieval tests.
- `frontend/`: Vitest API and React component tests, grouped by component/lib.
- `fixtures/`: shared input documents.
- `conftest.py`: Python import-path support.

Run backend tests from the repository root:

```sh
uv run pytest -q
```

Run frontend tests from `frontend/`:

```sh
npm test
```

For one frontend test, use its path relative to the repository root, for example:

```sh
npm test -- tests/frontend/components/ResponseActions.test.jsx
```

Keep future tests under this directory; do not colocate them with production code.
Evaluation datasets and benchmark scripts remain under `evaluation/`.
