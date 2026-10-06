# Project conventions

Keep all project-owned tests in the repository root `tests/` directory:
- Backend tests: `tests/backend/`.
- Frontend tests: `tests/frontend/`, mirroring component/lib folders as useful.
- Shared fixtures and support: `tests/`.

Do not colocate tests with production source files. Update imports and test runner
configuration when moving tests. Run `uv run pytest -q` from the repository root
and `npm test` from `frontend/` to verify test organization.
