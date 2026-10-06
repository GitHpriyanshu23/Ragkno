
### Background document indexing

Deploy the backend before the new frontend: the frontend now submits uploads to
`POST /ingest/jobs/files` and Drive imports to `POST /ingest/jobs/drive`, then polls
`GET /ingest/jobs/{job_id}`. These return HTTP 202 before parsing/embedding, avoiding
proxy read timeouts caused by waiting for indexing in the upload response. Legacy
synchronous endpoints remain for compatibility.

Jobs and pending upload bytes live in `/data/ingestion_jobs` on `backend_data`.
One worker per host (protected by a process lock) processes jobs in order; queued
or interrupted jobs resume when the backend starts. Raw uploads are deleted after
completion/failure, and job metadata expires after seven days. This queue is for
this single-host deployment, not a distributed multi-host deployment. Keep the
persistent volume when recreating containers. Data Center reconnects to active
jobs when reopened; closing the browser does not cancel server processing.

There is one active job per user and a maximum of ten pending jobs, with uploads
limited to 50MB per file and 100MB per batch. Status percentages measure processing
stages/batches, not time remaining. Failed processing logs include the job ID.
Check backend logs for the underlying exception if a job fails.

The fast path reuses the loaded embedding model and creates overlapping
parent/child chunks without sentence-level model calls. Set
`RAG_SEMANTIC_CHUNKING=true` to opt into the former semantic splitting behavior.
`RAG_EMBEDDING_BATCH_SIZE` defaults to 32. Existing indexed documents are unchanged;
new/reindexed documents use the configured chunking mode.
