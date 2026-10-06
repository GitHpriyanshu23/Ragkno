import json
import threading
import time

import pytest

from src.ingestion_jobs import IngestionJobs


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('Job did not finish')


def test_queue_returns_before_processing_and_is_user_scoped(tmp_path):
    started, release = threading.Event(), threading.Event()
    def process(job, directory, progress):
        assert (directory / '0').read_bytes() == b'document'
        progress('Creating searchable embeddings', 45)
        started.set()
        release.wait(3)
        return {'ok': True, 'source_count': 1}
    queue = IngestionJobs(tmp_path, process)
    try:
        job = queue.submit('alice', [('a.txt', 'text/plain', b'document')])
        assert started.wait(2)
        assert queue.get(job['job_id'], 'alice')['percent'] == 45
        with pytest.raises(KeyError):
            queue.get(job['job_id'], 'bob')
        assert queue.active('bob') == []
        with pytest.raises(ValueError):
            queue.submit('alice', [('b.txt', 'text/plain', b'new')])
        release.set()
        wait_until(lambda: queue.get(job['job_id'], 'alice')['status'] == 'completed')
        wait_until(lambda: not (tmp_path / job['job_id'] / '0').exists())
    finally:
        release.set()
        queue.stop()


def test_restart_resumes_interrupted_job(tmp_path):
    directory = tmp_path / 'abcdef'
    directory.mkdir()
    (directory / '0').write_bytes(b'persisted')
    (directory / 'status.json').write_text(json.dumps({
        'job_id': 'abcdef', 'user_id': 'alice', 'status': 'running',
        'created_at': time.time(), 'updated_at': time.time(), 'inputs': [{'file': '0'}],
    }))
    queue = IngestionJobs(tmp_path, lambda job, directory, progress: {'ok': True, 'source_count': 1})
    try:
        queue.start()
        wait_until(lambda: queue.get('abcdef', 'alice')['status'] == 'completed')
    finally:
        queue.stop()


def test_failed_job_has_safe_error_and_removes_upload(tmp_path):
    def process(*args):
        raise RuntimeError('private internal details')
    queue = IngestionJobs(tmp_path, process)
    try:
        job = queue.submit('alice', [('a.txt', 'text/plain', b'data')])
        wait_until(lambda: queue.get(job['job_id'], 'alice')['status'] == 'failed')
        assert 'private internal details' not in queue.get(job['job_id'], 'alice')['error']
        wait_until(lambda: not (tmp_path / job['job_id'] / '0').exists())
    finally:
        queue.stop()
