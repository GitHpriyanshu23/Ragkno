"""Durable ingestion queue for a single-host deployment; one worker across processes."""
import fcntl
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)


class IngestionJobs:
    def __init__(self, root, process):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.process = process
        self._started = False
        self._stop = threading.Event()
        self._start_lock = threading.Lock()

    def _write(self, directory, job):
        temporary = directory / (uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(job))
        os.chmod(temporary, 0o600)
        temporary.replace(directory / 'status.json')

    def get(self, job_id, user_id):
        if not str(job_id).isalnum():
            raise KeyError(job_id)
        try:
            job = json.loads((self.root / job_id / 'status.json').read_text())
        except (OSError, ValueError):
            raise KeyError(job_id)
        if job['user_id'] != user_id:
            raise KeyError(job_id)
        return {key: value for key, value in job.items() if key not in {'user_id', 'inputs'}}

    def active(self, user_id):
        jobs = []
        for path in self.root.glob('*/status.json'):
            job = json.loads(path.read_text())
            if job['user_id'] == user_id and job['status'] in {'queued', 'running'}:
                jobs.append(self.get(job['job_id'], user_id))
        return jobs

    def submit(self, user_id, files=(), kind='upload', payload=None):
        # Serialize admission across threads/processes; bound storage and CPU backlog.
        with (self.root / 'admission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = [json.loads(path.read_text()) for path in self.root.glob('*/status.json')]
            pending = [job for job in records if job['status'] in {'queued', 'running'}]
            if any(job['user_id'] == user_id for job in pending):
                raise ValueError('You already have documents processing. Wait for them to finish before adding more.')
            if len(pending) >= 10:
                raise ValueError('The indexing queue is full. Please try again shortly.')
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir(mode=0o700)
            inputs = []
            try:
                for index, (name, content_type, content) in enumerate(files):
                    path = directory / str(index)
                    path.write_bytes(content)
                    os.chmod(path, 0o600)
                    inputs.append({'name': name, 'content_type': content_type, 'file': str(index)})
                job = {'job_id': job_id, 'user_id': user_id, 'kind': kind, 'payload': payload,
                       'inputs': inputs, 'name': ('Google Drive documents' if kind == 'drive' else inputs[0]['name'] if len(inputs) == 1 else f'{len(inputs)} documents'),
                       'size': sum(len(item[2]) for item in files), 'status': 'queued', 'phase': 'Waiting to index',
                       'percent': 0, 'created_at': time.time(), 'updated_at': time.time()}
                self._write(directory, job)
            except Exception:
                shutil.rmtree(directory)
                raise
        self.start()
        return self.get(job_id, user_id)

    def start(self):
        with self._start_lock:
            if self._started:
                return
            self._started = True
            threading.Thread(target=self._run, daemon=True, name='ingestion-worker').start()

    def stop(self):
        self._stop.set()

    def _run(self):
        # Held for the worker lifetime. Another process can take over after a crash.
        with (self.root / 'worker.lock').open('a') as lock:
            while not self._stop.is_set():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self._stop.wait(1)
            if self._stop.is_set():
                return
            for path in self.root.glob('*/status.json'):
                job = json.loads(path.read_text())
                if job['status'] == 'running':
                    job.update(status='queued', phase='Resuming indexing', percent=0)
                    self._write(path.parent, job)
            while not self._stop.is_set():
                paths = sorted(self.root.glob('*/status.json'), key=lambda path: path.parent.name)
                jobs = [(path, json.loads(path.read_text())) for path in paths]
                queued = sorted([(path, job) for path, job in jobs if job['status'] == 'queued'], key=lambda item: item[1]['created_at'])
                for path, job in queued:
                    directory = path.parent
                    def progress(phase, percent):
                        job.update(status='running', phase=phase, percent=int(percent), updated_at=time.time())
                        self._write(directory, job)
                    try:
                        progress('Reading documents', 2)
                        result = self.process(job, directory, progress)
                        failed = result.get('ok') is False or result.get('source_count', result.get('count', 1)) == 0
                        job.update(status='failed' if failed else 'completed', percent=100, result=result,
                                   phase='No indexable text' if failed else 'Ready to ask questions')
                    except Exception:
                        log.exception('Ingestion job %s failed', job['job_id'])
                        job.update(status='failed', error='Document processing failed. Please try again or contact support with job ' + job['job_id'], phase='Indexing failed')
                    finally:
                        job['updated_at'] = time.time()
                        self._write(directory, job)
                        for item in job['inputs']:
                            (directory / item['file']).unlink(missing_ok=True)
                    log.info('Ingestion job %s %s in %.1fs', job['job_id'], job['status'], time.time() - job['created_at'])
                for path, job in jobs:
                    if job['status'] in {'completed', 'failed'}:
                        for item in job['inputs']:
                            (path.parent / item['file']).unlink(missing_ok=True)
                        if time.time() - job['updated_at'] > 7 * 86400:
                            shutil.rmtree(path.parent, ignore_errors=True)
                self._stop.wait(1)
