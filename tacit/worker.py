"""The background worker: drains queued sessions batch by batch, one worker at a time.

The Stop hook spawns it detached and returns at once, so Claude Code never waits on a model. A file lock keeps it
single: a second spawn while one is running just exits, and the running worker picks the new lines up on its next pass.
A batch that doesn't settle keeps its session's cursor; the session is skipped for the rest of this run and retried on
the next Stop hook (when it may also see a wider batch).
"""
import fcntl
import logging
import os
import subprocess
import sys

from tacit.llm import LLMError, client_from_env
from tacit.observer import observe

log = logging.getLogger("tacit.worker")


class Lock:
    def __init__(self, path):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
        return self.fd is not None

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)


def drain(store, client, max_batches=50, max_lines=200):
    """Observe pending batches until every session is caught up or held. Returns counts (numbers only)."""
    stats = {"batches": 0, "settled": 0, "held": 0}
    held = set()
    progress = True
    while progress and stats["batches"] < max_batches:
        progress = False
        for session in store.pending_sessions():
            if session in held or stats["batches"] >= max_batches:
                continue
            batch = store.next_batch(session, max_lines=max_lines)
            if batch is None:
                continue
            result = observe(store, batch, client)
            stats["batches"] += 1
            if result.settled:
                stats["settled"] += 1
                progress = True
            else:
                stats["held"] += 1
                held.add(session)
                log.info("batch %s held: %s", batch.batch, ",".join(result.retry_reasons))
    return stats


def run(store, home, client=None):
    """Entry point for `tacit worker`. Exits quietly if another worker holds the lock or no model is configured."""
    with Lock(os.path.join(home, "worker.lock")) as acquired:
        if not acquired:
            log.info("another worker is running")
            return None
        if client is None:
            try:
                client = client_from_env()
            except LLMError as error:
                log.warning("not observing: %s", error)  # batches stay queued until a model is configured
                return None
        stats = drain(store, client)
        log.info("drained: %s", stats)
        return stats


def spawn(home):
    """Start a detached worker that outlives the hook. Output goes to $TACIT_HOME/worker.log."""
    with open(os.path.join(home, "worker.log"), "a") as out:
        subprocess.Popen([sys.executable, "-m", "tacit.cli", "worker"], stdin=subprocess.DEVNULL, stdout=out,
                         stderr=out, start_new_session=True, env={**os.environ, "TACIT_HOME": home})
