"""The durable job queue: claiming, leases, retries, restart recovery, clean shutdown.

The queue functions are exercised directly (with an explicit clock, so nothing sleeps for a
backoff), then the worker around them with a fake job body. Everything runs against a real
file-backed database with one connection per thread, like production would have.
"""
import asyncio
import threading
import time
from datetime import timedelta

import pytest

from src.db.models import UploadJob
from src.db.session import SessionLocal
from src.jobs import queue as q
from src.jobs.worker import JobWorker

from tests.test_api_flow import fresh_state  # noqa: F401  (autouse: clean tables per test)


def _enqueue(db, job_id="j1", max_attempts=3, now=None):
    return q.enqueue(db, job_id=job_id, owner="dr.rao", ip="127.0.0.1", patient_id="P001", filename="r.pdf",
                     ext=".pdf", report_date="2026-01-01", language="en", spool_path=f"/spool/{job_id}.pdf",
                     max_attempts=max_attempts, now=now)


def _status(job_id):
    with SessionLocal() as db:
        row = q.get(db, job_id)
        return row.status, row.attempts, row.stage, row.percent


# ── queue protocol ─────────────────────────────────────────────────────────────

def test_a_job_id_can_only_be_queued_once():
    with SessionLocal() as db:
        _enqueue(db, "dup")
        with pytest.raises(q.DuplicateJob):
            _enqueue(db, "dup")


def test_each_job_is_claimed_by_exactly_one_of_many_concurrent_workers():
    with SessionLocal() as db:
        for i in range(20):
            _enqueue(db, f"c{i}")
    claimed: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(n):
        barrier.wait()
        while True:
            with SessionLocal() as db:
                job = q.claim_next(db, f"w{n}", lease_seconds=60)
            if job is None:
                return
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]
    assert sorted(claimed) == sorted(f"c{i}" for i in range(20))     # all taken, none twice


def test_failed_attempt_is_retried_after_backoff_then_failed_when_attempts_run_out():
    now = q.utcnow()
    with SessionLocal() as db:
        _enqueue(db, "r1", max_attempts=2, now=now)
        job = q.claim_next(db, "w", 60, now=now)
        assert job.attempts == 1
        assert q.fail_or_retry(db, "r1", "w", "boom", retry_base_seconds=10, now=now) == "retry"
        assert q.claim_next(db, "w", 60, now=now) is None                       # still backing off
        assert q.claim_next(db, "w", 60, now=now + timedelta(seconds=9)) is None
        again = q.claim_next(db, "w", 60, now=now + timedelta(seconds=13))      # 10s + up to 25% jitter
        assert again.attempts == 2 and again.is_last_attempt
        assert q.fail_or_retry(db, "r1", "w", "boom again", retry_base_seconds=10, now=now) == "failed"
    assert _status("r1")[0] == "failed"
    with SessionLocal() as db:
        assert q.get(db, "r1").error == "boom again"
        assert q.claim_next(db, "w", 60, now=now + timedelta(hours=1)) is None  # never offered again


def test_permanent_error_fails_immediately_even_with_attempts_left():
    with SessionLocal() as db:
        _enqueue(db, "p1", max_attempts=5)
        q.claim_next(db, "w", 60)
        assert q.fail_or_retry(db, "p1", "w", "unreadable", retry_base_seconds=10, permanent=True) == "failed"


def test_backoff_is_exponential_capped_and_jittered():
    assert q.backoff_seconds(1, 10, jitter=False) == 10
    assert q.backoff_seconds(3, 10, jitter=False) == 40
    assert q.backoff_seconds(20, 10, cap=300, jitter=False) == 300
    assert all(10 <= q.backoff_seconds(1, 10) <= 12.5 for _ in range(50))
    assert q.backoff_seconds(1, 0) == 0                                          # tests/dev: retry at once


def test_expired_lease_is_reclaimed_and_the_old_worker_can_no_longer_change_the_job():
    now = q.utcnow()
    with SessionLocal() as db:
        _enqueue(db, "l1", now=now)
        assert q.claim_next(db, "old", 30, now=now).worker_id == "old"
        assert q.claim_next(db, "new", 30, now=now + timedelta(seconds=10)) is None   # lease still valid
        taken = q.claim_next(db, "new", 30, now=now + timedelta(seconds=31))          # lease expired
        assert taken.worker_id == "new" and taken.attempts == 2
        # The paused worker wakes up: every write it attempts is refused.
        assert q.heartbeat(db, "l1", "old", 30) is False
        assert q.complete(db, "l1", "old") is False
        assert q.fail_or_retry(db, "l1", "old", "x", retry_base_seconds=0) is None
        assert q.complete(db, "l1", "new") is True
    assert _status("l1")[0] == "complete"


def test_job_whose_worker_died_on_the_last_attempt_is_failed_not_left_running():
    now = q.utcnow()
    with SessionLocal() as db:
        _enqueue(db, "d1", max_attempts=1, now=now)
        q.claim_next(db, "w", 30, now=now)
        assert q.reap_exhausted(db, now=now + timedelta(seconds=10)) == []        # lease still valid
        reaped = q.reap_exhausted(db, now=now + timedelta(seconds=31))
        assert [j.id for j in reaped] == ["d1"]
        assert q.claim_next(db, "w2", 30, now=now + timedelta(seconds=40)) is None
    assert _status("d1")[0] == "failed"


def test_releasing_a_job_gives_the_attempt_back():
    with SessionLocal() as db:
        _enqueue(db, "x1")
        q.claim_next(db, "w", 60)
        assert q.release(db, "x1", "w") is True
    assert _status("x1")[:2] == ("queued", 0)


def test_progress_never_moves_backwards_and_only_the_owner_can_report_it():
    with SessionLocal() as db:
        _enqueue(db, "g1")
        q.claim_next(db, "w", 60)
        q.set_progress(db, "g1", "w", "orchestrator_complete", 50)
        q.set_progress(db, "g1", "w", "tool_call", 20)            # late, out-of-order write
        q.set_progress(db, "g1", "intruder", "saving", 95)        # not the owner
    assert _status("g1")[2:] == ("orchestrator_complete", 50)


def test_queue_depth_counts_by_status():
    with SessionLocal() as db:
        _enqueue(db, "a")
        _enqueue(db, "b")
        q.claim_next(db, "w", 60)
        assert q.depth_by_status(db) == {"queued": 1, "running": 1, "complete": 0, "failed": 0}
        assert q.active_count(db) == 2


# ── worker ─────────────────────────────────────────────────────────────────────

def _make_worker(process, **kw):
    finished = []
    failures = []

    def on_failure(job, exc, final):
        failures.append((job.attempts, final, str(exc)))
        return "Processing failed. Reference: test."

    worker = JobWorker(
        session_factory=SessionLocal, process=process, on_failure=on_failure,
        on_finished=lambda job, status, err: finished.append((job.id, status, err)),
        concurrency=kw.pop("concurrency", 1), poll_seconds=0.02, retry_base_seconds=0, **kw)
    return worker, finished, failures


async def _until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached in time")


def test_worker_runs_a_job_to_completion():
    async def body():
        async def process(job, progress):
            progress("saving", 95)

        worker, finished, _ = _make_worker(process)
        await worker.start()
        with SessionLocal() as db:
            _enqueue(db, "w1")
        worker.wake()
        await _until(lambda: finished)
        await worker.stop()
        return finished

    assert asyncio.run(body()) == [("w1", "complete", None)]
    assert _status("w1")[:2] == ("complete", 1)


def test_worker_retries_a_failing_job_until_it_succeeds():
    calls = []

    async def body():
        async def process(job, progress):
            calls.append(job.attempts)
            if job.attempts < 3:
                raise RuntimeError("model overloaded")

        worker, finished, failures = _make_worker(process)
        await worker.start()
        with SessionLocal() as db:
            _enqueue(db, "w2", max_attempts=3)
        await _until(lambda: finished)
        await worker.stop()
        return finished, failures

    finished, failures = asyncio.run(body())
    assert calls == [1, 2, 3]
    assert finished == [("w2", "complete", None)]
    assert [f[1] for f in failures] == [False, False]                      # two non-final failures


def test_worker_fails_the_job_after_the_last_attempt_and_reports_a_safe_message():
    async def body():
        async def process(job, progress):
            raise RuntimeError("secret path C:\\x")

        worker, finished, failures = _make_worker(process)
        await worker.start()
        with SessionLocal() as db:
            _enqueue(db, "w3", max_attempts=2)
        await _until(lambda: finished)
        await worker.stop()
        return finished, failures

    finished, failures = asyncio.run(body())
    assert finished == [("w3", "failed", "Processing failed. Reference: test.")]
    assert failures[-1][1] is True
    assert _status("w3")[:2] == ("failed", 2)


def test_permanent_error_is_not_retried():
    async def body():
        async def process(job, progress):
            raise q.PermanentJobError("unreadable")

        worker, finished, _ = _make_worker(process)
        await worker.start()
        with SessionLocal() as db:
            _enqueue(db, "w4", max_attempts=5)
        await _until(lambda: finished)
        await worker.stop()

    asyncio.run(body())
    assert _status("w4")[:2] == ("failed", 1)


def test_a_job_left_running_by_a_crashed_process_is_picked_up_after_restart():
    """Process A claims a job and dies (its lease is simply never renewed). A new process starts
    and finishes the job once the lease has expired: no manual cleanup, no lost upload."""
    with SessionLocal() as db:
        _enqueue(db, "crash")
        q.claim_next(db, "dead-process", lease_seconds=0.2)

    async def body():
        async def process(job, progress):
            assert job.attempts == 2                                         # second attempt

        worker, finished, _ = _make_worker(process)
        await worker.start()
        await _until(lambda: finished)
        await worker.stop()

    asyncio.run(body())
    assert _status("crash")[:2] == ("complete", 2)


def test_clean_shutdown_returns_the_running_job_to_the_queue_without_costing_an_attempt():
    started = threading.Event()

    async def body():
        async def process(job, progress):
            started.set()
            await asyncio.sleep(60)

        worker, finished, _ = _make_worker(process)
        await worker.start()
        with SessionLocal() as db:
            _enqueue(db, "stop1")
        await _until(started.is_set)
        await worker.stop()
        assert finished == []

    asyncio.run(body())
    assert _status("stop1")[:2] == ("queued", 0)


def test_a_worker_that_loses_its_lease_abandons_the_run():
    cancelled = threading.Event()
    started = threading.Event()

    async def body():
        async def process(job, progress):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        worker, finished, _ = _make_worker(process, lease_seconds=0.3)
        await worker.start()
        with SessionLocal() as db:
            _enqueue(db, "lost1")
        await _until(started.is_set)
        with SessionLocal() as db:                                           # another worker takes over
            db.query(UploadJob).filter(UploadJob.id == "lost1").update({"locked_by": "someone-else"})
            db.commit()
        await _until(cancelled.is_set)
        await worker.stop()
        return finished

    assert asyncio.run(body()) == []                 # the abandoned run does not report a result
    assert _status("lost1")[0] == "running"          # and does not touch the new owner's job


def test_concurrency_limit_bounds_how_many_jobs_run_at_once():
    active = {"now": 0, "peak": 0}

    async def body():
        async def process(job, progress):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.1)
            active["now"] -= 1

        worker, finished, _ = _make_worker(process, concurrency=2)
        await worker.start()
        with SessionLocal() as db:
            for i in range(6):
                _enqueue(db, f"cc{i}")
        await _until(lambda: len(finished) == 6)
        await worker.stop()

    asyncio.run(body())
    assert active["peak"] == 2
