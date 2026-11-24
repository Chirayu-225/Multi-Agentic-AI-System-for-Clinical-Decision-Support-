"""
src/jobs/worker.py

Runs queued jobs. It knows how to claim, keep a lease alive, retry and shut down cleanly, and
nothing about uploads: the actual work is the `process` coroutine the application passes in.

  * `concurrency` runner loops share the queue. Each claims one job at a time, so the number of
    uploads analysed at once (and therefore the load on the model provider and on memory) is a
    setting, not an accident of how many requests arrived.
  * While a job runs, a heartbeat extends its lease. If the heartbeat finds the lease gone
    (another worker took the job, e.g. after a long pause) the local run is cancelled instead of
    racing the new owner.
  * On a clean shutdown a running job is released back to the queue without using up an attempt.
    On a crash the lease simply runs out and the job is picked up again.
  * Database calls are plain blocking SQLAlchemy, so they run in a thread (`asyncio.to_thread`)
    and never stall the event loop that is also serving requests and websockets.
"""
import asyncio
import logging
import time
from datetime import timedelta
from typing import Awaitable, Callable

from src.monitoring import observability as obs
from src.jobs import queue as q

logger = logging.getLogger("medsight.jobs")

obs.describe("jobs_claimed_total", "Jobs handed to a worker (a retried job counts each time).")
obs.describe("jobs_finished_total", "Jobs that reached a terminal state, by status.")
obs.describe("job_retries_total", "Attempts that failed and were requeued.")
obs.describe("job_deferred_total", "Attempts postponed (not failed) because the AI service was unavailable.")
obs.describe("job_lease_lost_total", "Runs abandoned because another worker owned the job.")
obs.describe("job_duration_seconds", "Wall-clock time of one job attempt.")

Process = Callable[[q.JobSnapshot, Callable[[str, int], None]], Awaitable[None]]


class JobWorker:
    def __init__(self, *, session_factory, process: Process,
                 on_failure: Callable[[q.JobSnapshot, BaseException, bool], str],
                 on_finished: Callable[[q.JobSnapshot, str, str | None], None],
                 concurrency: int = 2, lease_seconds: float = 60.0, poll_seconds: float = 1.0,
                 retry_base_seconds: float = 15.0, retention: timedelta = timedelta(hours=24),
                 maintenance_seconds: float = 30.0):
        self._session_factory = session_factory
        self._process = process
        self._on_failure = on_failure
        self._on_finished = on_finished
        self.concurrency = max(1, concurrency)
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self.retry_base_seconds = retry_base_seconds
        self.retention = retention
        self.maintenance_seconds = maintenance_seconds
        self.worker_id = q.new_worker_id()
        self._tasks: list[asyncio.Task] = []
        self._bg: set[asyncio.Task] = set()
        self._wake = asyncio.Event()
        self._stopping = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._stopping = False
        self._tasks = [asyncio.create_task(self._runner(i), name=f"job-runner-{i}") for i in range(self.concurrency)]
        self._tasks.append(asyncio.create_task(self._maintenance(), name="job-maintenance"))
        logger.info("job worker %s started concurrency=%d lease=%ss", self.worker_id, self.concurrency, self.lease_seconds)

    async def stop(self) -> None:
        self._stopping = True
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("job worker %s stopped", self.worker_id)

    @property
    def alive(self) -> bool:
        return bool(self._tasks) and all(not t.done() for t in self._tasks)

    def wake(self) -> None:
        """Tell idle runners there is new work (otherwise they notice within `poll_seconds`)."""
        self._wake.set()

    # ── database access ───────────────────────────────────────────────────────

    def _call(self, fn, *args, **kwargs):
        db = self._session_factory()
        try:
            return fn(db, *args, **kwargs)
        finally:
            db.close()

    async def _db(self, fn, *args, **kwargs):
        return await asyncio.to_thread(self._call, fn, *args, **kwargs)

    # ── loops ─────────────────────────────────────────────────────────────────

    async def _runner(self, index: int) -> None:
        wid = f"{self.worker_id}.{index}"
        while not self._stopping:
            try:
                job = await self._db(q.claim_next, wid, self.lease_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - database hiccup: back off, do not kill the runner
                logger.exception("claim failed; retrying in %ss", self.poll_seconds)
                await asyncio.sleep(self.poll_seconds)
                continue
            if job is None:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                continue
            await self._run(job)

    async def _maintenance(self) -> None:
        while not self._stopping:
            try:
                for job in await self._db(q.reap_exhausted):
                    logger.error("job %s lost its worker on the final attempt; marked failed", job.id)
                    obs.inc("jobs_finished_total", status=q.FAILED)
                    self._on_finished(job, q.FAILED, "Processing was interrupted and could not be completed. Please upload again.")
                pruned = await self._db(q.prune, self.retention)
                if pruned:
                    logger.info("pruned %d finished job(s)", pruned)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("job maintenance failed")
            await asyncio.sleep(self.maintenance_seconds)

    # ── one job ───────────────────────────────────────────────────────────────

    async def _run(self, job: q.JobSnapshot) -> None:
        token = obs.job_id_var.set(job.id)
        started = time.monotonic()
        obs.inc("jobs_claimed_total")
        logger.info("job started attempt=%d/%d", job.attempts, job.max_attempts)
        lease_lost = asyncio.Event()
        work = asyncio.create_task(self._process(job, self._progress_reporter(job)))
        beat = asyncio.create_task(self._heartbeat(job, work, lease_lost))
        try:
            await work
        except asyncio.CancelledError:
            if lease_lost.is_set():
                obs.inc("job_lease_lost_total")
                logger.warning("job lease lost; abandoning this run (another worker owns it now)")
                return
            # Shutdown while running: hand the job back, this attempt does not count.
            await self._db(q.release, job.id, job.worker_id)
            raise
        except q.DeferJob as d:
            if await self._db(q.defer, job.id, job.worker_id, d.delay_seconds):
                obs.inc("job_deferred_total")
                logger.warning("job deferred %.0fs: %s", d.delay_seconds, d.reason)
            else:
                obs.inc("job_lease_lost_total")
        except Exception as exc:  # noqa: BLE001 - every failure goes through the retry policy
            permanent = isinstance(exc, q.PermanentJobError)
            final = permanent or job.is_last_attempt
            message = self._on_failure(job, exc, final)
            outcome = await self._db(q.fail_or_retry, job.id, job.worker_id, message,
                                     retry_base_seconds=self.retry_base_seconds, permanent=permanent)
            if outcome == q.FAILED:
                obs.inc("jobs_finished_total", status=q.FAILED)
                self._on_finished(job, q.FAILED, message)
            elif outcome == "retry":
                obs.inc("job_retries_total")
                logger.warning("job failed on attempt %d/%d; will retry", job.attempts, job.max_attempts)
        else:
            if await self._db(q.complete, job.id, job.worker_id):
                obs.inc("jobs_finished_total", status=q.COMPLETE)
                self._on_finished(job, q.COMPLETE, None)
            else:
                obs.inc("job_lease_lost_total")
                logger.warning("job finished but its lease was gone; the new owner will settle it")
        finally:
            beat.cancel()
            obs.observe("job_duration_seconds", time.monotonic() - started)
            obs.job_id_var.reset(token)

    async def _heartbeat(self, job: q.JobSnapshot, work: asyncio.Task, lease_lost: asyncio.Event) -> None:
        interval = max(self.lease_seconds / 3, 0.05)
        while True:
            await asyncio.sleep(interval)
            try:
                alive = await self._db(q.heartbeat, job.id, job.worker_id, self.lease_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a missed beat is survivable; the lease has slack
                logger.exception("heartbeat failed")
                continue
            if not alive:
                lease_lost.set()
                work.cancel()
                return

    def _progress_reporter(self, job: q.JobSnapshot) -> Callable[[str, int], None]:
        """A sync callback for the pipeline. The write is queued on the loop; because the SQL
        never lowers `percent`, writes that land out of order are harmless."""
        def report(stage: str, percent: int) -> None:
            async def _write():
                try:
                    await self._db(q.set_progress, job.id, job.worker_id, stage, percent)
                except Exception:  # noqa: BLE001 - progress is best-effort
                    logger.debug("progress write failed", exc_info=True)
            task = asyncio.get_running_loop().create_task(_write())
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)
        return report
