"""
Benchmark the upload job queue itself (claim, lease, complete) with a fake job body, so you get
numbers for the queue and nothing else - the model calls, which dominate a real upload, are not
involved. Run from the backend folder:

    python scripts/bench_queue.py --jobs 300 --workers 4 --work-ms 50
    python scripts/bench_queue.py --jobs 300 --workers 4 --work-ms 50 --database-url postgresql://user:pw@localhost/db

Without --database-url it uses a throw-away SQLite file in the temp folder and never touches
medsight.db. With one, it only creates the upload_jobs table if missing and deletes the rows it
added (ids starting "bench-"); it never drops anything.

What the output means:
  throughput        jobs finished per second, start to finish
  ideal             workers / work-seconds: the ceiling if the queue added zero overhead
  efficiency        throughput / ideal. Close to 100% means the queue is not the bottleneck
  queue wait        time from enqueue to a worker starting the job (p50 / p95 / max). With more
                    jobs than workers this mostly measures the backlog, by design.
"""
import argparse
import asyncio
import os
import statistics
import sys
import tempfile
import time

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=300)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--work-ms", type=float, default=50.0, help="simulated analysis time per job")
ap.add_argument("--database-url", default="")
args = ap.parse_args()

tmpdir = None
if args.database_url:
    os.environ["DATABASE_URL"] = args.database_url
else:
    tmpdir = tempfile.mkdtemp(prefix="medsight-bench-")
    os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tmpdir, "bench.db").replace("\\", "/")
os.environ.setdefault("MEDSIGHT_SKIP_DOTENV", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.db.models import UploadJob  # noqa: E402
from src.db.session import SessionLocal, engine  # noqa: E402
from src.jobs import queue as q  # noqa: E402
from src.jobs.worker import JobWorker  # noqa: E402

UploadJob.__table__.create(bind=engine, checkfirst=True)
PREFIX = "bench-"


async def main() -> None:
    done = []

    async def process(job, progress):
        await asyncio.sleep(args.work_ms / 1000)

    worker = JobWorker(session_factory=SessionLocal, process=process,
                       on_failure=lambda j, e, f: "failed", on_finished=lambda j, s, e: done.append(s),
                       concurrency=args.workers, poll_seconds=0.02)
    t0 = time.perf_counter()
    with SessionLocal() as db:
        for i in range(args.jobs):
            q.enqueue(db, job_id=f"{PREFIX}{i}", owner="bench", ip=None, patient_id="P", filename="x", ext=".pdf",
                      report_date="", language="en", spool_path="-", max_attempts=1)
    enqueue_s = time.perf_counter() - t0
    await worker.start()
    t1 = time.perf_counter()
    while len(done) < args.jobs:
        await asyncio.sleep(0.02)
    elapsed = time.perf_counter() - t1
    await worker.stop()

    with SessionLocal() as db:
        rows = db.query(UploadJob).filter(UploadJob.id.like(PREFIX + "%")).all()
        waits = sorted((r.started_at - r.created_at).total_seconds() for r in rows)
        ok = sum(1 for r in rows if r.status == "complete")
        db.query(UploadJob).filter(UploadJob.id.like(PREFIX + "%")).delete(synchronize_session=False)
        db.commit()

    throughput = args.jobs / elapsed
    ideal = args.workers / (args.work_ms / 1000)
    pct = lambda p: waits[min(len(waits) - 1, int(len(waits) * p))]
    print(f"database        {engine.dialect.name}")
    print(f"jobs            {args.jobs} ({ok} completed)   workers {args.workers}   work {args.work_ms:g} ms")
    print(f"enqueue         {args.jobs / enqueue_s:,.0f} jobs/s")
    print(f"throughput      {throughput:,.1f} jobs/s")
    print(f"ideal           {ideal:,.1f} jobs/s   efficiency {throughput / ideal:.0%}")
    print(f"queue wait      p50 {statistics.median(waits):.2f}s   p95 {pct(0.95):.2f}s   max {waits[-1]:.2f}s")


try:
    asyncio.run(main())
finally:
    engine.dispose()
    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
