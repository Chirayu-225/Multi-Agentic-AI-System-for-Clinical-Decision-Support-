"""The audit chain stays one straight line however many writers append at once."""
import threading

import pytest
from sqlalchemy.exc import IntegrityError

from src.db.models import AuditEvent
from src.db.session import SessionLocal
from src.security import audit

from tests.test_api_flow import fresh_state  # noqa: F401


def test_concurrent_appends_never_fork_the_chain():
    threads_n, per_thread = 8, 15
    barrier = threading.Barrier(threads_n)

    def writer(n):
        barrier.wait()
        for i in range(per_thread):
            audit.record(actor=f"t{n}", action="test.event", resource=f"r{i}", critical=True)

    # The module lock only serialises one process; defeat it so this test exercises the database's
    # guarantee (what protects several uvicorn workers), not the lock.
    real_lock = audit._lock

    class _NoLock:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    audit._lock = _NoLock()
    try:
        ts = [threading.Thread(target=writer, args=(n,)) for n in range(threads_n)]
        [t.start() for t in ts]
        [t.join(timeout=60) for t in ts]
    finally:
        audit._lock = real_lock

    with SessionLocal() as db:
        result = audit.verify_chain(db)
        assert result["ok"], result
        assert result["checked"] == threads_n * per_thread


def test_database_rejects_a_second_event_appended_after_the_same_head():
    audit.record(actor="a", action="first", critical=True)
    with SessionLocal() as db:
        head = db.query(AuditEvent).order_by(AuditEvent.id.desc()).first()
        fork = AuditEvent(ts="2026-01-01T00:00:00", actor="mallory", action="fork", outcome="success",
                          prev_hash=head.prev_hash, hash="x" * 64)     # same parent as the head
        db.add(fork)
        with pytest.raises(IntegrityError):
            db.commit()
