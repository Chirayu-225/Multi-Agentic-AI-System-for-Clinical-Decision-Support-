"""Circuit breaker, rate limiter, and how src/core/llm.py uses them. A fake clock drives everything,
so none of these tests sleeps."""
import asyncio

import pytest

from src.core import llm
from src.core.resilience import CLOSED, HALF_OPEN, OPEN, CircuitBreaker, CircuitOpen, RateLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def make_breaker(clock, threshold=3, cooldown=30.0, probes=1):
    return CircuitBreaker(failure_threshold=threshold, cooldown_seconds=cooldown,
                          half_open_max=probes, clock=clock)


# ── circuit breaker ────────────────────────────────────────────────────────────

def test_breaker_opens_after_consecutive_failures_and_then_refuses_instantly():
    clock = Clock()
    b = make_breaker(clock)
    for _ in range(2):
        b.allow()
        b.record_failure()
    assert b.state == CLOSED
    b.allow()
    b.record_failure()
    assert b.state == OPEN
    with pytest.raises(CircuitOpen) as e:
        b.allow()
    assert 0 < e.value.retry_after <= 30


def test_a_success_resets_the_failure_count():
    b = make_breaker(Clock())
    for _ in range(2):
        b.record_failure()
    b.record_success()
    for _ in range(2):
        b.record_failure()
    assert b.state == CLOSED                       # 2 + 2 failures, but never 3 in a row


def test_half_open_lets_one_probe_through_and_a_success_closes_it():
    clock = Clock()
    b = make_breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock.advance(31)
    assert b.state == HALF_OPEN
    b.allow()                                      # the probe
    with pytest.raises(CircuitOpen):
        b.allow()                                  # a second concurrent caller is held back
    b.record_success()
    assert b.state == CLOSED
    b.allow()


def test_a_failed_probe_reopens_for_a_full_cooldown():
    clock = Clock()
    b = make_breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock.advance(31)
    b.allow()
    b.record_failure()
    assert b.state == OPEN and b.opened_total == 2
    clock.advance(29)
    assert b.state == OPEN
    clock.advance(2)
    assert b.state == HALF_OPEN


def test_a_cancelled_probe_does_not_wedge_the_breaker_half_open():
    clock = Clock()
    b = make_breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock.advance(31)
    b.allow()
    b.cancel_probe()
    b.allow()                                      # the slot is free again


# ── rate limiter ───────────────────────────────────────────────────────────────

def test_rate_limiter_allows_a_burst_then_paces_calls():
    clock = Clock()
    slept = []

    async def sleep(s):
        slept.append(s)
        clock.advance(s)

    lim = RateLimiter(rate_per_minute=60, burst=3, clock=clock, sleep=sleep)

    async def run():
        for _ in range(3):
            assert await lim.acquire() == 0        # the burst is free
        waited = await lim.acquire()               # the 4th must wait ~1s (60/min = 1/s)
        return waited

    waited = asyncio.run(run())
    assert 0.9 <= waited <= 1.1 and len(slept) >= 1


def test_rate_limiter_is_off_when_the_rate_is_zero():
    lim = RateLimiter(rate_per_minute=0)
    assert not lim.enabled
    assert asyncio.run(lim.acquire()) == 0


def test_rate_limiter_is_shared_by_concurrent_callers():
    """However the callers interleave, grants never outrun burst + rate * elapsed time."""
    clock = Clock()
    start = clock.t

    async def sleep(s):
        clock.advance(s)
        await asyncio.sleep(0)

    lim = RateLimiter(rate_per_minute=120, burst=2, clock=clock, sleep=sleep)
    grants = []

    async def one():
        await lim.acquire()
        grants.append(clock.t - start)

    async def run():
        await asyncio.gather(*[one() for _ in range(8)])

    asyncio.run(run())
    assert len(grants) == 8
    for i, t in enumerate(sorted(grants), start=1):
        assert i <= 2 + 2.0 * t + 1e-6, f"grant {i} at t={t:.2f}s exceeded the allowed rate"


# ── how llm.with_retry uses them ───────────────────────────────────────────────

class Overloaded(Exception):
    def __str__(self):
        return "503 UNAVAILABLE: high demand"


def _install_breaker(monkeypatch, model="m-main", threshold=3):
    b = CircuitBreaker(failure_threshold=threshold, cooldown_seconds=60)
    monkeypatch.setitem(llm._breakers, model, b)
    return b


def test_with_retry_trips_the_breaker_and_later_calls_fail_without_calling_the_provider(monkeypatch):
    real_sleep = asyncio.sleep

    async def no_wait(_s):
        await real_sleep(0)

    monkeypatch.setattr(llm.asyncio, "sleep", no_wait)
    b = _install_breaker(monkeypatch, llm.MODEL, threshold=3)
    calls = {"n": 0}

    async def down():
        calls["n"] += 1
        raise Overloaded()

    with pytest.raises(Overloaded):
        asyncio.run(llm.with_retry(down, label="t", attempts=3, base_wait=1))
    assert calls["n"] == 3 and b.state == OPEN

    with pytest.raises(CircuitOpen):
        asyncio.run(llm.with_retry(down, label="t", attempts=3, base_wait=1))
    assert calls["n"] == 3                         # the provider was not touched again


def test_a_bad_request_does_not_count_as_an_outage(monkeypatch):
    b = _install_breaker(monkeypatch, llm.MODEL, threshold=2)

    async def bad():
        raise ValueError("400 INVALID_ARGUMENT")

    for _ in range(5):
        with pytest.raises(ValueError):
            asyncio.run(llm.with_retry(bad, label="t", attempts=3, base_wait=1))
    assert b.state == CLOSED


def test_open_main_circuit_falls_back_to_the_fallback_model(monkeypatch):
    monkeypatch.setattr(llm, "FALLBACK_MODEL", "m-fallback")
    main_b = _install_breaker(monkeypatch, llm.MODEL)
    _install_breaker(monkeypatch, "m-fallback")
    main_b.reset()
    for _ in range(3):
        main_b.record_failure()
    seen = []

    async def make_call(model):
        seen.append(model)
        return "ok"

    result, model = asyncio.run(llm.with_fallback(make_call, label="t"))
    assert result == "ok" and model == "m-fallback" and seen == ["m-fallback"]


def test_llm_unavailable_for_is_zero_when_a_fallback_is_usable(monkeypatch):
    monkeypatch.setattr(llm, "FALLBACK_MODEL", "m-fallback")
    main_b = _install_breaker(monkeypatch, llm.MODEL)
    _install_breaker(monkeypatch, "m-fallback")
    for _ in range(3):
        main_b.record_failure()
    assert llm.llm_unavailable_for() == 0          # main is down, fallback is fine
    for _ in range(3):
        llm.get_breaker("m-fallback").record_failure()
    assert llm.llm_unavailable_for() > 0           # both are down


def test_limiter_is_applied_before_every_attempt(monkeypatch):
    waits = []

    class Spy:
        async def acquire(self):
            waits.append(1)
            return 0

    monkeypatch.setattr(llm, "llm_limiter", Spy())
    _install_breaker(monkeypatch, llm.MODEL)

    async def ok():
        return "fine"

    assert asyncio.run(llm.with_retry(ok, label="t")) == "fine"
    assert waits == [1]
