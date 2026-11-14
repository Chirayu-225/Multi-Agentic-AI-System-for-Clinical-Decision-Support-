"""
src/resilience.py

Two small primitives that keep an unhealthy model provider from taking the whole pipeline down
with it. Both are plain, dependency-free and take an injectable clock so tests never sleep.

CircuitBreaker
  closed     calls flow; consecutive provider failures are counted.
  open       after `failure_threshold` failures in a row, calls are refused instantly
             (CircuitOpen) for `cooldown_seconds`. Without this, every queued report would sit
             through its own multi-minute retry ladder against a provider that is plainly down.
  half-open  once the cooldown passes, up to `half_open_max` probe calls are let through. One
             success closes the circuit; one failure re-opens it for another full cooldown.
  Only provider-health failures count (429, 503, timeouts). A bad request is our bug, not an outage.

RateLimiter
  Token bucket shared by every worker in the process: `rate_per_minute` sustained, with a burst
  of up to `burst`. The free Gemini tier is a requests-per-minute quota; spending it with
  uncoordinated concurrency turns into a self-inflicted 429 storm.
"""
import asyncio
import threading
import time
from typing import Callable

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitOpen(Exception):
    """The provider is considered down; the call was not attempted."""

    def __init__(self, retry_after: float):
        super().__init__(f"circuit open; retry in {retry_after:.0f}s")
        self.retry_after = max(retry_after, 0.0)


class CircuitBreaker:
    def __init__(self, *, failure_threshold: int = 5, cooldown_seconds: float = 60.0,
                 half_open_max: int = 1, clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max = max(1, half_open_max)
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probes = 0
        self.opened_total = 0

    @property
    def state(self) -> str:
        with self._lock:
            self._advance()
            return self._state

    def _advance(self) -> None:
        if self._state == OPEN and self._clock() - self._opened_at >= self.cooldown_seconds:
            self._state, self._probes = HALF_OPEN, 0

    def retry_after(self) -> float:
        with self._lock:
            self._advance()
            if self._state != OPEN:
                return 0.0
            return max(self.cooldown_seconds - (self._clock() - self._opened_at), 0.0)

    def allow(self) -> None:
        """Raise CircuitOpen if a call must not be made now; otherwise reserve the slot."""
        with self._lock:
            self._advance()
            if self._state == OPEN:
                raise CircuitOpen(self.cooldown_seconds - (self._clock() - self._opened_at))
            if self._state == HALF_OPEN:
                if self._probes >= self.half_open_max:
                    raise CircuitOpen(1.0)   # a probe is already in flight
                self._probes += 1

    def cancel_probe(self) -> None:
        """A reserved call was cancelled before it learned anything; free its probe slot."""
        with self._lock:
            if self._state == HALF_OPEN and self._probes > 0:
                self._probes -= 1

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state, self._probes = CLOSED, 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state == HALF_OPEN:
                self._trip()
                return
            self._failures += 1
            if self._state == CLOSED and self._failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state, self._opened_at, self._probes = OPEN, self._clock(), 0
        self.opened_total += 1

    def reset(self) -> None:
        with self._lock:
            self._state, self._failures, self._probes = CLOSED, 0, 0


class RateLimiter:
    def __init__(self, *, rate_per_minute: float, burst: int | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], "asyncio.Future"] = asyncio.sleep):
        self.rate_per_second = rate_per_minute / 60.0 if rate_per_minute > 0 else 0.0
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_minute // 6) or 1))
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate_per_second > 0

    def _take(self) -> float:
        """Take a token now (returns 0) or return how long until one is available."""
        with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate_per_second)
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            return (1.0 - self._tokens) / self.rate_per_second

    async def acquire(self) -> float:
        """Wait for a token. Returns the total time spent waiting."""
        if not self.enabled:
            return 0.0
        waited = 0.0
        while True:
            delay = self._take()
            if delay <= 0:
                return waited
            await self._sleep(delay)
            waited += delay
