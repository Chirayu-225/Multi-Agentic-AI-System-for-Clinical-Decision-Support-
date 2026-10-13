"""
src/security/rate_limit.py

Minimal in-process rate limiter for the upload endpoint. The pipeline it
guards is expensive (OCR/NER + two Gemini calls + external API calls per
upload), and the endpoint previously had no limit at all — anyone
with the API key (or, before Phase 1's auth fix, anyone at all) could
exhaust the free-tier model quota or run up cost with a burst of uploads.

This is a fixed-window counter per key (IP by default), held in process
memory. That's the right amount of engineering for a single-instance
deployment; it resets on restart and does NOT coordinate across multiple
backend replicas. If this ever runs behind a load balancer with more than
one backend instance, replace this with a shared store (Redis
INCR/EXPIRE is the standard pattern) — noted here rather than silently
left as a scaling trap.
"""

import time
from collections import defaultdict, deque

_WINDOWS: dict[str, deque] = defaultdict(deque)


def allow(key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
    """
    Returns (allowed, retry_after_seconds). Call once per incoming request;
    only counts calls where you actually intend to consume a slot.
    """
    now = time.monotonic()
    window = _WINDOWS[key]

    # Drop timestamps outside the window
    while window and window[0] <= now - window_seconds:
        window.popleft()

    if len(window) >= max_requests:
        retry_after = int(window_seconds - (now - window[0])) + 1
        return False, max(retry_after, 1)

    window.append(now)
    return True, 0
