from src.security import rate_limit


def test_allows_up_to_max_then_blocks():
    key = "rl-test-1"
    results = [rate_limit.allow(key, 3, 60)[0] for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_blocked_call_reports_retry_after():
    key = "rl-test-2"
    for _ in range(2):
        rate_limit.allow(key, 2, 60)
    ok, retry = rate_limit.allow(key, 2, 60)
    assert ok is False and 0 < retry <= 60


def test_keys_are_independent():
    rate_limit.allow("rl-a", 1, 60)
    assert rate_limit.allow("rl-b", 1, 60)[0] is True


def test_retry_after_never_exceeds_window_with_coarse_clock(monkeypatch):
    """Windows' monotonic clock has ~15 ms resolution, so two calls can see the same time."""
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: 1000.0)
    key = "rl-coarse-clock"
    for _ in range(2):
        rate_limit.allow(key, 2, 60)
    ok, retry = rate_limit.allow(key, 2, 60)
    assert ok is False and retry == 60
