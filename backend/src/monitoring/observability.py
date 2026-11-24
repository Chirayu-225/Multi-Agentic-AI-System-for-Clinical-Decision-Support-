"""
src/monitoring/observability.py

The three things you need to operate the service, with no external dependencies:

  1. Correlation ids. Every request gets an id (X-Request-ID, honoured if the caller or the
     proxy sent a sane one) and every background job logs under its job id. Both are added to
     every log line, so one upload can be followed from the HTTP request, through the queue,
     to the worker's log lines.
  2. Metrics. Counters, gauges and a small histogram, rendered in the Prometheus text format
     at GET /metrics. In-process only: with several processes, scrape each one (the numbers are
     per process by design, which is how Prometheus expects counters to work).
  3. Nothing here ever records report content. Labels are bounded sets (status, route template,
     outcome) - never ids, usernames or free text - so cardinality stays small and PHI stays out.
"""
import logging
import os
import re
import threading
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
job_id_var: ContextVar[str] = ContextVar("job_id", default="-")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def accept_request_id(candidate: str | None, generate) -> str:
    """Use the caller's X-Request-ID only if it is a plain token (it ends up in logs and a
    response header, so a newline or a 10 KB string must not get through)."""
    return candidate if candidate and _REQUEST_ID_RE.match(candidate) else generate()


def configure_logging() -> None:
    """One format for every logger, carrying the request and job ids."""
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.request_id = request_id_var.get()
        record.job_id = job_id_var.get()
        return record

    logging.setLogRecordFactory(factory)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [req=%(request_id)s job=%(job_id)s] %(name)s: %(message)s",
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_counters: dict[tuple, float] = {}
_help: dict[str, str] = {}
_histograms: dict[tuple, dict] = {}
_gauge_fns: dict[str, tuple[str, callable]] = {}

# Seconds. Upload jobs take tens of seconds, HTTP calls milliseconds; one set covers both.
_BUCKETS = (0.05, 0.25, 1, 5, 15, 30, 60, 120, 300)


def _key(name: str, labels: dict) -> tuple:
    return (name, tuple(sorted(labels.items())))


def describe(name: str, text: str) -> None:
    _help[name] = text


def inc(name: str, amount: float = 1, **labels) -> None:
    with _lock:
        k = _key(name, labels)
        _counters[k] = _counters.get(k, 0) + amount


def observe(name: str, value: float, **labels) -> None:
    with _lock:
        h = _histograms.setdefault(_key(name, labels), {"sum": 0.0, "count": 0, "buckets": [0] * len(_BUCKETS)})
        h["sum"] += value
        h["count"] += 1
        for i, upper in enumerate(_BUCKETS):
            if value <= upper:
                h["buckets"][i] += 1


def gauge(name: str, help_text: str, fn) -> None:
    """Register a gauge that is computed when scraped (e.g. queue depth read from the database)."""
    _gauge_fns[name] = (help_text, fn)


def reset() -> None:
    """For tests."""
    with _lock:
        _counters.clear()
        _histograms.clear()


def value(name: str, **labels) -> float:
    with _lock:
        return _counters.get(_key(name, labels), 0)


def _fmt_labels(labels: tuple, extra: tuple = ()) -> str:
    items = list(labels) + list(extra)
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{str(v).replace(chr(34), "")}"' for k, v in items) + "}"


def render() -> str:
    lines: list[str] = []
    with _lock:
        names = sorted({k[0] for k in _counters})
        for name in names:
            if name in _help:
                lines.append(f"# HELP {name} {_help[name]}")
            lines.append(f"# TYPE {name} counter")
            for (n, labels), v in sorted(_counters.items()):
                if n == name:
                    lines.append(f"{name}{_fmt_labels(labels)} {v:g}")
        hnames = sorted({k[0] for k in _histograms})
        for name in hnames:
            if name in _help:
                lines.append(f"# HELP {name} {_help[name]}")
            lines.append(f"# TYPE {name} histogram")
            for (n, labels), h in sorted(_histograms.items()):
                if n != name:
                    continue
                for upper, c in zip(_BUCKETS, h["buckets"]):
                    lines.append(f"{name}_bucket{_fmt_labels(labels, (('le', upper),))} {c}")
                lines.append(f"{name}_bucket{_fmt_labels(labels, (('le', '+Inf'),))} {h['count']}")
                lines.append(f"{name}_sum{_fmt_labels(labels)} {h['sum']:g}")
                lines.append(f"{name}_count{_fmt_labels(labels)} {h['count']}")
    for name, (help_text, fn) in sorted(_gauge_fns.items()):
        try:
            v = fn()
        except Exception:  # noqa: BLE001 - a broken gauge must not break the scrape
            continue
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        if isinstance(v, dict):
            for label_value, num in sorted(v.items()):
                lines.append(f'{name}{{status="{label_value}"}} {num:g}')
        else:
            lines.append(f"{name} {v:g}")
    return "\n".join(lines) + "\n"
