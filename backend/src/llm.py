"""
src/llm.py

The ONE place this project talks to Gemini. Every agent (orchestrator, summarizer,
verifier, translator) imports from here, so the model, the SDK and the retry policy
cannot drift apart again.

Why the google-genai SDK and not google-generativeai:
  google-generativeai is Google's legacy SDK. Gemini 3 models attach a "thought
  signature" to every function call and reject the next turn with a 400 if it is not
  sent back. The legacy SDK predates that field and drops it, so the orchestrator's
  tool-calling loop fails on current models. google-genai's chat sessions keep the
  full model turn (signature included) in history automatically.

Why the model is pinned by setting, not chosen by a router:
  The eval numbers in EVAL_RESULTS.md only mean something for the model that
  produced them. GEMINI_MODEL picks it; the model(s) that actually answered are
  recorded on every AgentRun so a report can always be traced to its model.

Timeouts, retries, fallback:
  The SDK's defaults are no request timeout and up to 5 silent attempts with delays of
  up to 60 s, so a busy model ("503 UNAVAILABLE: high demand") made an upload look
  frozen. Here: every request times out (GEMINI_TIMEOUT_SECONDS), the SDK's own retries
  are off, and with_retry() is the single retry policy — rate limits (429), overload
  (503/500) and timeouts back off and retry; anything else fails at once. If the main
  model is still unavailable after that and GEMINI_FALLBACK_MODEL is set, the call is
  made once more on the fallback, and the run records that it was used.
"""

import asyncio
import contextvars
import logging
import os

import time

from src import observability as obs
from src.resilience import CircuitBreaker, CircuitOpen, RateLimiter

logger = logging.getLogger("medsight")

# gemini-2.5-flash is not deprecated, but Google limits it to API keys that used it
# before, so a fresh deployment with a new key cannot call it. Default to a current
# stable model; override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.8-flash"
MODEL = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
FALLBACK_MODEL = (os.getenv("GEMINI_FALLBACK_MODEL") or "").strip() or None
TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))

# Provider-health protection (see src/resilience.py). One breaker per model, because the fallback
# model exists precisely for when the main one is overloaded; one rate limiter for the whole
# process, because the quota belongs to the API key, not to a model.
BREAKER_THRESHOLD = int(os.getenv("LLM_BREAKER_FAILURES", "5"))
BREAKER_COOLDOWN = float(os.getenv("LLM_BREAKER_COOLDOWN_SECONDS", "60"))
llm_limiter = RateLimiter(rate_per_minute=float(os.getenv("LLM_RPM", "0")),
                          burst=int(os.getenv("LLM_BURST", "5")))
_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(model: str | None = None) -> CircuitBreaker:
    key = model or MODEL
    if key not in _breakers:
        _breakers[key] = CircuitBreaker(failure_threshold=BREAKER_THRESHOLD, cooldown_seconds=BREAKER_COOLDOWN)
    return _breakers[key]


def llm_unavailable_for() -> float:
    """Seconds until the model path is worth trying again; 0 if it is usable now.
    Unavailable only when the main model is open AND there is no usable fallback."""
    main = get_breaker(MODEL).retry_after()
    if main <= 0:
        return 0.0
    if FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        fb = get_breaker(FALLBACK_MODEL).retry_after()
        return min(main, fb) if fb > 0 else 0.0
    return main


_client = None

# Models that actually answered during the current request (set by begin_model_tracking).
_models_used: contextvars.ContextVar[set | None] = contextvars.ContextVar("models_used", default=None)


class LlmBudgetExceeded(Exception):
    """One report tried to make more model calls than LLM_MAX_CALLS_PER_REPORT allows.
    A runaway loop (a retry bug, a tool-calling cycle) must cost a bounded amount of quota."""


obs.describe("llm_calls_total", "Model call attempts by purpose and outcome.")
obs.describe("llm_tokens_total", "Tokens used, by direction (input/output).")
obs.describe("llm_call_duration_seconds", "Wall-clock time of one model call attempt, by purpose.")

# Per-request trace of every model call attempt (set by begin_call_tracking, read by the caller).
# Holds metadata only - purpose, model, status, latency, token counts - never prompt or response text.
_calls_var: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_calls", default=None)
_budget_var: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_budget", default=None)


def begin_call_tracking(max_calls: int | None = None) -> list:
    """Start recording model call attempts for this request/job; returns the live list."""
    calls: list = []
    _calls_var.set(calls)
    _budget_var.set([max_calls, 0] if max_calls else None)
    return calls


def _spend_budget() -> None:
    b = _budget_var.get()
    if b is None:
        return
    if b[1] >= b[0]:
        raise LlmBudgetExceeded(f"more than {b[0]} model calls for one report")
    b[1] += 1


def _usage(result) -> tuple[int, int]:
    """Token counts from a response (or from a tuple that contains one, as the chat-opening call
    returns). Zeros when the SDK reports none - never an exception: tracing must not break a call."""
    for r in (result if isinstance(result, tuple) else (result,)):
        meta = getattr(r, "usage_metadata", None)
        if meta is not None:
            try:
                return (int(getattr(meta, "prompt_token_count", 0) or 0),
                        int(getattr(meta, "candidates_token_count", 0) or 0))
            except (TypeError, ValueError):
                return 0, 0
    return 0, 0


def _record_call(label: str, model: str | None, attempt: int, status: str, started: float,
                 result=None, error: Exception | None = None) -> None:
    purpose = label.split("/")[0].lower()
    tokens_in, tokens_out = _usage(result) if result is not None else (0, 0)
    latency = max(time.monotonic() - started, 0.0)
    obs.inc("llm_calls_total", purpose=purpose, status=status)
    if tokens_in or tokens_out:
        obs.inc("llm_tokens_total", tokens_in, direction="input")
        obs.inc("llm_tokens_total", tokens_out, direction="output")
    if status not in ("circuit_open", "budget"):
        obs.observe("llm_call_duration_seconds", latency, purpose=purpose)
    calls = _calls_var.get()
    if calls is not None:
        calls.append({"purpose": purpose, "model": model or MODEL, "attempt": attempt, "status": status,
                      "error_class": type(error).__name__ if error else None,
                      "latency_ms": int(latency * 1000), "input_tokens": tokens_in,
                      "output_tokens": tokens_out})


def begin_model_tracking() -> set:
    """Start recording which models answer during this request; returns the live set."""
    used: set = set()
    _models_used.set(used)
    return used


def note_model(model: str) -> None:
    used = _models_used.get()
    if used is not None:
        used.add(model)


def get_client():
    """Lazily build one shared client (import + key check happen on first use)."""
    global _client
    if _client is None:
        from google import genai
        from google.genai import types
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=int(TIMEOUT_SECONDS * 1000),                    # milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),      # we retry, not the SDK
            ),
        )
    return _client


def is_rate_limit(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def is_unavailable(err: Exception) -> bool:
    """Overloaded / transient server-side failure, or a timeout: worth retrying."""
    s = str(err)
    if any(code in s for code in ("503", "UNAVAILABLE", "500 INTERNAL", "504", "DEADLINE_EXCEEDED")):
        return True
    return "timeout" in type(err).__name__.lower() or isinstance(err, asyncio.TimeoutError)


def is_retryable(err: Exception) -> bool:
    return is_rate_limit(err) or is_unavailable(err)


async def with_retry(call, *, label: str, attempts: int = 3, base_wait: int = 20,
                     breaker: CircuitBreaker | None = None, model: str | None = None):
    """Run an async callable, backing off on rate limits, overload and timeouts only.
    Other errors (bad key, bad request, invalid model) propagate immediately.

    Every attempt first asks the circuit breaker (fail fast with CircuitOpen while the provider is
    down) and then the shared rate limiter. Only provider-health failures count against the
    breaker; an error the provider answered with (a bad request) proves it is up."""
    breaker = breaker or get_breaker()
    last = None
    for attempt in range(attempts):
        try:
            breaker.allow()
            _spend_budget()
        except CircuitOpen as e:
            _record_call(label, model, attempt + 1, "circuit_open", time.monotonic(), error=e)
            raise
        except LlmBudgetExceeded as e:
            _record_call(label, model, attempt + 1, "budget", time.monotonic(), error=e)
            raise
        await llm_limiter.acquire()
        started = time.monotonic()
        try:
            result = await call()
        except asyncio.CancelledError:
            breaker.cancel_probe()
            raise
        except Exception as e:  # noqa: BLE001 - classified below
            _record_call(label, model, attempt + 1,
                         "rate_limited" if is_rate_limit(e) else ("unavailable" if is_unavailable(e) else "error"),
                         started, error=e)
            if not is_retryable(e):
                breaker.record_success()
                raise
            breaker.record_failure()
            last = e
            if attempt == attempts - 1:
                break
            # Overload clears faster than a per-minute quota: shorter waits for 503s.
            wait = (base_wait if is_rate_limit(e) else max(3, base_wait // 4)) * (attempt + 1)
            logger.warning("%s %s, waiting %ss (attempt %d/%d)", label,
                           "rate limited" if is_rate_limit(e) else f"unavailable ({type(e).__name__})",
                           wait, attempt + 1, attempts)
            await asyncio.sleep(wait)
        else:
            breaker.record_success()
            _record_call(label, model, attempt + 1, "ok", started, result=result)
            return result
    raise last


async def with_fallback(make_call, *, label: str, attempts: int = 3, base_wait: int = 20):
    """make_call(model) -> awaitable. Try MODEL (with retries); if it is still unavailable
    and GEMINI_FALLBACK_MODEL is set, try the fallback (with retries). Returns
    (result, model_that_answered)."""
    try:
        result = await with_retry(lambda: make_call(MODEL), label=label, attempts=attempts,
                                  base_wait=base_wait, breaker=get_breaker(MODEL), model=MODEL)
        note_model(MODEL)
        return result, MODEL
    except Exception as e:  # noqa: BLE001
        # CircuitOpen on the main model also means "try the fallback" (its own breaker decides).
        if not (FALLBACK_MODEL and FALLBACK_MODEL != MODEL and (is_retryable(e) or isinstance(e, CircuitOpen))):
            raise
        logger.warning("%s: %s unavailable after retries (%s); using fallback %s",
                       label, MODEL, type(e).__name__, FALLBACK_MODEL)
    result = await with_retry(lambda: make_call(FALLBACK_MODEL), label=f"{label}/fallback",
                              attempts=attempts, base_wait=base_wait, breaker=get_breaker(FALLBACK_MODEL), model=FALLBACK_MODEL)
    note_model(FALLBACK_MODEL)
    return result, FALLBACK_MODEL


async def generate_text(prompt: str, *, system_instruction: str | None = None,
                        label: str = "LLM", attempts: int = 3, base_wait: int = 35,
                        json_schema=None) -> str:
    """Single-turn text generation used by the summarizer, verifier and translator.
    With `json_schema` (a pydantic model) Gemini is asked for JSON matching it; the caller still
    validates the result, because a schema request is a strong hint, not a guarantee."""
    from google.genai import types

    kwargs = {}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if json_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = json_schema
    config = types.GenerateContentConfig(**kwargs) if kwargs else None

    async def _call(model: str):
        resp = await get_client().aio.models.generate_content(
            model=model, contents=prompt, config=config,
        )
        return (resp.text or "").strip()

    text, _model = await with_fallback(_call, label=label, attempts=attempts, base_wait=base_wait)
    return text
