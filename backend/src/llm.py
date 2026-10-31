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

logger = logging.getLogger("medsight")

# gemini-2.5-flash is not deprecated, but Google limits it to API keys that used it
# before, so a fresh deployment with a new key cannot call it. Default to a current
# stable model; override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.8-flash"
MODEL = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
FALLBACK_MODEL = (os.getenv("GEMINI_FALLBACK_MODEL") or "").strip() or None
TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))

_client = None

# Models that actually answered during the current request (set by begin_model_tracking).
_models_used: contextvars.ContextVar[set | None] = contextvars.ContextVar("models_used", default=None)


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


async def with_retry(call, *, label: str, attempts: int = 3, base_wait: int = 20):
    """Run an async callable, backing off on rate limits, overload and timeouts only.
    Other errors (bad key, bad request, invalid model) propagate immediately."""
    last = None
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as e:  # noqa: BLE001 - classified below
            if not is_retryable(e):
                raise
            last = e
            if attempt == attempts - 1:
                break
            # Overload clears faster than a per-minute quota: shorter waits for 503s.
            wait = (base_wait if is_rate_limit(e) else max(3, base_wait // 4)) * (attempt + 1)
            logger.warning("%s %s, waiting %ss (attempt %d/%d)", label,
                           "rate limited" if is_rate_limit(e) else f"unavailable ({type(e).__name__})",
                           wait, attempt + 1, attempts)
            await asyncio.sleep(wait)
    raise last


async def with_fallback(make_call, *, label: str, attempts: int = 3, base_wait: int = 20):
    """make_call(model) -> awaitable. Try MODEL (with retries); if it is still unavailable
    and GEMINI_FALLBACK_MODEL is set, try the fallback (with retries). Returns
    (result, model_that_answered)."""
    try:
        result = await with_retry(lambda: make_call(MODEL), label=label,
                                  attempts=attempts, base_wait=base_wait)
        note_model(MODEL)
        return result, MODEL
    except Exception as e:  # noqa: BLE001
        if not (FALLBACK_MODEL and FALLBACK_MODEL != MODEL and is_retryable(e)):
            raise
        logger.warning("%s: %s unavailable after retries (%s); using fallback %s",
                       label, MODEL, type(e).__name__, FALLBACK_MODEL)
    result = await with_retry(lambda: make_call(FALLBACK_MODEL), label=f"{label}/fallback",
                              attempts=attempts, base_wait=base_wait)
    note_model(FALLBACK_MODEL)
    return result, FALLBACK_MODEL


async def generate_text(prompt: str, *, system_instruction: str | None = None,
                        label: str = "LLM", attempts: int = 3, base_wait: int = 35) -> str:
    """Single-turn text generation used by the summarizer, verifier and translator."""
    from google.genai import types

    config = types.GenerateContentConfig(system_instruction=system_instruction) \
        if system_instruction else None

    async def _call(model: str):
        resp = await get_client().aio.models.generate_content(
            model=model, contents=prompt, config=config,
        )
        return (resp.text or "").strip()

    text, _model = await with_fallback(_call, label=label, attempts=attempts, base_wait=base_wait)
    return text
