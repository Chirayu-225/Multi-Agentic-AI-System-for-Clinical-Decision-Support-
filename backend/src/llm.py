"""
src/llm.py

The ONE place this project talks to a model provider: Gemini (google-genai) or Groq (its
OpenAI-compatible HTTP API). LLM_PROVIDER picks one; with no setting, Groq is used when
GROQ_API_KEY is set, otherwise Gemini. Everything below the provider switch (retries, circuit
breaker, rate limiter, call tracing, fallback model) is shared, so both behave the same way.

The rest of this note was written for Gemini and still holds for it.

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
import json
import logging
import os
from typing import Any, NamedTuple

import time

from src import observability as obs
from src.resilience import CircuitBreaker, CircuitOpen, RateLimiter

logger = logging.getLogger("medsight")

# gemini-2.5-flash is not deprecated, but Google limits it to API keys that used it
# before, so a fresh deployment with a new key cannot call it. Default to a current
# stable model; override with GEMINI_MODEL.
def resolve_provider(env=os.environ) -> str:
    """LLM_PROVIDER if set; else groq when a Groq key is present; else gemini."""
    chosen = (env.get("LLM_PROVIDER") or "").strip().lower()
    if chosen in ("gemini", "groq"):
        return chosen
    if chosen:
        raise RuntimeError(f"LLM_PROVIDER must be 'gemini' or 'groq', got {chosen!r}")
    return "groq" if env.get("GROQ_API_KEY") else "gemini"


PROVIDER = resolve_provider()
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"     # free tier, supports tool calling and JSON mode
DEFAULT_MODEL = GROQ_DEFAULT_MODEL if PROVIDER == "groq" else GEMINI_DEFAULT_MODEL
if PROVIDER == "groq":
    MODEL = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
    # Opt-in only: a smaller fallback model (for example llama-3.1-8b-instant, which has a much larger
    # daily allowance) gives different answers, so it is never used unless asked for.
    FALLBACK_MODEL = (os.getenv("GROQ_FALLBACK_MODEL") or "").strip() or None
else:
    MODEL = os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
    FALLBACK_MODEL = (os.getenv("GEMINI_FALLBACK_MODEL") or "").strip() or None
TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS") or os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
_groq_transport = None          # tests inject an httpx.MockTransport here; None means the real network


def required_key_name() -> str:
    return "GROQ_API_KEY" if PROVIDER == "groq" else "GEMINI_API_KEY"

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


class ProviderHTTPError(Exception):
    """The provider answered with an HTTP error status (used for Groq; the Gemini SDK has its own)."""

    def __init__(self, status: int, detail: str = ""):
        super().__init__(f"{status} {_STATUS_WORDS.get(status, 'ERROR')}: {detail[:300]}")
        self.status = status


_STATUS_WORDS = {429: "RESOURCE_EXHAUSTED", 500: "INTERNAL", 502: "BAD_GATEWAY", 503: "UNAVAILABLE",
                 504: "DEADLINE_EXCEEDED"}


def is_rate_limit(err: Exception) -> bool:
    if isinstance(err, ProviderHTTPError):
        return err.status == 429
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def is_unavailable(err: Exception) -> bool:
    """Overloaded / transient server-side failure, or a timeout: worth retrying."""
    if isinstance(err, ProviderHTTPError):
        return err.status in (500, 502, 503, 504)
    try:
        import httpx
        if isinstance(err, httpx.TransportError):      # timeouts, refused or dropped connections
            return True
    except ImportError:                                # pragma: no cover - httpx is a hard requirement
        pass
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


# ── Groq (OpenAI-compatible chat completions over HTTP) ───────────────────────────────────────────

class FunctionCall(NamedTuple):
    name: str
    args: dict
    id: str | None = None


class ToolResult(NamedTuple):
    """A tool's output going back to the model. `call_id` ties it to the call it answers."""
    name: str
    result: Any
    call_id: str | None = None


class _Usage(NamedTuple):
    prompt_token_count: int
    candidates_token_count: int


class GroqResponse:
    """The parts of a reply the rest of the app reads, named as the Gemini SDK names them so the
    tracing and the orchestrator loop need no provider checks."""

    def __init__(self, message: dict, usage: dict | None):
        self.message = message
        self.text = (message.get("content") or "").strip()
        self.function_calls = [_parse_tool_call(tc) for tc in (message.get("tool_calls") or [])]
        u = usage or {}
        self.usage_metadata = _Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))


def _parse_tool_call(tc: dict) -> FunctionCall:
    fn = tc.get("function") or {}
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        args = {}
    return FunctionCall(name=fn.get("name", ""), args=args if isinstance(args, dict) else {}, id=tc.get("id"))


async def _groq_chat(model: str, messages: list[dict], *, tools: list | None = None,
                     json_mode: bool = False) -> GroqResponse:
    import httpx
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    payload: dict = {"model": model, "messages": messages, "temperature": 0.2}
    if tools:
        payload["tools"], payload["tool_choice"] = tools, "auto"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # One client per call: calls can run on different event loops (the app, tests, the eval gate).
    async with httpx.AsyncClient(base_url=GROQ_BASE_URL, timeout=TIMEOUT_SECONDS, transport=_groq_transport) as c:
        r = await c.post("/chat/completions", json=payload, headers={"Authorization": f"Bearer {key}"})
    if r.status_code >= 400:
        raise ProviderHTTPError(r.status_code, r.text)
    body = r.json()
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise ProviderHTTPError(502, "malformed response from provider") from None
    return GroqResponse(message, body.get("usage"))


class GroqChat:
    """A multi-turn conversation with tool calling, shaped like the Gemini chat the orchestrator uses.
    The history only changes after a turn succeeds, so a retried turn never duplicates a message."""

    def __init__(self, model: str, system_instruction: str, tools: list):
        self.model = model
        self.tools = tools
        self.messages: list[dict] = [{"role": "system", "content": system_instruction}]

    async def send_message(self, message) -> GroqResponse:
        if isinstance(message, str):
            new = [{"role": "user", "content": message}]
        else:                                           # a list of ToolResult
            new = [{"role": "tool", "tool_call_id": r.call_id or r.name,
                    "content": json.dumps(r.result, default=str)} for r in message]
        resp = await _groq_chat(self.model, self.messages + new, tools=self.tools)
        reply = {"role": "assistant", "content": resp.message.get("content")}
        if resp.message.get("tool_calls"):
            reply["tool_calls"] = resp.message["tool_calls"]
        self.messages = self.messages + new + [reply]
        return resp


def openai_tools(tool_defs: list[dict]) -> list[dict]:
    """TOOL_DEFINITIONS (name, description, JSON-Schema parameters) in the chat-completions format."""
    return [{"type": "function", "function": {"name": d["name"], "description": d["description"],
                                              "parameters": d["parameters"]}} for d in tool_defs]


async def generate_text(prompt: str, *, system_instruction: str | None = None,
                        label: str = "LLM", attempts: int = 3, base_wait: int = 35,
                        json_schema=None) -> str:
    """Single-turn text generation used by the summarizer, verifier and translator.
    With `json_schema` (a pydantic model) Gemini is asked for JSON matching it; the caller still
    validates the result, because a schema request is a strong hint, not a guarantee."""
    if PROVIDER == "groq":
        system = system_instruction or ""
        if json_schema is not None:
            # Groq's JSON mode guarantees valid JSON, not this shape: state the shape, and the caller validates.
            system = (system + "\n\n" if system else "") + (
                "Respond with ONE JSON object that matches this JSON Schema, and nothing else:\n"
                + json.dumps(json_schema.model_json_schema()))
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

        async def _groq_call(model: str):
            return (await _groq_chat(model, messages, json_mode=json_schema is not None)).text

        text, _model = await with_fallback(_groq_call, label=label, attempts=attempts, base_wait=base_wait)
        return text

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
