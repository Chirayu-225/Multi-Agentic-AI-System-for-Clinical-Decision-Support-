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
  produced them. GEMINI_MODEL picks it; the id is recorded on every AgentRun so a
  report can always be traced to the model that wrote it.
"""

import asyncio
import logging
import os

logger = logging.getLogger("medsight")

# gemini-2.5-flash is not deprecated, but Google limits it to API keys that used it
# before, so a fresh deployment with a new key cannot call it. Default to a current
# stable model; override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.8-flash"
MODEL = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

_client = None


def get_client():
    """Lazily build one shared client (import + key check happen on first use)."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def is_rate_limit(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


async def with_retry(call, *, label: str, attempts: int = 3, base_wait: int = 20):
    """Run an async callable, backing off on rate limits only. Other errors propagate."""
    last = None
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as e:  # noqa: BLE001 - classified below
            if not is_rate_limit(e):
                raise
            last = e
            wait = base_wait * (attempt + 1)
            logger.warning("%s rate limited, waiting %ss (attempt %d/%d)",
                           label, wait, attempt + 1, attempts)
            await asyncio.sleep(wait)
    raise last


async def generate_text(prompt: str, *, system_instruction: str | None = None,
                        label: str = "LLM", attempts: int = 3, base_wait: int = 35) -> str:
    """Single-turn text generation used by the summarizer, verifier and translator."""
    from google.genai import types

    config = types.GenerateContentConfig(system_instruction=system_instruction) \
        if system_instruction else None

    async def _call():
        resp = await get_client().aio.models.generate_content(
            model=MODEL, contents=prompt, config=config,
        )
        return (resp.text or "").strip()

    return await with_retry(_call, label=label, attempts=attempts, base_wait=base_wait)
