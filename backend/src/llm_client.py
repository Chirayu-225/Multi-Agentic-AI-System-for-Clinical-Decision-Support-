"""
src/llm_client.py

Shared OpenRouter client used by all three agents (orchestrator, verifier,
summarizer). OpenRouter is OpenAI-API-compatible, so the standard `openai`
SDK is used with a different base_url.

Previously each agent file called google.generativeai.configure()
independently — consolidated here since all three now share one client.
"""

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# openrouter/free is OpenRouter's own auto-router: it picks from whatever
# free models are CURRENTLY available and filters for the capabilities the
# request needs (e.g. tool calling) — this avoids hardcoding one specific
# :free model slug, which can get delisted from the free tier with no
# warning (as happened with meta-llama/llama-3.3-70b-instruct:free here).
# Override via OPENROUTER_MODEL in .env if you want a fixed, specific model
# instead (check current tool-calling-capable free models first at
# https://openrouter.ai/models?max_price=0&supported_parameters=tools).
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
