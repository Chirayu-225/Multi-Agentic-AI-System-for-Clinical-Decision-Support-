"""
scripts/check_gemini.py - check that this machine can make every kind of Gemini call the app uses.

    python scripts\\check_gemini.py

Runs, with the key and model from backend/.env:
  1. plain text generation   (summarizer, verifier)
  2. a translation            (translator, same code path as the app)
  3. a two-turn tool call     (orchestrator; needs thought signatures on Gemini 3 models)
and prints the FULL traceback of anything that fails. The API key itself is never printed.
"""
import asyncio
import os
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND / ".env", override=True)


def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


async def _plain():
    from src.llm import generate_text
    out = await generate_text("Reply with exactly the word: OK", label="CHECK", attempts=1)
    return out[:80]


async def _translate():
    from src.agents.translator import _gemini_generate
    out = await _gemini_generate("Translate into Hindi, output only the translation: Your hemoglobin is 9.2 g/dL.")
    return out[:80]


async def _tool_call():
    from google.genai import types
    from src.llm import MODEL, get_client

    tool = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="get_reference_range", description="Look up the normal range for a lab test.",
        parameters_json_schema={"type": "object", "properties": {"test": {"type": "string"}},
                                "required": ["test"]})])
    cfg = types.GenerateContentConfig(
        tools=[tool], automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    chat = get_client().aio.chats.create(model=MODEL, config=cfg)
    r1 = await chat.send_message("Use the tool to look up the normal range for hemoglobin, then answer.")
    calls = list(r1.function_calls or [])
    if not calls:
        return "model answered without calling the tool (allowed, but the multi-turn path was not exercised)"
    parts = [types.Part.from_function_response(name=c.name, response={"range": "12-15 g/dL"}) for c in calls]
    r2 = await chat.send_message(parts)      # fails here with a 400 if thought signatures are dropped
    return f"tool called ({calls[0].name}), follow-up answered: {(r2.text or '')[:60]!r}"


async def main() -> int:
    from src.llm import MODEL
    print(f"python          {sys.version.split()[0]}")
    print(f"google-genai    {_ver('google-genai')}")
    print(f"legacy SDK      google-generativeai {_ver('google-generativeai')} (unused; safe to uninstall)")
    print(f"model           {MODEL}")
    print(f"api key set     {'yes' if os.getenv('GEMINI_API_KEY') else 'NO - set GEMINI_API_KEY in backend/.env'}")
    print()
    failed = 0
    for name, fn in (("1. plain generation", _plain), ("2. translation", _translate),
                     ("3. two-turn tool call", _tool_call)):
        try:
            print(f"{name:<24} OK   {await fn()}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"{name:<24} FAILED")
            traceback.print_exc()
            print()
    print("\nAll Gemini calls work." if not failed else f"\n{failed} check(s) failed - paste the output above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
