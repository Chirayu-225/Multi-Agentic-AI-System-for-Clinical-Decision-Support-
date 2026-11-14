"""
src/prompts.py

Prompt versions. A version is a short hash of the prompt's own text, so it changes exactly when
the prompt changes and nobody has to remember to bump a number. Every model call records the
version it used (llm_calls.prompt_version) and every report's agent run records the set
(agent_runs.prompt_versions), so "which prompt produced this summary?" and "did quality change
when we edited the verifier?" are answerable from the database.

Prompts that are f-strings are versioned by the source of the function that builds them: the
text with its placeholders, not any one patient's data.
"""
import hashlib
import inspect
from functools import lru_cache


def fingerprint(*parts) -> str:
    """Hash strings, or the source code of functions/classes, into a 10-character version."""
    h = hashlib.sha256()
    for p in parts:
        text = p if isinstance(p, str) else inspect.getsource(p)
        h.update(text.replace("\r\n", "\n").strip().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:10]


@lru_cache(maxsize=1)
def current_versions() -> dict[str, str]:
    """{agent name: version}, computed once per process from the live prompt text."""
    from src.agents import orchestrator, summarizer, translator, verifier
    return {
        "orchestrator": fingerprint(orchestrator.ORCHESTRATOR_SYSTEM_PROMPT),
        "summarizer": fingerprint(summarizer._build_prompt),
        "verifier": fingerprint(verifier.VERIFIER_SYSTEM_PROMPT, verifier._build_audit_prompt),
        "translator": fingerprint(translator._build_prompt),
    }


def version_of(purpose: str) -> str | None:
    """'summarizer@1a2b3c4d5e' for a call purpose, or None for one we do not version."""
    v = current_versions().get((purpose or "").split("/")[0].lower())
    return f"{purpose.split('/')[0].lower()}@{v}" if v else None
