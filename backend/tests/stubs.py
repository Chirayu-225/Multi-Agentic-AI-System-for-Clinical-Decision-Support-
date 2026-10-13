"""Install inert stand-ins for heavy optional dependencies (see conftest.py)."""
import sys
import types


def _missing(name: str) -> bool:
    try:
        __import__(name)
        return False
    except Exception:
        return True


class Anything:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return Anything()
    def __getattr__(self, item): return Anything()


def _mod(name: str, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)
    return sys.modules[name]


def install_stubs() -> None:
    if _missing("google.generativeai"):
        genai = _mod("google.generativeai", configure=lambda **k: None, GenerativeModel=Anything, protos=Anything())
        if "google" in sys.modules:
            sys.modules["google"].generativeai = genai
    if _missing("easyocr"):
        _mod("easyocr", Reader=Anything)
    if _missing("chromadb"):
        _mod("chromadb", PersistentClient=Anything)
        _mod("chromadb.config", Settings=Anything)
    if _missing("sentence_transformers"):
        _mod("sentence_transformers", SentenceTransformer=Anything)
    if _missing("gliner"):
        _mod("gliner", GLiNER=Anything)
    if _missing("langdetect"):
        _mod("langdetect", detect=lambda *_a, **_k: "en", LangDetectException=Exception)
    if _missing("indic_transliteration"):
        ss = _mod("indic_transliteration.sanscript", DEVANAGARI="devanagari", IAST="iast",
                  transliterate=lambda text, *_a, **_k: text)
        _mod("indic_transliteration", sanscript=ss)
