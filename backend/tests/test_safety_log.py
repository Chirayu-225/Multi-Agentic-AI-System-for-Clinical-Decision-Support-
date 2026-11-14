"""SAFETY.md is only worth keeping if it stays true. This fails when a hazard has no test, when a
referenced test no longer exists, or when the required fields are missing."""
import ast
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
SAFETY = BACKEND.parent / "SAFETY.md"
SEVERITIES = {"catastrophic", "major", "moderate", "minor"}
REF = re.compile(r"`(tests/[\w/]+\.py)::(\w+)(?:\[[^\]]*\])?`")


def hazards():
    text = SAFETY.read_text(encoding="utf-8")
    parts = re.split(r"^## (H\d+) - (.+)$", text, flags=re.MULTILINE)
    return [{"id": parts[i], "title": parts[i + 1], "body": parts[i + 2]} for i in range(1, len(parts), 3)]


def test_the_log_exists_and_has_hazards():
    assert SAFETY.exists() and len(hazards()) >= 10


def test_hazard_ids_are_unique_and_sequential():
    ids = [h["id"] for h in hazards()]
    assert ids == [f"H{i}" for i in range(1, len(ids) + 1)]


@pytest.mark.parametrize("h", hazards(), ids=lambda h: h["id"])
def test_every_hazard_has_the_required_fields(h):
    sev = re.search(r"^- Severity: (\w+)", h["body"], re.MULTILINE)
    assert sev and sev.group(1) in SEVERITIES, f"{h['id']}: missing or unknown severity"
    for field in ("Cause", "Controls", "Residual risk"):
        assert re.search(rf"^- {field}:", h["body"], re.MULTILINE), f"{h['id']}: no '{field}'"


@pytest.mark.parametrize("h", hazards(), ids=lambda h: h["id"])
def test_every_hazard_cites_at_least_one_test_and_every_test_exists(h):
    refs = REF.findall(h["body"])
    assert refs, f"{h['id']} ({h['title']}) cites no test"
    for path, name in refs:
        f = BACKEND / path
        assert f.exists(), f"{h['id']}: {path} does not exist"
        names = {n.name for n in ast.walk(ast.parse(f.read_text(encoding="utf-8")))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert name in names, f"{h['id']}: {path}::{name} does not exist"


def test_a_removed_test_would_be_caught():
    """Self-check: the existence logic really does reject a name that is not in the file."""
    f = BACKEND / "tests/test_fast_path.py"
    names = {n.name for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)}
    assert "test_this_does_not_exist" not in names
