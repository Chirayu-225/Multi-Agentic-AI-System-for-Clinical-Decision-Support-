"""The --live Verifier evaluation can't run in CI (needs a key). This checks its plumbing against a
fake Gemini model so a broken harness is caught without spending API calls."""
import eval.run_eval as ev
from eval.synthetic import generate_dataset


def test_live_verifier_eval_plumbing(monkeypatch):
    import src.agents.verifier as verifier

    class FakeResp:
        text = ("VERDICT_CLINICAL: FAIL\nISSUES_CLINICAL: bad\nCORRECTED_CLINICAL: fixed\n"
                "VERDICT_PATIENT: PASS\nISSUES_PATIENT: none\n")

    class FakeModel:
        def __init__(self, *a, **k): pass
        def generate_content(self, prompt): return FakeResp()

    monkeypatch.setattr(verifier.genai, "GenerativeModel", FakeModel, raising=False)
    out = ev.eval_live_verifier(generate_dataset(6, 1), limit=4)
    assert out["n_per_kind"] == 4
    assert out["novel_value"] == 1.0 and out["clean"] == 1.0      # fake flags everything
    assert "note" in out
