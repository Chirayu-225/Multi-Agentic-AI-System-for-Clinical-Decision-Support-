"""
eval/frozen_gate.py - a small, frozen, on-demand check for prompt and model changes.

    python -m eval.frozen_gate --build                 # (re)create eval/frozen_set.json (deterministic, committed)
    python -m eval.frozen_gate --dry-run               # show what would run and how many calls it would use
    python -m eval.frozen_gate                         # run only what the change touches (see below)
    python -m eval.frozen_gate --all                   # run everything regardless
    python -m eval.frozen_gate --update-baseline       # accept the current results as the new baseline

Why it is shaped this way (free-tier quota, no paid services):
  * FROZEN: the cases live in a committed file, so two runs compare the same inputs.
  * SMALL: 8 cases -> about 16 verifier calls + 4 summarizer calls. A full gate is ~20 calls.
  * TARGETED: each component's prompt has a version (src/prompts.py). The baseline records the
    versions and the model it was measured with; only components whose prompt version changed (or
    everything, if the model changed) are re-run. Editing the summarizer does not spend the
    verifier's calls.
  * ON DEMAND: never run by CI; you run it before merging a prompt/model change.
  * QUOTA-GUARDED: refuses to start if the plan exceeds GATE_MAX_CALLS (default 40) or today's
    ledger (eval/results/gate_ledger.json) would pass GATE_DAILY_CALLS (default 100), and stops
    mid-run if the real call count (repairs included) reaches the cap.
  * A regression fails the run (exit 1). Granularity is one case: with 8 cases a drop of more
    than GATE_TOLERANCE_CASES (default 1) cases is a regression. This is a smoke test of
    direction, not a statistical benchmark, and says so in its output.

Data is synthetic (eval/synthetic.py); nothing here is clinical validation.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN_PATH = ROOT / "eval" / "frozen_set.json"
BASELINE_PATH = ROOT / "eval" / "gate_baseline.json"
LEDGER_PATH = ROOT / "eval" / "results" / "gate_ledger.json"

N_CASES = 8            # verifier cases (each costs a clean + a corrupted call)
N_SUMMARY_CASES = 4    # summarizer cases (one call each)
SEED = 7

# metric -> (direction, component). "up": higher is better.
METRICS = {
    "verifier.catch_rate": ("up", "verifier"),
    "verifier.false_alarm_rate": ("down", "verifier"),
    "verifier.unverified_rate": ("down", "verifier"),
    "summarizer.ungrounded_rate": ("down", "summarizer"),
}


class ProviderDown(Exception):
    """Several calls in a row failed: the run is invalid and further calls would only burn quota."""


class QuotaGuard(Exception):
    pass


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# ── the frozen set ─────────────────────────────────────────────────────────────

def build_frozen_set(path: Path = FROZEN_PATH) -> dict:
    from eval.run_eval import _num, _truth_summary          # noqa: PLC0415 (heavy import, build only)
    from eval.synthetic import generate_dataset
    from src.flagging.risk_flagging import flag_lab_values

    cases = []
    for c in generate_dataset(40, SEED):
        if len(cases) >= N_CASES:
            break
        if not c.labs:
            continue
        flags = flag_lab_values([{"test": l.display, "value": l.value, "unit": l.unit} for l in c.labs], c.gender)
        clean = _truth_summary(c)
        lab = c.labs[0]
        bad = clean.replace(f"{lab.display} is {_num(lab.value)}",
                            f"{lab.display} is {_num(round(lab.value * 1.4 + 0.4, 2))}")
        if bad == clean:
            continue
        cases.append({
            "id": f"frozen-{len(cases) + 1}",
            "entities": {"patient": {}, "clinical": {"diagnoses": [c.diagnosis]},
                         "medications": [{"name": m} for m in c.meds],
                         "lab_values": [{"test": l.display, "value": l.value, "unit": l.unit} for l in c.labs]},
            "flags": [{**{k: v for k, v in f.items() if k != "severity"},
                       "severity": str(f["severity"]).replace("Severity.", "")} for f in flags],
            "summary_clean": clean, "summary_novel_value": bad,
        })
    doc = {"_comment": "FROZEN. Do not edit by hand: regenerate with --build, then --update-baseline.",
           "seed": SEED, "cases": cases}
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


def load_frozen(path: Path = FROZEN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── planning ───────────────────────────────────────────────────────────────────

def current_state() -> dict:
    from src import llm, prompts
    return {"prompt_versions": prompts.current_versions(), "model": llm.MODEL}


def plan(baseline: dict | None, state: dict, force_all: bool = False) -> dict:
    """Which components to run, and why."""
    reasons: dict[str, str] = {}
    for comp in ("verifier", "summarizer"):
        if force_all:
            reasons[comp] = "--all"
        elif not baseline:
            reasons[comp] = "no baseline yet"
        elif baseline.get("model") != state["model"]:
            reasons[comp] = f"model changed ({baseline.get('model')} -> {state['model']})"
        elif baseline.get("prompt_versions", {}).get(comp) != state["prompt_versions"].get(comp):
            reasons[comp] = (f"{comp} prompt changed ({baseline.get('prompt_versions', {}).get(comp)} -> "
                             f"{state['prompt_versions'].get(comp)})")
    calls = 0
    if "verifier" in reasons:
        calls += 2 * N_CASES
    if "summarizer" in reasons:
        calls += N_SUMMARY_CASES
    return {"run": reasons, "planned_calls": calls}


# ── quota ──────────────────────────────────────────────────────────────────────

def _ledger_used_today(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get(date.today().isoformat(), 0))
    except (OSError, ValueError):
        return 0


def _ledger_add(path: Path, n: int) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    today = date.today().isoformat()
    data = {today: int(data.get(today, 0)) + n}            # keep only today: the file stays tiny
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class CallCounter:
    """Counts real model calls (repairs included) and stops the run at the cap.

    The agents deliberately swallow every exception (they fail closed), so a raise from inside a
    call would be absorbed as "the model failed". The wrapper therefore refuses calls and sets a
    flag, and the measurement loops call check() between cases, outside the agents, to abort."""

    def __init__(self, cap: int):
        self.cap, self.used, self.exhausted = cap, 0, False
        self.errors: list[str] = []                  # provider/transport failures, not bad model output
        self.streak, self.aborted = 0, False         # consecutive failures; set when the provider is plainly down
        self.max_streak = _env_int("GATE_MAX_CONSECUTIVE_ERRORS", 3)
        self.served: dict[str, int] = {}             # which model actually answered each successful call

    def wrap(self, generate):
        async def counted(*a, **k):
            if self.aborted:
                raise RuntimeError("gate aborted: provider unavailable")   # costs nothing, spends no quota
            if self.used >= self.cap:
                self.exhausted = True
                raise RuntimeError("gate call cap reached")      # absorbed by the agent; check() aborts
            self.used += 1
            from src import llm
            answered = llm.begin_model_tracking()   # filled by with_fallback with the model that replied
            try:
                out = await generate(*a, **k)
                self.streak = 0
                for m in (answered or {"unknown"}):  # "unknown": a stand-in that does not report a model
                    self.served[m] = self.served.get(m, 0) + 1
                return out
            except Exception as e:                    # the agents absorb this and fail closed; we must not
                self.errors.append(f"{type(e).__name__}: {e}"[:200])
                if type(e).__name__ == "CircuitOpen":
                    self.used -= 1                    # refused locally: never reached the provider's quota
                self.streak += 1
                if self.streak >= self.max_streak:
                    self.aborted = True
                raise
        return counted

    def check(self) -> None:
        if self.aborted:
            raise ProviderDown(f"{self.streak} calls in a row failed; stopped to save quota")
        if self.exhausted:
            raise QuotaGuard(f"call cap of {self.cap} reached mid-run")


# ── measurement ────────────────────────────────────────────────────────────────

async def measure_verifier(frozen: dict, generate, check=lambda: None) -> dict:
    import src.agents.verifier as verifier
    original = verifier.generate_text
    verifier.generate_text = generate
    caught = false_alarm = unverified = total = 0
    try:
        for c in frozen["cases"]:
            for kind, summary in (("novel", c["summary_novel_value"]), ("clean", c["summary_clean"])):
                out = await verifier.verify_and_correct(summary, "Your results were reviewed.",
                                                        c["entities"], c["flags"], [])
                check()
                total += 1
                if out["clinical_unverified"]:
                    unverified += 1
                elif out["clinical_corrected"]:
                    if kind == "novel":
                        caught += 1
                    else:
                        false_alarm += 1
    finally:
        verifier.generate_text = original
    n = len(frozen["cases"])
    return {"verifier.catch_rate": round(caught / n, 4), "verifier.false_alarm_rate": round(false_alarm / n, 4),
            "verifier.unverified_rate": round(unverified / total, 4)}


async def measure_summarizer(frozen: dict, generate, check=lambda: None) -> dict:
    import src.agents.summarizer as summarizer
    from src.evidence.grounding import collect_allowed_numbers, find_ungrounded_numbers
    original = summarizer.generate_text
    summarizer.generate_text = generate
    bad = 0
    cases = frozen["cases"][:N_SUMMARY_CASES]
    try:
        for c in cases:
            out = await summarizer.generate_summaries(c["entities"], c["flags"], [], [], [], [])
            check()
            allowed = collect_allowed_numbers(c["entities"], c["flags"], [], [], [], [], [])
            bad += bool(find_ungrounded_numbers(out["clinical_summary"], allowed)
                        or find_ungrounded_numbers(out["patient_summary"], allowed))
    finally:
        summarizer.generate_text = original
    return {"summarizer.ungrounded_rate": round(bad / len(cases), 4)}


def compare(baseline_metrics: dict, new_metrics: dict, tolerance_cases: int) -> list[str]:
    """Regressions: a metric worse than baseline by more than `tolerance_cases` cases."""
    problems = []
    for name, value in new_metrics.items():
        if name not in baseline_metrics:
            continue
        direction, comp = METRICS[name]
        n = N_CASES if comp == "verifier" else N_SUMMARY_CASES
        slack = tolerance_cases / n + 1e-9
        base = baseline_metrics[name]
        worse = (base - value) if direction == "up" else (value - base)
        if worse > slack:
            problems.append(f"{name}: {base} -> {value} (worse by {worse:.3f}, allowed {slack:.3f})")
    return problems


async def run_gate(*, generate, frozen_path=FROZEN_PATH, baseline_path=BASELINE_PATH, ledger_path=LEDGER_PATH,
                   force_all=False, dry_run=False, update_baseline=False, state: dict | None = None) -> dict:
    frozen = load_frozen(frozen_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    state = state or current_state()
    pl = plan(baseline, state, force_all)
    report = {"plan": pl, "state": state, "baseline_found": baseline is not None,
              "note": "8 frozen synthetic cases: a direction check, not a benchmark or clinical validation."}
    if dry_run or not pl["run"]:
        report["status"] = "dry-run" if dry_run else "nothing-to-run"
        return report

    max_calls = _env_int("GATE_MAX_CALLS", 40)
    daily = _env_int("GATE_DAILY_CALLS", 100)
    used_today = _ledger_used_today(ledger_path)
    if pl["planned_calls"] > max_calls:
        raise QuotaGuard(f"plan needs {pl['planned_calls']} calls, GATE_MAX_CALLS is {max_calls}")
    if used_today + pl["planned_calls"] > daily:
        raise QuotaGuard(f"{used_today} calls already used today; this run needs {pl['planned_calls']} "
                         f"and GATE_DAILY_CALLS is {daily}")

    counter = CallCounter(max_calls)
    guarded = counter.wrap(generate)
    metrics: dict = {}
    try:
        if "verifier" in pl["run"]:
            metrics.update(await measure_verifier(frozen, guarded, counter.check))
        if "summarizer" in pl["run"]:
            metrics.update(await measure_summarizer(frozen, guarded, counter.check))
    except ProviderDown:
        pass                                          # reported below as an invalid run
    finally:
        _ledger_add(ledger_path, counter.used)           # real spend is recorded even on failure
    report["calls_used"] = counter.used
    report["metrics"] = metrics
    report["served_models"] = dict(counter.served)
    off_model = sum(n for m, n in counter.served.items() if m not in (state["model"], "unknown"))
    report["off_primary_calls"] = off_model
    if counter.errors:
        # The agents swallow call failures, so the metrics above describe a fallback path, not the
        # model. A gate that cannot run must not report "pass" or write a baseline.
        report["call_errors"] = len(counter.errors)
        report["first_error"] = counter.errors[0]
        report["regressions"] = []
        report["status"] = "invalid"
        if update_baseline:
            report["baseline_updated"] = False
        return report

    allowed_off = _env_int("GATE_MAX_FALLBACK_CALLS", 2)
    if off_model > allowed_off:
        # The numbers describe another model than the one this baseline is named after. Comparing them
        # to a baseline, or saving them as one, would be a false statement.
        report["regressions"] = []
        report["status"] = "off-model"
        report["allowed_off_primary_calls"] = allowed_off
        if update_baseline:
            report["baseline_updated"] = False
        return report

    problems = compare((baseline or {}).get("metrics", {}), metrics, _env_int("GATE_TOLERANCE_CASES", 1))
    report["regressions"] = problems
    report["status"] = "regression" if problems else "pass"
    if update_baseline and not problems:
        merged = {**(baseline or {}).get("metrics", {}), **metrics}
        baseline_path.write_text(json.dumps({"model": state["model"], "prompt_versions": state["prompt_versions"],
                                             "metrics": merged}, indent=1), encoding="utf-8")
        report["baseline_updated"] = True
    elif update_baseline:
        report["baseline_updated"] = False
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)
    if args.build:
        doc = build_frozen_set()
        print(f"wrote {FROZEN_PATH} with {len(doc['cases'])} cases")
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from src.llm import generate_text
    try:
        report = asyncio.run(run_gate(generate=generate_text, force_all=args.all, dry_run=args.dry_run,
                                      update_baseline=args.update_baseline))
    except QuotaGuard as e:
        print(f"REFUSED (quota guard): {e}")
        return 2
    print(json.dumps(report, indent=1))
    if report["status"] == "invalid":
        print(f"INVALID RUN: {report['call_errors']} of {report['calls_used']} calls failed "
              f"(first: {report['first_error']}). Nothing was measured; no baseline written.")
        return 3
    if report["status"] == "off-model":
        print(f"OFF-MODEL RUN: {report['off_primary_calls']} of {report['calls_used']} calls were answered by "
              f"{', '.join(sorted(m for m in report['served_models'] if m != report['state']['model']))}, not "
              f"{report['state']['model']} (allowed: {report['allowed_off_primary_calls']}). "
              f"No baseline written. Run again when the primary model is healthy.")
        return 4
    return 1 if report["status"] == "regression" else 0


if __name__ == "__main__":
    sys.exit(main())
