"""Run the vertical slice N times and report how often it actually works.

One successful run proves the path exists. It says nothing about how often the
agent takes it. This harness answers the narrower question the README leaves
open: over N cold starts against the same mock ATS, how many applications reach
a *confirmed* submission, and what does the agent spend to get there.

Each run is a fresh subprocess with its own data directory, database, browser
profile and artifact store, so nothing -- not a cached model decision, not a
deduplicated job, not a leftover resume -- carries between runs. The mock ATS is
reset at the start of every run by the slice itself.

    python -m cvnator.mockats                        # terminal 1
    python scripts/measure_reliability.py --runs 5   # terminal 2

Results land in ``data/reliability/<timestamp>/`` as one JSON file per run plus
``summary.json``. Failures are kept, not discarded: a run that fails is the
interesting one.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLICE = ROOT / "scripts" / "run_vertical_slice.py"

# Rates the spec asks the agent to be judged on (section 74), expressed as
# (label, numerator metric, denominator metric). A denominator of zero means the
# run never got far enough for the rate to mean anything; it is skipped, not
# reported as 0.0.
RATES = [
    ("human_intervention_rate", "human_interventions", "applications_started"),
    ("vision_fallback_rate", "vision_fallbacks", "agent_steps_total"),
    ("tool_retry_rate", "tool_retries_total", "tool_calls_total"),
    ("steps_per_application", "agent_steps_total", "applications_started"),
    ("tool_calls_per_application", "tool_calls_total", "applications_started"),
]

COUNTERS = [
    "jobs_discovered",
    "jobs_matched",
    "jobs_rejected",
    "applications_started",
    "applications_completed",
    "applications_failed",
    "applications_waiting_user",
    "resumes_generated",
    "invalid_answers_blocked",
    # Absolute totals matter alongside the per-application rates: a run that spends
    # 58 steps and one that spends 12 both read as "one application".
    "agent_steps_total",
    "tool_calls_total",
]


def run_once(index: int, out_dir: Path, timeout_s: float) -> dict:
    """Execute one isolated slice run and return its parsed result."""
    run_dir = out_dir / f"run_{index:02d}"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = (data_dir / "cvnator.sqlite3").resolve().as_posix()

    env = {
        **os.environ,
        "CVNATOR_DATA_DIR": str(data_dir.resolve()),
        "CVNATOR_DB_URL": f"sqlite+pysqlite:///{db_path}",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }

    migrate = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if migrate.returncode != 0:
        return {
            "run": index,
            "succeeded": False,
            "failure": "migration failed",
            "stderr": migrate.stderr[-2000:],
        }

    json_out = run_dir / "result.json"
    log_path = run_dir / "run.log"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, str(SLICE), "--json-out", str(json_out)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        log_path.write_text(
            completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8"
        )
        exit_code: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            (exc.stdout or "") + "\n--- timed out ---\n" + (exc.stderr or ""),
            encoding="utf-8",
        )
        exit_code = None

    elapsed = round(time.monotonic() - started, 1)

    if not json_out.exists():
        return {
            "run": index,
            "succeeded": False,
            "failure": "timed out" if exit_code is None else f"exit {exit_code}",
            "wall_seconds": elapsed,
            "log": str(log_path),
        }

    result = json.loads(json_out.read_text(encoding="utf-8"))
    result["run"] = index
    result["exit_code"] = exit_code
    result["log"] = str(log_path)
    return result


def summarise(results: list[dict]) -> dict:
    successes = [r for r in results if r.get("succeeded")]
    summary: dict = {
        "runs": len(results),
        "succeeded": len(successes),
        "success_rate": round(len(successes) / len(results), 3) if results else 0.0,
        "failures": [
            {"run": r["run"], "reason": r.get("failure", "no confirmed submission")}
            for r in results
            if not r.get("succeeded")
        ],
        "counters": {},
        "rates": {},
        "wall_seconds": {},
    }

    def spread(values: list[float]) -> dict[str, float]:
        return {
            "min": round(min(values), 3),
            "median": round(statistics.median(values), 3),
            "max": round(max(values), 3),
            "mean": round(statistics.fmean(values), 3),
        }

    walls = [r["wall_seconds"] for r in results if "wall_seconds" in r]
    if walls:
        summary["wall_seconds"] = spread(walls)

    with_metrics = [r for r in results if "metrics" in r]
    for name in COUNTERS:
        values = [float(r["metrics"].get(name, 0)) for r in with_metrics]
        if values:
            summary["counters"][name] = spread(values)

    for label, numerator, denominator in RATES:
        values = [
            r["metrics"][numerator] / r["metrics"][denominator]
            for r in with_metrics
            if r["metrics"].get(denominator)
        ]
        if values:
            summary["rates"][label] = spread(values) | {"runs_counted": len(values)}

    # Every distinct terminal state the applications ended in, across all runs.
    states: dict[str, int] = {}
    for result in results:
        for application in result.get("applications", []):
            states[application["state"]] = states.get(application["state"], 0) + 1
    summary["application_states"] = dict(sorted(states.items()))

    # A tailored CV left on disk is only a leak when its application was submitted.
    # An application that failed keeps its file deliberately: deleting the CV for an
    # unsent application is the thing the cleanup path is forbidden to do.
    summary["runs_leaking_temp_files"] = sum(
        1
        for result in with_metrics
        if result.get("leftover_temp_files")
        and all(a["state"] == "SUBMITTED" for a in result.get("applications", []))
    )

    return summary


def print_report(summary: dict, results: list[dict]) -> None:
    print("\n" + "=" * 66)
    print(f"  {summary['succeeded']}/{summary['runs']} runs reached a confirmed submission")
    print("=" * 66)

    for result in results:
        mark = "ok  " if result.get("succeeded") else "FAIL"
        refs = ", ".join(
            str(entry.get("reference", "")) for entry in result.get("submitted", {}).values()
        )
        detail = refs or result.get("failure", "no confirmed submission")
        steps = result.get("metrics", {}).get("agent_steps_total", "-")
        wall = result.get("wall_seconds", "-")
        print(f"  run {result['run']:02d}  {mark}  {wall:>6}s  {steps:>3} steps  {detail}")

    if summary["rates"]:
        print("\n  rate                          min    median     max      mean")
        print("  " + "-" * 62)
        for label, values in summary["rates"].items():
            print(
                f"  {label:<26}"
                f"{values['min']:>7.3f}{values['median']:>10.3f}"
                f"{values['max']:>8.3f}{values['mean']:>10.3f}"
            )

    if summary["counters"]:
        print("\n  counter                       min    median     max      mean")
        print("  " + "-" * 62)
        for label, values in summary["counters"].items():
            print(
                f"  {label:<26}"
                f"{values['min']:>7.1f}{values['median']:>10.1f}"
                f"{values['max']:>8.1f}{values['mean']:>10.1f}"
            )

    print(f"\n  application states: {summary['application_states']}")
    print(
        "  runs leaving a tailored CV behind after submitting: "
        f"{summary['runs_leaking_temp_files']}"
    )
    if summary["failures"]:
        print("\n  failures:")
        for failure in summary["failures"]:
            print(f"    run {failure['run']:02d}: {failure['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="How many runs to execute.")
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="Per-run wall clock budget in seconds."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for run artifacts (default: data/reliability/<timestamp>).",
    )
    args = parser.parse_args()

    out_dir = args.out or (
        ROOT / "data" / "reliability" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing run artifacts to {out_dir}")

    results: list[dict] = []
    for index in range(1, args.runs + 1):
        print(f"\n--- run {index}/{args.runs} ---", flush=True)
        result = run_once(index, out_dir, args.timeout)
        status = "confirmed submission" if result.get("succeeded") else result.get(
            "failure", "no confirmed submission"
        )
        print(f"    {status}", flush=True)
        results.append(result)
        # Write the summary as we go: a crash on run 4 must not lose runs 1-3.
        summary = summarise(results)
        (out_dir / "summary.json").write_text(
            json.dumps({"summary": summary, "runs": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print_report(summarise(results), results)
    print(f"\nFull results: {out_dir / 'summary.json'}")
    return 0 if all(r.get("succeeded") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
