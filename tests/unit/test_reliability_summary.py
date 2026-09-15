"""The reliability harness must not flatter the agent.

`measure_reliability.summarise` is the only part of the harness that interprets
results, so it is the only part that can lie. These tests pin the two ways it
could: counting a run that never reached the denominator as a perfect 0.0 rate,
and treating a crashed run as anything other than a failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_reliability.py"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_reliability", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    index: int,
    *,
    succeeded: bool = True,
    steps: int = 20,
    started: int = 1,
    interventions: int = 0,
    leftovers: list[str] | None = None,
    state: str = "SUBMITTED",
) -> dict:
    return {
        "run": index,
        "succeeded": succeeded,
        "wall_seconds": 100.0,
        "metrics": {
            "agent_steps_total": steps,
            "applications_started": started,
            "human_interventions": interventions,
            "tool_calls_total": steps,
            "tool_retries_total": 0,
            "vision_fallbacks": 0,
            "jobs_discovered": 3,
        },
        "applications": [{"state": state}],
        "leftover_temp_files": leftovers or [],
    }


def test_success_rate_counts_only_confirmed_runs(harness: ModuleType) -> None:
    summary = harness.summarise([_run(1), _run(2, succeeded=False), _run(3)])

    assert summary["succeeded"] == 2
    assert summary["success_rate"] == pytest.approx(0.667, abs=0.001)
    assert [failure["run"] for failure in summary["failures"]] == [2]


def test_crashed_run_without_metrics_is_a_failure_not_a_gap(harness: ModuleType) -> None:
    crashed = {"run": 2, "succeeded": False, "failure": "timed out", "wall_seconds": 1800.0}

    summary = harness.summarise([_run(1), crashed])

    assert summary["runs"] == 2
    assert summary["succeeded"] == 1
    assert summary["failures"] == [{"run": 2, "reason": "timed out"}]
    # The crashed run contributes no metrics, but it must still divide the rate.
    assert summary["rates"]["steps_per_application"]["runs_counted"] == 1


def test_a_run_that_never_started_an_application_is_skipped_not_scored_zero(
    harness: ModuleType,
) -> None:
    """Zero applications means the rate is undefined, not perfect."""
    aborted = _run(2, succeeded=False, started=0, steps=0)

    summary = harness.summarise([_run(1, interventions=1), aborted])

    rate = summary["rates"]["human_intervention_rate"]
    assert rate["runs_counted"] == 1
    assert rate["mean"] == pytest.approx(1.0)
    assert rate["min"] == pytest.approx(1.0)


def test_application_states_are_pooled_across_runs(harness: ModuleType) -> None:
    summary = harness.summarise(
        [_run(1), _run(2, succeeded=False, state="FAILED"), _run(3)]
    )

    assert summary["application_states"] == {"FAILED": 1, "SUBMITTED": 2}


def test_leaked_temporary_files_are_counted_per_run(harness: ModuleType) -> None:
    summary = harness.summarise(
        [_run(1), _run(2, leftovers=["application_7.pdf", "application_7.docx"])]
    )

    assert summary["runs_leaking_temp_files"] == 1


def test_spread_reports_the_worst_case_not_only_the_average(harness: ModuleType) -> None:
    summary = harness.summarise([_run(1, steps=10), _run(2, steps=50), _run(3, steps=12)])

    steps = summary["counters"]["agent_steps_total"]
    assert steps["min"] == 10.0
    assert steps["max"] == 50.0
    assert steps["median"] == 12.0
