"""Artifact garbage collection, tool schemas and submission validators."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from pydantic import ValidationError

from cvnator.domain.actions import (
    ACTION_RISK,
    ActionType,
    AgentDecision,
    ElementTarget,
    RiskLevel,
    ToolResult,
)
from cvnator.storage.artifacts import ArtifactKind, ArtifactStore, RetentionPolicy

# ------------------------------------------------------------- tool schemas


def test_every_action_has_a_declared_risk_level():
    missing = [a for a in ActionType if a not in ACTION_RISK]
    assert not missing, f"Actions without a risk classification: {missing}"


@pytest.mark.parametrize(
    "action,expected",
    [
        (ActionType.READ_PAGE, RiskLevel.LOW),
        (ActionType.SCROLL_PAGE, RiskLevel.LOW),
        (ActionType.FILL_FIELD, RiskLevel.MEDIUM),
        (ActionType.UPLOAD_FILE, RiskLevel.MEDIUM),
        (ActionType.CREATE_ACCOUNT, RiskLevel.HIGH),
        (ActionType.SUBMIT_APPLICATION, RiskLevel.HIGH),
        (ActionType.ACCEPT_LEGAL_TERMS, RiskLevel.HIGH),
    ],
)
def test_risk_levels_match_the_specification(action, expected):
    assert AgentDecision(action=action).risk is expected


def test_decision_rejects_an_unknown_field():
    """extra="forbid" keeps a hallucinated field from silently passing through."""
    with pytest.raises(ValidationError):
        AgentDecision(action=ActionType.READ_PAGE, nonsense="x")


def test_decision_requires_an_action():
    with pytest.raises(ValidationError):
        AgentDecision()


def test_tool_result_detects_a_url_change():
    unchanged = ToolResult(
        success=True, action=ActionType.CLICK_ELEMENT,
        url_before="https://a/1", url_after="https://a/1",
    )
    changed = ToolResult(
        success=True, action=ActionType.CLICK_ELEMENT,
        url_before="https://a/1", url_after="https://a/2",
    )
    assert not unchanged.changed_url
    assert changed.changed_url


def test_empty_target_is_recognised():
    assert ElementTarget().is_empty()
    assert not ElementTarget(element_id="e1").is_empty()


# ------------------------------------------------------- garbage collection


def _make_file(tmp_path: Path, name: str, size: int = 512) -> Path:
    path = tmp_path / name
    path.write_bytes(b"x" * size)
    return path


_JOB_SEQUENCE = itertools.count(1)


def _make_application(session) -> int:
    """Artifacts reference a real application; the FK constraint is enforced."""
    from cvnator.db.models import ApplicationRow, JobRow

    # (platform, platform_job_id) is unique, so each fixture job needs its own id.
    job = JobRow(
        platform="mock",
        platform_job_id=f"J{next(_JOB_SEQUENCE)}",
        company="X",
        role="Y",
    )
    session.add(job)
    session.flush()
    application = ApplicationRow(job_id=job.id, platform="mock", state="APPLYING")
    session.add(application)
    session.flush()
    return application.id


def test_temporary_resume_is_deleted_after_submission(db_session, tmp_path):
    store = ArtifactStore(db_session)
    application_id = _make_application(db_session)
    path = _make_file(tmp_path, "application_1.pdf")
    row = store.register(path, ArtifactKind.TEMP_RESUME, application_id=application_id)

    deleted, reason = store.delete(row, reason="submission confirmed")
    assert deleted, reason
    assert not path.exists()
    assert row.deleted


def test_the_master_cv_is_never_deleted(db_session, tmp_path):
    store = ArtifactStore(db_session)
    path = _make_file(tmp_path, "master_cv.pdf")
    row = store.register(path, ArtifactKind.MASTER_CV)

    deleted, reason = store.delete(row)
    assert not deleted
    assert path.exists()
    assert reason


def test_a_file_in_use_is_never_deleted(db_session, tmp_path):
    store = ArtifactStore(db_session)
    path = _make_file(tmp_path, "uploading.pdf")
    row = store.register(path, ArtifactKind.TEMP_RESUME, in_use=True)

    assert not store.delete(row)[0]
    assert path.exists()

    store.mark_in_use(path, False)
    assert store.delete(row)[0]
    assert not path.exists()


def test_files_marked_permanent_survive_collection(db_session, tmp_path):
    store = ArtifactStore(db_session)
    keep = _make_file(tmp_path, "keep.pdf")
    drop = _make_file(tmp_path, "drop.pdf")
    store.register(keep, ArtifactKind.TEMP_RESUME, permanent=True)
    store.register(drop, ArtifactKind.TEMP_RESUME)

    result = store.collect(RetentionPolicy.KEEP_LAST_N, keep_last_n=0)
    assert keep.exists()
    assert not drop.exists()
    assert result.skipped_protected >= 1


def test_keep_everything_deletes_nothing(db_session, tmp_path):
    store = ArtifactStore(db_session)
    path = _make_file(tmp_path, "temp.pdf")
    store.register(path, ArtifactKind.TEMP_RESUME)

    assert store.collect(RetentionPolicy.KEEP_EVERYTHING).deleted == 0
    assert path.exists()


def test_screenshots_survive_when_debug_retention_is_on(db_session, tmp_path):
    store = ArtifactStore(db_session)
    shot = _make_file(tmp_path, "shot.png")
    store.register(shot, ArtifactKind.SCREENSHOT)

    store.collect(RetentionPolicy.KEEP_LAST_N, keep_last_n=0, keep_screenshots=True)
    assert shot.exists()

    store.collect(RetentionPolicy.KEEP_LAST_N, keep_last_n=0, keep_screenshots=False)
    assert not shot.exists()


def test_delete_for_application_cleans_only_that_application(db_session, tmp_path):
    store = ArtifactStore(db_session)
    first = _make_application(db_session)
    second = _make_application(db_session)
    mine = _make_file(tmp_path, "mine.pdf")
    theirs = _make_file(tmp_path, "theirs.pdf")
    store.register(mine, ArtifactKind.TEMP_RESUME, application_id=first)
    store.register(theirs, ArtifactKind.TEMP_RESUME, application_id=second)

    result = store.delete_for_application(first)
    assert result.deleted == 1
    assert not mine.exists()
    assert theirs.exists()


def test_usage_report_groups_by_kind(db_session, tmp_path):
    store = ArtifactStore(db_session)
    store.register(_make_file(tmp_path, "a.pdf", 100), ArtifactKind.TEMP_RESUME)
    store.register(_make_file(tmp_path, "b.pdf", 200), ArtifactKind.TEMP_RESUME)
    store.register(_make_file(tmp_path, "c.png", 50), ArtifactKind.SCREENSHOT)

    report = store.usage_report()
    assert report["temp_resume"] == {"count": 2, "bytes": 300}
    assert report["screenshot"]["count"] == 1


def test_sweep_marks_files_that_vanished(db_session, tmp_path):
    store = ArtifactStore(db_session)
    path = _make_file(tmp_path, "gone.pdf")
    store.register(path, ArtifactKind.TEMP_RESUME)
    path.unlink()

    assert store.sweep_missing() == 1
    assert store.usage_report() == {}
