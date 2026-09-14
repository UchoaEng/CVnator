"""Persistent memory scoping and duplicate detection."""

from __future__ import annotations

from cvnator.agent.memory import MemoryCategory, MemoryScope, MemoryStore, normalize_question
from cvnator.db.models import ApplicationRow, JobRow
from cvnator.domain.application import ApplicationState
from cvnator.jobs.dedupe import find_duplicate, normalize_url

# ---------------------------------------------------------------------- memory


def test_question_normalisation_collapses_noise():
    assert normalize_question("Desired salary? *") == normalize_question("desired salary")
    assert normalize_question("Phone (required)") == "phone"


def test_recall_returns_a_stored_answer(db_session):
    store = MemoryStore(db_session)
    store.remember_answer("What is your desired salary?", "R$ 11.000")
    recalled = store.recall_answer("What is your desired salary?", MemoryScope())
    assert recalled is not None
    assert recalled.value == "R$ 11.000"
    assert recalled.matched_by == "exact"


def test_recall_matches_a_reworded_question(db_session):
    store = MemoryStore(db_session)
    store.remember_answer("What is your desired salary?", "R$ 11.000")
    recalled = store.recall_answer("What is your desired salary? *", MemoryScope())
    assert recalled is not None
    assert recalled.value == "R$ 11.000"


def test_more_specific_scope_wins(db_session):
    """A company-specific answer overrides the global one without replacing it."""
    store = MemoryStore(db_session)
    store.remember_answer("Desired salary", "R$ 10.000", scope="global")
    store.remember_answer("Desired salary", "R$ 14.000", scope="company", scope_value="acme")

    at_acme = store.recall_answer("Desired salary", MemoryScope(company="Acme"))
    elsewhere = store.recall_answer("Desired salary", MemoryScope(company="Other Co"))

    assert at_acme is not None and at_acme.value == "R$ 14.000"
    assert elsewhere is not None and elsewhere.value == "R$ 10.000"


def test_out_of_scope_answers_are_not_returned(db_session):
    store = MemoryStore(db_session)
    store.remember_answer("Notice period", "30 dias", scope="company", scope_value="acme")
    assert store.recall_answer("Notice period", MemoryScope(company="Other")) is None


def test_unknown_question_returns_none(db_session):
    assert MemoryStore(db_session).recall_answer("Shoe size?", MemoryScope()) is None


def test_remembering_the_same_key_updates_rather_than_duplicates(db_session):
    store = MemoryStore(db_session)
    first = store.remember_answer("Desired salary", "R$ 10.000")
    second = store.remember_answer("Desired salary", "R$ 12.000")
    assert first.id == second.id
    assert second.value == "R$ 12.000"


def test_recall_increments_the_hit_counter(db_session):
    store = MemoryStore(db_session)
    row = store.remember_answer("Desired salary", "R$ 10.000")
    store.recall_answer("Desired salary", MemoryScope())
    store.recall_answer("Desired salary", MemoryScope())
    assert row.hit_count == 2


def test_platform_knowledge_is_scoped_to_its_platform(db_session):
    store = MemoryStore(db_session)
    store.platform_note("gupy", "login_style", "email + password")
    assert store.recall(MemoryCategory.PLATFORM_KNOWLEDGE, "gupy:login_style")


# -------------------------------------------------------------------- dedupe


def test_url_normalisation_strips_tracking_parameters():
    assert normalize_url("https://WWW.Example.com/jobs/1/?utm_source=x&id=7") == (
        "https://example.com/jobs/1?id=7"
    )


def _store_applied_job(session, job, state=ApplicationState.SUBMITTED, dry_run=False):
    row = JobRow(
        platform=job.platform,
        platform_job_id=job.platform_job_id,
        company=job.company,
        role=job.role,
        location=job.location,
        url=job.url,
        apply_url=job.apply_url,
        description=job.description,
        description_hash=job.description_hash(),
        identity_key=job.identity_key(),
        status="SUBMITTED",
    )
    session.add(row)
    session.flush()
    session.add(
        ApplicationRow(job_id=row.id, platform=job.platform, state=state.value, dry_run=dry_run)
    )
    session.flush()
    return row


def test_a_job_never_applied_to_is_not_a_duplicate(db_session, job):
    assert not find_duplicate(db_session, job).is_duplicate


def test_same_platform_listing_already_applied_is_a_duplicate(db_session, job):
    _store_applied_job(db_session, job)
    verdict = find_duplicate(db_session, job)
    assert verdict.is_duplicate
    assert "platform listing" in verdict.reason


def test_a_dry_run_application_does_not_block_a_real_one(db_session, job):
    """A dry run did not actually apply, so the job is still open."""
    _store_applied_job(db_session, job, dry_run=True)
    assert not find_duplicate(db_session, job).is_duplicate


def test_an_abandoned_application_does_not_block_a_retry(db_session, job):
    _store_applied_job(db_session, job, state=ApplicationState.ABANDONED)
    assert not find_duplicate(db_session, job).is_duplicate


def test_repost_on_another_platform_is_detected_by_identity(db_session, job):
    _store_applied_job(db_session, job)
    repost = job.model_copy(deep=True)
    repost.platform = "linkedin"
    repost.platform_job_id = "LI-999"
    repost.url = "https://linkedin.com/jobs/999"
    repost.apply_url = repost.url
    repost.description = job.description + " (reposted)"

    verdict = find_duplicate(db_session, repost)
    assert verdict.is_duplicate
    assert verdict.is_repost
