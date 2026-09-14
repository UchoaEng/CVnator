"""Session orchestration - the thing the START button actually starts.

Runs the full pipeline as a long-lived asyncio task:

    discover -> analyse -> match -> optimise resume -> apply -> validate -> record

Four queues keep the stages decoupled, which is what lets the session keep analysing
while one application sits waiting for the user. Control commands (pause, resume,
stop, skip, take/return control) are honoured at well-defined points rather than by
killing the task mid-action, so the browser and the database never end up in a state
nobody can resume from.

Everything the UI shows - metrics, the current activity, the action-required queue -
is published from here as events.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import select

from cvnator.agent.loop import FormAgent
from cvnator.agent.memory import MemoryStore
from cvnator.agent.pacing import RateLimiter, detect_block
from cvnator.agent.policy import PolicyEngine
from cvnator.ai.gateway import AIGateway
from cvnator.browser.controller import BrowserController
from cvnator.browser.perception import observe
from cvnator.browser.tools import BrowserToolbox, ToolContext
from cvnator.config import Settings, get_settings
from cvnator.db.base import session_scope
from cvnator.db.models import (
    ApplicationRow,
    ArtifactRow,
    JobRow,
    ResumeVersionRow,
    StateTransitionRow,
    UserActionRow,
)
from cvnator.domain.application import (
    ApplicationState,
    InterventionKind,
    SessionMetrics,
    UserActionRequest,
    can_transition,
)
from cvnator.domain.job import JobPosting
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.domain.search import AutonomySettings, SearchProfile
from cvnator.jobs.dedupe import find_duplicate
from cvnator.jobs.matching import JobMatcher, MatchReport
from cvnator.jobs.prioritization import PrioritizedJob, prioritize, score_priority
from cvnator.observability import get_logger
from cvnator.platforms.base import registry
from cvnator.resume.ats import ATSAnalysis, ATSAnalyzer, score_resume
from cvnator.resume.optimizer import ResumeOptimizer, baseline_document
from cvnator.resume.renderer import render_docx, render_pdf
from cvnator.storage.artifacts import ArtifactKind, ArtifactStore

log = get_logger(__name__)

EventSink = Callable[[str, dict[str, Any]], None]


class SessionStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    USER_CONTROL = "user_control"
    STOPPING = "stopping"


@dataclass
class CurrentActivity:
    stage: str = "idle"
    job_title: str = ""
    company: str = ""
    platform: str = ""
    match_percent: int | None = None
    ats_percent: int | None = None
    step: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "job_title": self.job_title,
            "company": self.company,
            "platform": self.platform,
            "match_percent": self.match_percent,
            "ats_percent": self.ats_percent,
            "step": self.step,
        }


@dataclass
class SessionQueues:
    """Stage-separated queues, so one blocked application does not stall the rest."""

    discovery: asyncio.Queue[SearchProfile] = field(default_factory=asyncio.Queue)
    analysis: asyncio.Queue[tuple[JobPosting, SearchProfile]] = field(
        default_factory=asyncio.Queue
    )
    application: list[PrioritizedJob] = field(default_factory=list)
    user_action: list[UserActionRequest] = field(default_factory=list)

    def sizes(self) -> dict[str, int]:
        return {
            "discovery": self.discovery.qsize(),
            "analysis": self.analysis.qsize(),
            "application": len(self.application),
            "user_action": len([u for u in self.user_action if not u.resolved]),
        }


class AgentSession:
    """One autonomous run. Owns the browser, the queues and the metrics."""

    def __init__(
        self,
        profile: CandidateMasterProfile,
        searches: list[SearchProfile],
        autonomy: AutonomySettings,
        *,
        gateway: AIGateway,
        settings: Settings | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.profile = profile
        self.searches = searches
        self.autonomy = autonomy
        self.gateway = gateway
        self.event_sink = event_sink

        self.status = SessionStatus.STOPPED
        self.metrics = SessionMetrics()
        self.activity = CurrentActivity()
        self.queues = SessionQueues()
        self.rate_limiter = RateLimiter()
        self.rate_limiter.default.daily_limit = autonomy.daily_application_limit
        self.rate_limiter.default.min_application_gap_s = (
            self.settings.min_seconds_between_applications
        )
        self.rate_limiter.default.max_application_gap_s = (
            self.settings.max_seconds_between_applications
        )

        self.browser = BrowserController(self.settings)
        self.policy = PolicyEngine(profile)
        self.session_row_id: int | None = None

        self._task: asyncio.Task[None] | None = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stop_requested = False
        self._skip_requested = False
        self._started_at = 0.0
        self._pending_answers: dict[int, asyncio.Future[str | None]] = {}
        # Carries each job's ATS analysis and match report from the analysis stage to
        # the application stage, so neither is recomputed.
        self._analysis_cache: dict[int, tuple[ATSAnalysis, MatchReport, SearchProfile]] = {}

    # ------------------------------------------------------------------- events

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        if self.event_sink is not None:
            try:
                self.event_sink(kind, payload or {})
            except Exception as exc:  # pragma: no cover - a sink must never break the run
                log.warning("Event sink raised on %s: %s", kind, exc)

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "metrics": self.metrics.as_dict(),
            "activity": self.activity.as_dict(),
            "queues": self.queues.sizes(),
            "autonomy": self.autonomy.model_dump(),
            "browser_user_control": self.browser.user_has_control,
            "applications_today": self.rate_limiter.applications_today(),
        }

    def _set_activity(self, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(self.activity, key, value)
        self.emit("activity", self.activity.as_dict())

    # ------------------------------------------------------------------ controls

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            log.info("Session already running")
            return
        self._stop_requested = False
        self._pause_event.set()
        self._started_at = time.monotonic()
        self.metrics = SessionMetrics()
        self.status = SessionStatus.RUNNING
        self._task = asyncio.create_task(self._run(), name="cvnator-session")
        self.emit("session", self.snapshot())
        log.info("Session started with %s search profile(s)", len(self.searches))

    def pause(self) -> None:
        self._pause_event.clear()
        self.status = SessionStatus.PAUSED
        self.emit("session", self.snapshot())
        log.info("Session paused")

    def resume(self) -> None:
        self._pause_event.set()
        if self.browser.user_has_control:
            self.browser.reclaim_control()
        self.status = SessionStatus.RUNNING
        self.emit("session", self.snapshot())
        log.info("Session resumed")

    async def stop(self) -> None:
        self._stop_requested = True
        self.status = SessionStatus.STOPPING
        self._pause_event.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=30)
        await self._shutdown()

    def skip_current_job(self) -> None:
        self._skip_requested = True
        log.info("Skip requested for the current job")

    def take_control(self) -> None:
        """Hand the browser to the user. The agent stops acting but stays alive."""
        self._pause_event.clear()
        self.browser.grant_user_control()
        self.status = SessionStatus.USER_CONTROL
        self.emit("session", self.snapshot())

    def return_control(self) -> None:
        """Take the browser back. The agent must re-observe before it acts again."""
        self.browser.reclaim_control()
        self._pause_event.set()
        self.status = SessionStatus.RUNNING
        self.emit("session", self.snapshot())
        log.info("Control returned to the agent; it will re-observe the page before acting")

    async def _checkpoint(self) -> bool:
        """Honour pause/stop between steps. Returns False when the run should end."""
        await self._pause_event.wait()
        return not self._stop_requested

    # -------------------------------------------------------------- user answers

    async def ask_user(self, request: UserActionRequest) -> str | None:
        """Publish a request and wait for the UI to answer it.

        Returns ``None`` if the session is stopped or paused indefinitely, which the
        agent loop treats as "park this application in WAITING_USER".
        """
        with session_scope() as session:
            row = UserActionRow(
                kind=request.kind.value,
                platform=request.platform,
                company=request.company,
                job_title=request.job_title,
                problem=request.problem,
                action_needed=request.action_needed,
                question=request.question,
            )
            session.add(row)
            session.flush()
            request.id = row.id

        self.metrics.human_interventions += 1
        self.metrics.applications_waiting_user += 1
        self.queues.user_action.append(request)
        self.emit(
            "user_action_required",
            {
                "id": request.id,
                "kind": request.kind.value,
                "platform": request.platform,
                "company": request.company,
                "job_title": request.job_title,
                "problem": request.problem,
                "action_needed": request.action_needed,
                "question": request.question,
            },
        )
        log.needs_user(
            "[%s] %s - %s | %s",
            request.kind.value,
            request.company or request.platform,
            request.problem,
            request.action_needed,
        )

        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._pending_answers[request.id or -1] = future
        try:
            # A long wait is correct here: the user may be asleep. Stopping the
            # session resolves the future with None.
            return await future
        except asyncio.CancelledError:
            return None

    def answer_user_action(self, action_id: int, answer: str | None) -> bool:
        """Called by the API when the user responds."""
        with session_scope() as session:
            row = session.get(UserActionRow, action_id)
            if row is not None:
                row.answer = answer
                row.resolved = True
                row.resolved_at = datetime.now(UTC)
        for request in self.queues.user_action:
            if request.id == action_id:
                request.resolved = True
                request.answer = answer
        future = self._pending_answers.pop(action_id, None)
        if future is not None and not future.done():
            future.set_result(answer)
            self.metrics.applications_waiting_user = max(
                0, self.metrics.applications_waiting_user - 1
            )
            self.emit("session", self.snapshot())
            return True
        return False

    def _release_pending_answers(self) -> None:
        for future in self._pending_answers.values():
            if not future.done():
                future.set_result(None)
        self._pending_answers.clear()

    # ------------------------------------------------------------------ the run

    async def _run(self) -> None:
        try:
            await self.browser.start()
            for search in self.searches:
                if search.enabled:
                    self.queues.discovery.put_nowait(search)

            while await self._checkpoint():
                if (
                    self.queues.discovery.empty()
                    and self.queues.analysis.empty()
                    and not self.queues.application
                ):
                    log.info("All queues drained; session finished")
                    break

                if not self.queues.discovery.empty():
                    search = self.queues.discovery.get_nowait()
                    await self._discover(search)

                while not self.queues.analysis.empty():
                    if not await self._checkpoint():
                        break
                    job, search = self.queues.analysis.get_nowait()
                    await self._analyze(job, search)

                self.queues.application = prioritize(self.queues.application)
                while self.queues.application:
                    if not await self._checkpoint():
                        break
                    item = self.queues.application.pop(0)
                    await self._apply(item)

        except asyncio.CancelledError:
            log.info("Session task cancelled")
            raise
        except Exception as exc:
            log.error("Session failed: %s", exc, exc_info=True)
            self.emit("error", {"message": str(exc)})
        finally:
            self.metrics.execution_seconds = round(time.monotonic() - self._started_at, 1)
            await self._shutdown()

    async def _shutdown(self) -> None:
        self._release_pending_answers()
        with contextlib.suppress(Exception):
            await self.browser.stop()
        with contextlib.suppress(Exception):
            await self.gateway.resources.release_all()
        self.status = SessionStatus.STOPPED
        self._set_activity(stage="idle", step="")
        self.emit("session", self.snapshot())
        log.info("Session stopped. Metrics: %s", self.metrics.as_dict())

    # --------------------------------------------------------------- pipeline

    async def _discover(self, search: SearchProfile) -> None:
        self._set_activity(stage="discovering", step=f"Searching for {search.role_title}")
        for platform_name in search.platforms:
            if not await self._checkpoint():
                return
            try:
                adapter = registry.for_name(platform_name, self.browser)
            except LookupError as exc:
                log.warning("Skipping platform %s: %s", platform_name, exc)
                continue

            await self.rate_limiter.before_action(platform_name)
            try:
                found = await adapter.discover(search, limit=25)
            except Exception as exc:
                log.warning("Discovery failed on %s: %s", platform_name, exc)
                await self.rate_limiter.on_error(platform_name)
                continue

            self.rate_limiter.on_success(platform_name)
            self.metrics.jobs_discovered += len(found)
            for job in found:
                self.queues.analysis.put_nowait((job, search))
            self.emit("metrics", self.metrics.as_dict())
            log.info("Discovered %s jobs on %s", len(found), platform_name)

    async def _analyze(self, job: JobPosting, search: SearchProfile) -> None:
        self._set_activity(
            stage="analyzing", job_title=job.role, company=job.company, platform=job.platform,
            step="Reading the posting",
        )
        adapter = registry.for_name(job.platform, self.browser)
        await self.rate_limiter.before_action(job.platform)

        if not await adapter.open_job(job):
            log.warning("Could not open %s", job.url)
            await self.rate_limiter.on_error(job.platform)
            return

        # Block detection reads *visible text*, not raw HTML: a phrase like "rate
        # limit" inside a script bundle or a class name would otherwise pause the
        # whole session on a page that is working perfectly well.
        page_state = await observe(self.browser.page)
        blocked = detect_block(page_state.visible_text())
        if blocked:
            log.needs_user("Platform %s signalled a block ('%s'); pausing", job.platform, blocked)
            await self.ask_user(
                UserActionRequest(
                    kind=InterventionKind.PLATFORM_BLOCKED,
                    platform=job.platform,
                    problem=f"The platform reported: {blocked}",
                    action_needed=(
                        "Open the browser and check the account. CVnator will not attempt "
                        "to work around this."
                    ),
                )
            )
            return

        job = await adapter.enrich(job)
        self.metrics.jobs_analyzed += 1

        with session_scope() as session:
            duplicate = find_duplicate(session, job)
            if duplicate.is_duplicate:
                log.info("Skipping duplicate: %s", duplicate.reason)
                self.metrics.jobs_rejected += 1
                self._persist_job(session, job, search, status="REJECTED",
                                  match=None, reason=duplicate.reason)
                self.emit("metrics", self.metrics.as_dict())
                return

        analyzer = ATSAnalyzer(self.gateway)
        analysis = await analyzer.analyze(job)

        matcher = JobMatcher(self.gateway)
        report = await matcher.match(self.profile, job, search, analysis)

        threshold = max(search.minimum_match_score, self.autonomy.minimum_match_score)
        if report.rejected or report.overall < threshold:
            reason = report.rejected_reason or (
                f"Match {report.percent}% below the {threshold:.0%} threshold"
            )
            log.info("Rejected %s at %s: %s", job.role, job.company, reason)
            self.metrics.jobs_rejected += 1
            with session_scope() as session:
                self._persist_job(session, job, search, status="REJECTED",
                                  match=report, reason=reason)
            self.emit("metrics", self.metrics.as_dict())
            return

        self.metrics.jobs_matched += 1
        self._set_activity(match_percent=report.percent, step="Matched")
        log.info("Matched %s at %s: %s%%", job.role, job.company, report.percent)

        with session_scope() as session:
            job_id = self._persist_job(session, job, search, status="MATCHED", match=report)

        self.queues.application.append(
            score_priority(job, job_id, report.overall, search)
        )
        # Carry the analysis forward so the resume step does not re-run it.
        self._analysis_cache[job_id] = (analysis, report, search)
        self.emit("metrics", self.metrics.as_dict())

    def _persist_job(
        self,
        session: Any,
        job: JobPosting,
        search: SearchProfile,
        *,
        status: str,
        match: MatchReport | None,
        reason: str = "",
    ) -> int:
        existing = session.execute(
            select(JobRow).where(
                JobRow.platform == job.platform,
                JobRow.platform_job_id == job.platform_job_id,
            )
        ).scalar_one_or_none()

        payload = job.model_dump(mode="json")
        report = match.as_dict() if match else ({"rejected_reason": reason} if reason else None)

        if existing is not None:
            existing.status = status
            existing.match_score = match.overall if match else existing.match_score
            existing.match_report = report or existing.match_report
            existing.date_analyzed = datetime.now(UTC)
            existing.payload = payload
            session.flush()
            return existing.id

        row = JobRow(
            platform=job.platform,
            platform_job_id=job.platform_job_id,
            external_job_id=job.external_job_id,
            company=job.company,
            role=job.role,
            location=job.location,
            url=job.url,
            apply_url=job.apply_url,
            description=job.description,
            description_hash=job.description_hash(),
            identity_key=job.identity_key(),
            status=status,
            match_score=match.overall if match else None,
            match_report=report,
            search_profile_id=search.id,
            payload=payload,
            date_analyzed=datetime.now(UTC),
        )
        session.add(row)
        session.flush()
        return row.id

    # ---------------------------------------------------------------- applying

    async def _apply(self, item: PrioritizedJob) -> None:
        job = item.job
        self._skip_requested = False
        cached = self._analysis_cache.get(item.job_id)
        if cached is None:
            log.warning("No cached analysis for job %s; skipping", item.job_id)
            return
        analysis, match_report, search = cached

        if self.rate_limiter.daily_limit_reached(job.platform):
            log.info("Daily limit reached for %s; leaving the rest queued", job.platform)
            return

        await self.rate_limiter.before_application(job.platform)
        self._set_activity(
            stage="applying", job_title=job.role, company=job.company, platform=job.platform,
            match_percent=match_report.percent, step="Preparing a tailored resume",
        )

        resume_path: Path | None = None
        resume_version_id = ""
        ats_before = ats_after = None

        if self.autonomy.auto_optimize_resume:
            prepared = await self._prepare_resume(job, analysis, item.job_id)
            if prepared is not None:
                resume_path, resume_version_id, ats_before, ats_after = prepared
                self.metrics.resumes_generated += 1
                self._set_activity(
                    ats_percent=ats_after, step="Resume ready; opening the application"
                )

        application_id = self._open_application_record(item.job_id, job, resume_version_id)
        self.metrics.applications_started += 1
        self.emit("metrics", self.metrics.as_dict())

        adapter = registry.for_name(job.platform, self.browser)
        if not await adapter.open_job(job):
            self._close_application(application_id, ApplicationState.FAILED,
                                    errors=["Could not open the posting"])
            self.metrics.applications_failed += 1
            return

        await adapter.start_application(job)
        self._transition(application_id, ApplicationState.OPENING_APPLICATION,
                         "Application opened")
        self._transition(application_id, ApplicationState.APPLYING, "Filling the form")

        toolbox = BrowserToolbox(ToolContext(self.browser, on_screenshot=self._track_screenshot))
        with session_scope() as session:
            memory = MemoryStore(session)
            agent = FormAgent(
                self.gateway, toolbox, self.policy, memory, self.profile, self.autonomy,
                ask_user=self.ask_user,
                max_steps=self.settings.max_agent_steps_per_application,
                loop_threshold=self.settings.watchdog_loop_threshold,
            )
            self._set_activity(step="Completing the application form")
            outcome = await agent.run(
                job,
                resume_path=resume_path,
                applications_today=self.rate_limiter.applications_today(job.platform),
                applications_this_session=self.metrics.applications_started,
            )

        self.metrics.agent_steps_total += len(outcome.cycles)
        self.metrics.tool_calls_total += len(outcome.cycles)
        self.metrics.tool_retries_total += outcome.tool_retries
        self.metrics.vision_fallbacks += outcome.vision_fallbacks
        self.metrics.invalid_answers_blocked += sum(
            1 for c in outcome.cycles if c.policy_outcome == "block"
        )

        self._finish_application(application_id, outcome, job, resume_path, resume_version_id)
        self.emit("metrics", self.metrics.as_dict())

    async def _prepare_resume(
        self, job: JobPosting, analysis: ATSAnalysis, job_id: int
    ) -> tuple[Path, str, int, int] | None:
        """Generate, validate and render a tailored resume. Returns None on failure."""
        import uuid

        optimizer = ResumeOptimizer(self.gateway)
        baseline = baseline_document(self.profile)
        before = score_resume(baseline.as_text(), analysis)

        try:
            document, plan, report = await optimizer.optimize(self.profile, job, analysis)
        except Exception as exc:
            log.warning("Resume optimization failed for %s: %s", job.role, exc)
            return None

        if not report.ok:
            # The optimizer produced something the fact validator rejects. Fall back
            # to the untouched master resume rather than submitting a fabrication.
            log.warning(
                "Optimized resume rejected by the fact validator (%s); using the master CV",
                report.summary(),
            )
            document = baseline
            plan_summary: dict[str, Any] = {
                "rejected": True,
                "violations": [v.as_dict() for v in report.violations],
            }
        else:
            plan_summary = {
                "headline": plan.headline,
                "summary_rewritten": bool(plan.professional_summary),
                "experiences_reordered": plan.experience_order,
                "bullets_rewritten": sum(len(e.bullets) for e in plan.experiences),
                "skills_reordered": plan.skill_order[:20],
                "reasoning": plan.reasoning[:600],
            }

        after = score_resume(document.as_text(), analysis)
        version_id = uuid.uuid4().hex[:16]
        language = analysis.language or job.language or "en"
        pdf_path = self.settings.temp_resumes_dir / f"application_{version_id}.pdf"
        docx_path = self.settings.temp_resumes_dir / f"application_{version_id}.docx"

        rendered_pdf = await asyncio.to_thread(render_pdf, document, pdf_path, language=language)
        await asyncio.to_thread(render_docx, document, docx_path, language=language)

        with session_scope() as session:
            session.add(
                ResumeVersionRow(
                    resume_version_id=version_id,
                    source_profile_version=self.profile.version,
                    job_id=job_id,
                    job_description_hash=job.description_hash(),
                    template_used=rendered_pdf.template,
                    transformation_summary=plan_summary,
                    optimized_document={
                        "headline": document.headline,
                        "summary": document.professional_summary,
                        "skills": document.skills,
                        "experiences": [
                            {"company": e.company, "title": e.job_title, "bullets": e.bullets}
                            for e in document.experiences
                        ],
                    },
                    ats_score_before=before.match_score,
                    ats_score_after=after.match_score,
                    file_sha256=rendered_pdf.sha256,
                    file_path=str(pdf_path),
                )
            )
            store = ArtifactStore(session, self.settings)
            store.register(pdf_path, ArtifactKind.TEMP_RESUME, in_use=True)
            store.register(docx_path, ArtifactKind.INTERMEDIATE)

        log.info(
            "Resume for %s at %s: ATS estimate %s%% -> %s%%",
            job.role, job.company, before.percent, after.percent,
        )
        self.emit(
            "resume_generated",
            {
                "job": job.role, "company": job.company,
                "ats_before": before.percent, "ats_after": after.percent,
                "version": version_id, "fact_check_passed": report.ok,
            },
        )
        return pdf_path, version_id, before.percent, after.percent

    async def _track_screenshot(self, path: Path) -> None:
        with session_scope() as session:
            ArtifactStore(session, self.settings).register(path, ArtifactKind.SCREENSHOT)

    # --------------------------------------------------------- record keeping

    def _open_application_record(
        self, job_id: int, job: JobPosting, resume_version_id: str
    ) -> int:
        with session_scope() as session:
            row = ApplicationRow(
                job_id=job_id,
                session_id=self.session_row_id,
                platform=job.platform,
                external_platform=job.application_platform,
                state=ApplicationState.QUEUED.value,
                resume_version_id=resume_version_id,
                dry_run=self.autonomy.dry_run,
            )
            session.add(row)
            session.flush()
            return row.id

    def _transition(self, application_id: int, target: ApplicationState, reason: str) -> None:
        with session_scope() as session:
            row = session.get(ApplicationRow, application_id)
            if row is None:
                return
            current = ApplicationState(row.state)
            if current == target:
                return
            if not can_transition(current, target):
                log.warning(
                    "Refusing illegal transition %s -> %s for application %s",
                    current.value, target.value, application_id,
                )
                return
            session.add(
                StateTransitionRow(
                    application_id=application_id,
                    from_state=current.value,
                    to_state=target.value,
                    reason=reason,
                )
            )
            row.state = target.value
        log.transition("Application %s: %s -> %s (%s)", application_id, current.value,
                       target.value, reason)
        self.emit("application_state", {"id": application_id, "state": target.value,
                                        "reason": reason})

    def _close_application(
        self, application_id: int, state: ApplicationState, *, errors: list[str] | None = None
    ) -> None:
        self._transition(application_id, state, "; ".join(errors or []) or state.value)
        with session_scope() as session:
            row = session.get(ApplicationRow, application_id)
            if row is not None:
                row.completed_at = datetime.now(UTC)
                if errors:
                    row.errors = list(row.errors or []) + errors

    def _finish_application(
        self,
        application_id: int,
        outcome: Any,
        job: JobPosting,
        resume_path: Path | None,
        resume_version_id: str,
    ) -> None:
        job_row_id: int | None = None
        with session_scope() as session:
            row = session.get(ApplicationRow, application_id)
            if row is not None:
                job_row_id = row.job_id
                row.checkpoints = [c.value for c in outcome.checkpoints]
                row.answers_used = outcome.answers_used
                row.errors = outcome.errors
                row.decisions_summary = [c.as_dict() for c in outcome.cycles[-25:]]
                row.submission_evidence = (
                    outcome.evidence.as_dict() if outcome.evidence else None
                )

        target: ApplicationState = outcome.state
        self._transition(application_id, target, "Agent loop finished")

        if target is ApplicationState.SUBMITTED:
            self.metrics.applications_completed += 1
            self.rate_limiter.record_application(job.platform)
            with session_scope() as session:
                job_row = session.get(JobRow, job_row_id) if job_row_id else None
                if job_row is not None:
                    job_row.status = "SUBMITTED"
                    job_row.date_applied = datetime.now(UTC)
            if self.autonomy.delete_temp_resumes:
                self._cleanup_resume(application_id, resume_path, resume_version_id)
            self.emit("application_submitted", {"job": job.role, "company": job.company})
        elif target is ApplicationState.WAITING_USER:
            self.emit("application_waiting", {"job": job.role, "company": job.company})
        else:
            self.metrics.applications_failed += 1
            self._release_resume_hold(resume_path)

        with session_scope() as session:
            row = session.get(ApplicationRow, application_id)
            if row is not None and target.value in {"SUBMITTED", "FAILED", "ABANDONED"}:
                row.completed_at = datetime.now(UTC)

    def _cleanup_resume(
        self, application_id: int, resume_path: Path | None, version_id: str
    ) -> None:
        """Delete the tailored CV now that the submission is confirmed.

        The ``ResumeVersion`` row survives, which is what makes this safe: the file
        can be regenerated byte-for-byte from the recipe if it is ever needed.
        """
        if resume_path is None:
            return
        with session_scope() as session:
            store = ArtifactStore(session, self.settings)
            store.mark_in_use(resume_path, False)
            artifacts = session.execute(
                select(ArtifactRow).where(
                    ArtifactRow.path.in_(
                        [str(resume_path.resolve()), str(resume_path.with_suffix(".docx").resolve())]
                    )
                )
            ).scalars()
            freed = 0
            for artifact in artifacts:
                artifact.application_id = application_id
                deleted, _ = store.delete(artifact, reason="submission confirmed")
                if deleted:
                    freed += 1
            version = session.execute(
                select(ResumeVersionRow).where(
                    ResumeVersionRow.resume_version_id == version_id
                )
            ).scalar_one_or_none()
            if version is not None:
                version.file_deleted = True
        log.info("Deleted %s temporary resume artifact(s); the recipe is retained", freed)
        self.emit("artifacts_cleaned", {"application_id": application_id, "deleted": freed})

    def _release_resume_hold(self, resume_path: Path | None) -> None:
        if resume_path is None:
            return
        with session_scope() as session:
            ArtifactStore(session, self.settings).mark_in_use(resume_path, False)
