"""HTTP API and static UI host.

One process runs the API, the session and the browser, because the agent needs a
live browser and an event loop it can hold open for hours. The frontend is a
separate build served as static files from here, so there is one thing to start.
"""

from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from cvnator.agent.memory import MemoryCategory, MemoryStore
from cvnator.agent.orchestrator import AgentSession
from cvnator.ai.gateway import get_gateway
from cvnator.api.events import bus
from cvnator.config import get_settings
from cvnator.db import repositories as repo
from cvnator.db.base import session_scope
from cvnator.domain.profile import CandidateMasterProfile
from cvnator.domain.search import AutonomySettings, SearchProfile
from cvnator.observability import setup_logging
from cvnator.observability.logging_setup import RING_BUFFER

# Importing the adapters registers them with the registry.
from cvnator.platforms import generic as _generic  # noqa: F401
from cvnator.platforms import mock as _mock  # noqa: F401
from cvnator.resume.extract import UnsupportedDocument, extract_text, sha256_file
from cvnator.resume.parser import parse_cv_text
from cvnator.storage.artifacts import ArtifactKind, ArtifactStore, RetentionPolicy

settings = get_settings()

#: The single live session. One browser, one pipeline, one machine.
_session: AgentSession | None = None
_autonomy = AutonomySettings(dry_run=settings.dry_run, auto_submit=settings.auto_submit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level, settings.logs_dir / "cvnator.log")
    bus.bind_loop(asyncio.get_running_loop())
    yield
    if _session is not None:
        await _session.stop()
    await get_gateway().aclose()


app = FastAPI(title="CVnator", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # Local-only surface: the API binds to localhost and the dev UI runs on 5173.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerPayload(ApiModel):
    answer: str | None = None


class MemoryPayload(ApiModel):
    question: str
    answer: str
    scope: str = "global"
    scope_value: str = ""


# ------------------------------------------------------------------------ health


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ai = await get_gateway().health()
    return {
        "status": "ok",
        "dry_run": _autonomy.dry_run,
        "session": _session.status.value if _session else "stopped",
        "ai": ai,
    }


# ----------------------------------------------------------------------- profile


@app.get("/api/profile")
async def get_profile() -> dict[str, Any]:
    with session_scope() as session:
        profile = repo.load_active_profile(session)
    if profile is None:
        return {"exists": False, "profile": None}
    return {"exists": True, "profile": profile.model_dump(mode="json")}


@app.post("/api/profile/upload")
async def upload_cv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Store the master CV permanently, then parse it into a structured profile."""
    if not file.filename:
        raise HTTPException(400, "No filename supplied")

    destination = settings.master_cv_dir / file.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        text = await asyncio.to_thread(extract_text, destination)
    except UnsupportedDocument as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc

    digest = await asyncio.to_thread(sha256_file, destination)

    try:
        profile = await parse_cv_text(get_gateway(), text, source_path=str(destination))
    except Exception as exc:
        raise HTTPException(
            502,
            f"The CV was stored but could not be parsed by the local model: {exc}. "
            "Check that Ollama is running and the configured model is pulled.",
        ) from exc

    with session_scope() as session:
        repo.save_profile(
            session, profile, master_cv_path=str(destination), master_cv_sha256=digest
        )
        # The master CV is permanent; the garbage collector must never touch it.
        ArtifactStore(session, settings).register(
            destination, ArtifactKind.MASTER_CV, permanent=True
        )

    bus.publish("profile_updated", {"name": profile.personal_information.full_name})
    return {"profile": profile.model_dump(mode="json"), "master_cv_path": str(destination)}


@app.put("/api/profile")
async def update_profile(profile: CandidateMasterProfile) -> dict[str, Any]:
    """Save the profile after the user has reviewed and corrected the extraction."""
    with session_scope() as session:
        row = repo.save_profile(session, profile)
        version = row.version
    bus.publish("profile_updated", {"version": version})
    return {"ok": True, "version": version}


# ----------------------------------------------------------------------- searches


@app.get("/api/searches")
async def get_searches() -> list[dict[str, Any]]:
    with session_scope() as session:
        return [s.model_dump(mode="json") for s in repo.list_searches(session)]


@app.post("/api/searches")
async def create_search(search: SearchProfile) -> dict[str, Any]:
    with session_scope() as session:
        saved = repo.save_search(session, search)
    return saved.model_dump(mode="json")


@app.put("/api/searches/{search_id}")
async def update_search(search_id: int, search: SearchProfile) -> dict[str, Any]:
    search.id = search_id
    with session_scope() as session:
        saved = repo.save_search(session, search)
    return saved.model_dump(mode="json")


@app.delete("/api/searches/{search_id}")
async def remove_search(search_id: int) -> dict[str, bool]:
    with session_scope() as session:
        ok = repo.delete_search(session, search_id)
    if not ok:
        raise HTTPException(404, f"No search profile {search_id}")
    return {"deleted": True}


# ------------------------------------------------------------------------ session


@app.get("/api/session")
async def session_status() -> dict[str, Any]:
    if _session is None:
        return {
            "status": "stopped",
            "metrics": {},
            "activity": {},
            "queues": {},
            "autonomy": _autonomy.model_dump(),
        }
    return _session.snapshot()


@app.get("/api/settings/autonomy")
async def get_autonomy() -> dict[str, Any]:
    return _autonomy.model_dump()


@app.put("/api/settings/autonomy")
async def set_autonomy(autonomy: AutonomySettings) -> dict[str, Any]:
    global _autonomy
    _autonomy = autonomy
    if _session is not None:
        _session.autonomy = autonomy
    bus.publish("autonomy_updated", autonomy.model_dump())
    return autonomy.model_dump()


@app.post("/api/session/start")
async def start_session() -> dict[str, Any]:
    global _session
    if _session is not None and _session.status.value in {"running", "paused", "user_control"}:
        raise HTTPException(409, "A session is already running")

    with session_scope() as session:
        profile = repo.load_active_profile(session)
        searches = repo.list_searches(session, enabled_only=True)

    if profile is None:
        raise HTTPException(400, "Upload and confirm a CV before starting a session")
    if not searches:
        raise HTTPException(400, "Enable at least one job search before starting a session")

    _session = AgentSession(
        profile,
        searches,
        _autonomy,
        gateway=get_gateway(),
        settings=settings,
        event_sink=bus.publish,
    )
    with session_scope() as session:
        row = repo.open_session_row(session, _autonomy.model_dump())
        _session.session_row_id = row.id

    await _session.start()
    return _session.snapshot()


def _require_session() -> AgentSession:
    if _session is None:
        raise HTTPException(409, "No session has been started")
    return _session


@app.post("/api/session/pause")
async def pause_session() -> dict[str, Any]:
    session = _require_session()
    session.pause()
    return session.snapshot()


@app.post("/api/session/resume")
async def resume_session() -> dict[str, Any]:
    session = _require_session()
    session.resume()
    return session.snapshot()


@app.post("/api/session/stop")
async def stop_session() -> dict[str, Any]:
    session = _require_session()
    await session.stop()
    if session.session_row_id is not None:
        with session_scope() as db:
            repo.close_session_row(db, session.session_row_id, session.metrics.as_dict())
    return session.snapshot()


@app.post("/api/session/skip")
async def skip_job() -> dict[str, Any]:
    session = _require_session()
    session.skip_current_job()
    return session.snapshot()


@app.post("/api/session/take-control")
async def take_control() -> dict[str, Any]:
    session = _require_session()
    session.take_control()
    return session.snapshot()


@app.post("/api/session/return-control")
async def return_control() -> dict[str, Any]:
    session = _require_session()
    session.return_control()
    return session.snapshot()


# ------------------------------------------------------------------- applications


@app.get("/api/applications")
async def get_applications(limit: int = 100, state: str | None = None) -> list[dict[str, Any]]:
    with session_scope() as session:
        return repo.list_applications(session, limit=limit, state=state)


@app.get("/api/jobs")
async def get_jobs(limit: int = 200, status: str | None = None) -> list[dict[str, Any]]:
    with session_scope() as session:
        return repo.list_jobs(session, limit=limit, status=status)


@app.get("/api/session/interrupted")
async def get_interrupted() -> list[dict[str, Any]]:
    """Applications left mid-flight by a crash, offered for resumption on restart."""
    with session_scope() as session:
        return repo.interrupted_applications(session)


# ------------------------------------------------------------------- user actions


@app.get("/api/user-actions")
async def get_user_actions(unresolved_only: bool = True) -> list[dict[str, Any]]:
    with session_scope() as session:
        return repo.list_user_actions(session, unresolved_only=unresolved_only)


@app.post("/api/user-actions/{action_id}/answer")
async def answer_user_action(action_id: int, payload: AnswerPayload) -> dict[str, Any]:
    session = _require_session()
    delivered = session.answer_user_action(action_id, payload.answer)
    return {"delivered": delivered}


# ------------------------------------------------------------------------ memory


@app.get("/api/memory")
async def list_memory() -> list[dict[str, Any]]:
    from sqlalchemy import select

    from cvnator.db.models import MemoryRow

    with session_scope() as session:
        rows = session.execute(
            select(MemoryRow).order_by(MemoryRow.updated_at.desc()).limit(200)
        ).scalars()
        return [
            {
                "id": r.id, "category": r.category, "key": r.key,
                "question": r.question_text, "value": r.value,
                "scope": r.scope, "scope_value": r.scope_value,
                "hits": r.hit_count, "source": r.source,
            }
            for r in rows
        ]


@app.post("/api/memory")
async def add_memory(payload: MemoryPayload) -> dict[str, Any]:
    with session_scope() as session:
        row = MemoryStore(session).remember(
            MemoryCategory.KNOWN_ANSWER,
            payload.question,
            payload.answer,
            question_text=payload.question,
            scope=payload.scope,
            scope_value=payload.scope_value,
        )
        return {"id": row.id}


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: int) -> dict[str, bool]:
    with session_scope() as session:
        MemoryStore(session).forget(memory_id)
    return {"deleted": True}


# ----------------------------------------------------------------------- storage


@app.get("/api/storage")
async def storage_report() -> dict[str, Any]:
    with session_scope() as session:
        store = ArtifactStore(session, settings)
        store.sweep_missing()
        return {"usage": store.usage_report()}


@app.post("/api/storage/collect")
async def collect_storage(policy: str = RetentionPolicy.DELETE_AFTER_24H.value) -> dict[str, Any]:
    try:
        chosen = RetentionPolicy(policy)
    except ValueError as exc:
        raise HTTPException(400, f"Unknown retention policy {policy!r}") from exc
    with session_scope() as session:
        result = ArtifactStore(session, settings).collect(chosen)
    return {
        "deleted": result.deleted,
        "freed_bytes": result.freed_bytes,
        "skipped_protected": result.skipped_protected,
    }


# ------------------------------------------------------------------ logs & events


@app.get("/api/logs")
async def get_logs(limit: int = 200, levels: str | None = None) -> list[dict[str, Any]]:
    wanted = levels.split(",") if levels else None
    return RING_BUFFER.tail(limit=limit, levels=wanted)


@app.get("/api/events")
async def stream_events() -> StreamingResponse:
    return StreamingResponse(
        bus.subscribe(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/events/recent")
async def recent_events(limit: int = 50) -> list[dict[str, Any]]:
    return bus.recent(limit)


# --------------------------------------------------------------------- static UI

_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> FileResponse:
        """Serve the built SPA, letting client-side routing handle unknown paths."""
        candidate = _FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
