"""ORM models.

Design notes
------------
* Rich, evolving structures (the master profile, a scored match, an ATS report) are
  stored as JSON columns. Anything queried or joined on gets a real column.
* ``ResumeVersion`` stores the *recipe* for a generated CV, not the PDF. The binary
  is a disposable artifact; the recipe is permanent, so a deleted CV can be rebuilt.
* No credential material is ever stored here. See ``cvnator.storage.credentials``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from cvnator.db.base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class CandidateProfileRow(Base):
    """Versioned snapshots of the Candidate Master Profile."""

    __tablename__ = "candidate_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), default="")
    email: Mapped[str] = mapped_column(String(200), default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    master_cv_path: Mapped[str] = mapped_column(String(1000), default="")
    master_cv_sha256: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class SearchProfileRow(Base):
    __tablename__ = "search_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    role_title: Mapped[str] = mapped_column(String(300), default="")
    minimum_match_score: Mapped[float] = mapped_column(Float, default=0.7)
    daily_application_limit: Mapped[int] = mapped_column(Integer, default=10)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class JobRow(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("platform", "platform_job_id", name="uq_job_platform_id"),
        Index("ix_jobs_identity", "identity_key"),
        Index("ix_jobs_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    platform: Mapped[str] = mapped_column(String(60), default="unknown")
    platform_job_id: Mapped[str] = mapped_column(String(200), default="")
    external_job_id: Mapped[str] = mapped_column(String(200), default="")
    company: Mapped[str] = mapped_column(String(300), default="")
    role: Mapped[str] = mapped_column(String(300), default="")
    location: Mapped[str] = mapped_column(String(300), default="")
    url: Mapped[str] = mapped_column(String(1200), default="")
    apply_url: Mapped[str] = mapped_column(String(1200), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    description_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    identity_key: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(40), default="DISCOVERED")
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ats_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    search_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("search_profiles.id", ondelete="SET NULL"), nullable=True
    )
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    match_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    date_discovered: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    date_analyzed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    date_applied: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    applications: Mapped[list[ApplicationRow]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class ApplicationRow(Base):
    __tablename__ = "applications"
    __table_args__ = (Index("ix_applications_state", "state"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_sessions.id", ondelete="SET NULL"), nullable=True
    )
    platform: Mapped[str] = mapped_column(String(60), default="")
    external_platform: Mapped[str] = mapped_column(String(60), default="")
    state: Mapped[str] = mapped_column(String(40), default="DISCOVERED")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resume_version_id: Mapped[str] = mapped_column(String(64), default="")
    account_used: Mapped[str] = mapped_column(String(200), default="")
    checkpoints: Mapped[list] = mapped_column(JSON, default=list)
    answers_used: Mapped[dict] = mapped_column(JSON, default=dict)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    decisions_summary: Mapped[list] = mapped_column(JSON, default=list)
    submission_evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, default=True)

    job: Mapped[JobRow] = relationship(back_populates="applications")
    transitions: Mapped[list[StateTransitionRow]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )


class StateTransitionRow(Base):
    __tablename__ = "state_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    from_state: Mapped[str] = mapped_column(String(40), default="")
    to_state: Mapped[str] = mapped_column(String(40), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    application: Mapped[ApplicationRow] = relationship(back_populates="transitions")


class ResumeVersionRow(Base):
    """The recipe for a generated CV, kept after the file itself is deleted."""

    __tablename__ = "resume_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    resume_version_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_profile_version: Mapped[int] = mapped_column(Integer, default=1)
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    job_description_hash: Mapped[str] = mapped_column(String(64), default="")
    template_used: Mapped[str] = mapped_column(String(100), default="ats_plain")
    transformation_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    optimized_document: Mapped[dict] = mapped_column(JSON, default=dict)
    ats_score_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    ats_score_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_sha256: Mapped[str] = mapped_column(String(64), default="")
    file_path: Mapped[str] = mapped_column(String(1000), default="")
    file_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    generation_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MemoryRow(Base):
    """Persistent agent memory: known answers, platform knowledge, preferences."""

    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("category", "key", "scope", "scope_value", name="uq_memory_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(60), default="known_answer", index=True)
    key: Mapped[str] = mapped_column(String(300), default="")
    question_text: Mapped[str] = mapped_column(Text, default="")
    value: Mapped[str] = mapped_column(Text, default="")
    scope: Mapped[str] = mapped_column(String(40), default="global")
    scope_value: Mapped[str] = mapped_column(String(200), default="")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(String(40), default="user")
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class UserActionRow(Base):
    __tablename__ = "user_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(60), default="")
    platform: Mapped[str] = mapped_column(String(60), default="")
    company: Mapped[str] = mapped_column(String(300), default="")
    job_title: Mapped[str] = mapped_column(String(300), default="")
    problem: Mapped[str] = mapped_column(Text, default="")
    action_needed: Mapped[str] = mapped_column(Text, default="")
    question: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentSessionRow(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(30), default="stopped")
    autonomy: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    current_activity: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ArtifactRow(Base):
    """Tracked disposable files so the garbage collector never guesses."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(1000), unique=True)
    kind: Mapped[str] = mapped_column(String(40), default="temp_resume")
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="SET NULL"), nullable=True
    )
    permanent: Mapped[bool] = mapped_column(Boolean, default=False)
    in_use: Mapped[bool] = mapped_column(Boolean, default=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AppStateRow(Base):
    """Small key/value store for product state that is not domain data.

    Onboarding progress and feature flags live here rather than in a config file so
    they survive a reinstall of the code and travel with the user's database.
    """

    __tablename__ = "app_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AnalyticsEventRow(Base):
    """Product analytics, recorded locally and never sent anywhere.

    The point is the UX metrics in the spec (time to first search, takeover
    frequency); the payload is deliberately small and carries no free text from the
    candidate, the postings or the pages.
    """

    __tablename__ = "analytics_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), index=True)
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
