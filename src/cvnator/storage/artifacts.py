"""Artifact tracking and garbage collection.

Every disposable file the system produces is registered here before it is written,
so the collector works from a ledger rather than from guesses about a directory's
contents.

Three categories of file are **never** deleted automatically, regardless of policy:

* the master CV and anything else under the user's ``master/`` directory
* any artifact explicitly flagged ``permanent``
* any artifact currently flagged ``in_use`` (an upload in flight)

The default policy deletes a tailored resume once its application is confirmed
submitted. What survives is the ``ResumeVersion`` recipe, which is enough to rebuild
the identical file later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from cvnator.config import Settings, get_settings
from cvnator.db.models import ArtifactRow
from cvnator.observability import get_logger

log = get_logger(__name__)


class ArtifactKind(str, Enum):
    TEMP_RESUME = "temp_resume"
    SCREENSHOT = "screenshot"
    DOWNLOAD = "download"
    DEBUG = "debug"
    INTERMEDIATE = "intermediate"
    MASTER_CV = "master_cv"


class RetentionPolicy(str, Enum):
    DELETE_AFTER_SUBMISSION = "delete_after_submission"
    DELETE_AFTER_24H = "delete_after_24h"
    DELETE_AFTER_7D = "delete_after_7d"
    KEEP_LAST_N = "keep_last_n"
    KEEP_EVERYTHING = "keep_everything"


@dataclass(slots=True)
class GCResult:
    deleted: int = 0
    freed_bytes: int = 0
    skipped_protected: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ registration

    def register(
        self,
        path: Path,
        kind: ArtifactKind,
        *,
        application_id: int | None = None,
        permanent: bool = False,
        in_use: bool = False,
    ) -> ArtifactRow:
        resolved = path.resolve()
        existing = self.session.execute(
            select(ArtifactRow).where(ArtifactRow.path == str(resolved))
        ).scalar_one_or_none()

        size = resolved.stat().st_size if resolved.exists() else 0
        digest = _sha256(resolved) if resolved.exists() else ""

        if existing is not None:
            existing.kind = kind.value
            existing.application_id = application_id
            existing.permanent = permanent or existing.permanent
            existing.in_use = in_use
            existing.size_bytes = size
            existing.sha256 = digest
            existing.deleted = False
            self.session.flush()
            return existing

        row = ArtifactRow(
            path=str(resolved),
            kind=kind.value,
            application_id=application_id,
            permanent=permanent or kind is ArtifactKind.MASTER_CV,
            in_use=in_use,
            size_bytes=size,
            sha256=digest,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def mark_in_use(self, path: Path, in_use: bool) -> None:
        row = self.session.execute(
            select(ArtifactRow).where(ArtifactRow.path == str(Path(path).resolve()))
        ).scalar_one_or_none()
        if row is not None:
            row.in_use = in_use
            self.session.flush()

    # -------------------------------------------------------------------- protection

    def is_protected(self, row: ArtifactRow) -> tuple[bool, str]:
        if row.permanent:
            return True, "marked permanent"
        if row.in_use:
            return True, "currently in use"
        if row.kind == ArtifactKind.MASTER_CV.value:
            return True, "master CV"
        try:
            path = Path(row.path).resolve()
            master_dir = self.settings.master_cv_dir.resolve()
            if path.is_relative_to(master_dir):
                return True, "inside the protected master directory"
        except (OSError, ValueError):  # pragma: no cover - unresolvable path
            return True, "path could not be resolved safely"
        return False, ""

    # ------------------------------------------------------------------------ delete

    def delete(self, row: ArtifactRow, *, reason: str = "") -> tuple[bool, str]:
        protected, why = self.is_protected(row)
        if protected:
            log.info("Refusing to delete %s: %s", row.path, why)
            return False, why

        path = Path(row.path)
        freed = row.size_bytes
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            log.warning("Could not delete artifact %s: %s", path, exc)
            return False, str(exc)

        row.deleted = True
        row.deleted_at = datetime.now(UTC)
        self.session.flush()
        log.info("Deleted artifact %s (%s bytes)%s", path.name, freed, f" - {reason}" if reason else "")
        return True, ""

    def delete_for_application(self, application_id: int, *, reason: str = "") -> GCResult:
        """Called after a submission is confirmed. The recipe survives; the file does not."""
        result = GCResult()
        rows = self.session.execute(
            select(ArtifactRow).where(
                ArtifactRow.application_id == application_id,
                ArtifactRow.deleted.is_(False),
            )
        ).scalars()
        for row in rows:
            deleted, why = self.delete(row, reason=reason)
            if deleted:
                result.deleted += 1
                result.freed_bytes += row.size_bytes
            elif why:
                result.skipped_protected += 1
        return result

    # ------------------------------------------------------------------ bulk policies

    def collect(
        self,
        policy: RetentionPolicy,
        *,
        keep_last_n: int = 20,
        keep_screenshots: bool | None = None,
    ) -> GCResult:
        """Apply a retention policy across every tracked artifact."""
        result = GCResult()
        if policy is RetentionPolicy.KEEP_EVERYTHING:
            return result

        keep_shots = (
            self.settings.debug_keep_screenshots if keep_screenshots is None else keep_screenshots
        )
        now = datetime.now(UTC)
        cutoffs = {
            RetentionPolicy.DELETE_AFTER_24H: now - timedelta(hours=24),
            RetentionPolicy.DELETE_AFTER_7D: now - timedelta(days=7),
        }

        rows = list(
            self.session.execute(
                select(ArtifactRow).where(ArtifactRow.deleted.is_(False)).order_by(
                    ArtifactRow.created_at.desc()
                )
            ).scalars()
        )

        keep_ids: set[int] = set()
        if policy is RetentionPolicy.KEEP_LAST_N:
            keep_ids = {row.id for row in rows[:keep_last_n]}

        for row in rows:
            if keep_shots and row.kind == ArtifactKind.SCREENSHOT.value:
                continue
            if row.id in keep_ids:
                continue
            if policy in cutoffs:
                created = row.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                if created > cutoffs[policy]:
                    continue
            elif policy is RetentionPolicy.DELETE_AFTER_SUBMISSION:
                # Handled per-application; here only sweep orphans with no owner.
                if row.application_id is not None:
                    continue

            deleted, why = self.delete(row, reason=policy.value)
            if deleted:
                result.deleted += 1
                result.freed_bytes += row.size_bytes
            elif why:
                result.skipped_protected += 1
        return result

    def sweep_missing(self) -> int:
        """Mark ledger rows whose files vanished (manual deletion, disk cleanup)."""
        count = 0
        rows = self.session.execute(
            select(ArtifactRow).where(ArtifactRow.deleted.is_(False))
        ).scalars()
        for row in rows:
            if not Path(row.path).exists():
                row.deleted = True
                row.deleted_at = datetime.now(UTC)
                count += 1
        self.session.flush()
        return count

    def usage_report(self) -> dict[str, dict[str, int]]:
        """Bytes on disk per artifact kind, for the storage monitor."""
        report: dict[str, dict[str, int]] = {}
        rows = self.session.execute(
            select(ArtifactRow).where(ArtifactRow.deleted.is_(False))
        ).scalars()
        for row in rows:
            bucket = report.setdefault(row.kind, {"count": 0, "bytes": 0})
            bucket["count"] += 1
            bucket["bytes"] += row.size_bytes
        return report
