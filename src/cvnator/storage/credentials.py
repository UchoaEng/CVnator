"""Credential storage and the secret-handling boundary.

Rules this module exists to enforce:

* A password is never written to the database, a log line, or a JSON file.
* A password is never placed in an LLM prompt. The agent decides *that* a login
  should happen and *which* account to use; it receives an opaque handle, and the
  browser layer resolves that handle to the secret at the moment of typing.
* Storage uses the operating system's credential vault (Windows Credential Manager,
  macOS Keychain, Secret Service) via ``keyring``. Where no vault is available, the
  fallback is an age-style encrypted file whose key lives outside the repository -
  and the fallback announces itself loudly rather than pretending to be equivalent.

CVnator never creates accounts or submits credentials without the user's explicit,
per-account authorization; see ``cvnator.agent.policy``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from cvnator.config import Settings, get_settings
from cvnator.observability import get_logger

log = get_logger(__name__)

SERVICE_NAME = "CVnator"
_FALLBACK_KEY_ENV = "CVNATOR_VAULT_KEY"


class CredentialError(RuntimeError):
    pass


@dataclass(slots=True)
class CredentialHandle:
    """An opaque reference to a secret. Safe to log, safe to show a model."""

    platform: str
    username: str

    @property
    def key(self) -> str:
        return f"{self.platform}::{self.username}"

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"CredentialHandle({self.key})"

    def __str__(self) -> str:
        return self.key


class CredentialVault:
    """Stores and retrieves secrets. The only component allowed to see plaintext."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._backend = self._select_backend()

    def _select_backend(self) -> str:
        try:
            import keyring
            from keyring.backends.fail import Keyring as FailKeyring

            backend = keyring.get_keyring()
            if isinstance(backend, FailKeyring):
                raise RuntimeError("no usable keyring backend")
            log.info("Credential vault using OS keyring backend: %s", type(backend).__name__)
            return "keyring"
        except Exception as exc:
            log.warning(
                "OS credential vault unavailable (%s). Falling back to an encrypted file. "
                "Set %s to a base64 Fernet key, or credentials cannot be stored.",
                exc,
                _FALLBACK_KEY_ENV,
            )
            return "encrypted_file"

    # ------------------------------------------------------------------- file backend

    @property
    def _vault_path(self) -> Path:
        return self.settings.data_dir / "credentials.vault"

    def _fernet(self):
        from cryptography.fernet import Fernet

        raw = os.environ.get(_FALLBACK_KEY_ENV, "")
        if not raw:
            raise CredentialError(
                f"{_FALLBACK_KEY_ENV} is not set, so no credential can be stored or read. "
                f"Generate one with: python -c \"from cryptography.fernet import Fernet; "
                f'print(Fernet.generate_key().decode())"'
            )
        try:
            return Fernet(raw.encode())
        except Exception as exc:
            raise CredentialError(f"{_FALLBACK_KEY_ENV} is not a valid Fernet key") from exc

    def _read_file_vault(self) -> dict[str, str]:
        if not self._vault_path.exists():
            return {}
        token = self._vault_path.read_bytes()
        try:
            return json.loads(self._fernet().decrypt(token).decode("utf-8"))
        except CredentialError:
            raise
        except Exception as exc:
            raise CredentialError(f"Credential vault could not be decrypted: {exc}") from exc

    def _write_file_vault(self, data: dict[str, str]) -> None:
        payload = self._fernet().encrypt(json.dumps(data).encode("utf-8"))
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        self._vault_path.write_bytes(payload)
        # Windows ACLs differ from POSIX modes; failing to tighten them is not
        # a reason to abort a write that already succeeded.
        with contextlib.suppress(OSError):
            self._vault_path.chmod(0o600)

    # ----------------------------------------------------------------------- public

    def store(self, platform: str, username: str, secret: str) -> CredentialHandle:
        handle = CredentialHandle(platform=platform, username=username)
        if self._backend == "keyring":
            import keyring

            keyring.set_password(SERVICE_NAME, handle.key, secret)
        else:
            data = self._read_file_vault()
            data[handle.key] = base64.b64encode(secret.encode("utf-8")).decode("ascii")
            self._write_file_vault(data)
        log.info("Stored credential for %s (secret not logged)", handle.key)
        return handle

    def resolve(self, handle: CredentialHandle) -> str:
        """Return the plaintext secret. Call this as late as possible, never before."""
        if self._backend == "keyring":
            import keyring

            secret = keyring.get_password(SERVICE_NAME, handle.key)
        else:
            data = self._read_file_vault()
            encoded = data.get(handle.key)
            secret = base64.b64decode(encoded).decode("utf-8") if encoded else None
        if secret is None:
            raise CredentialError(f"No stored credential for {handle.key}")
        return secret

    def delete(self, handle: CredentialHandle) -> None:
        if self._backend == "keyring":
            import keyring

            try:
                keyring.delete_password(SERVICE_NAME, handle.key)
            except Exception as exc:  # pragma: no cover - already absent
                log.warning("Could not delete credential %s: %s", handle.key, exc)
        else:
            data = self._read_file_vault()
            data.pop(handle.key, None)
            self._write_file_vault(data)

    def has(self, handle: CredentialHandle) -> bool:
        try:
            self.resolve(handle)
            return True
        except CredentialError:
            return False


_vault: CredentialVault | None = None


def get_vault() -> CredentialVault:
    global _vault
    if _vault is None:
        _vault = CredentialVault()
    return _vault
