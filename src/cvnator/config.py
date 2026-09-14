"""Application configuration.

Local-first by default: everything (models, database, resumes) stays on the machine
unless the user explicitly configures a remote OpenAI-compatible endpoint.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CVNATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    data_dir: Path = Path("./data")
    db_url: str = "sqlite+pysqlite:///./data/cvnator.sqlite3"

    # --- AI providers ---
    reasoning_provider: str = "ollama"
    reasoning_model: str = "qwen3.5:9b"
    vision_provider: str = "ollama"
    vision_model: str = "gemma3:4b"
    resume_provider: str = "ollama"
    resume_model: str = "qwen3.5:9b"
    translation_provider: str = "ollama"
    translation_model: str = "qwen3.5:9b"

    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_base_url: str = ""
    openai_api_key: str = ""

    llm_timeout_s: float = 240.0
    llm_max_retries: int = 2

    # --- resource management ---
    vram_budget_mb: int = 11_000
    model_idle_unload_s: float = 180.0

    # --- browser ---
    browser_headless: bool = False
    browser_slowmo_ms: int = 0
    browser_nav_timeout_ms: int = 45_000
    browser_action_timeout_ms: int = 15_000

    # --- autonomy / safety defaults ---
    dry_run: bool = True
    auto_submit: bool = False
    debug_keep_screenshots: bool = False
    max_agent_steps_per_application: int = 60
    watchdog_loop_threshold: int = 4

    # --- pacing ---
    min_seconds_between_applications: float = 25.0
    max_seconds_between_applications: float = 75.0
    default_daily_application_limit: int = 30

    log_level: str = "INFO"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def temp_resumes_dir(self) -> Path:
        return self.artifacts_dir / "temp_resumes"

    @property
    def screenshots_dir(self) -> Path:
        return self.artifacts_dir / "screenshots"

    @property
    def master_cv_dir(self) -> Path:
        """Permanent user files. Never touched by the garbage collector."""
        return self.data_dir / "master"

    @property
    def browser_profile_dir(self) -> Path:
        return self.data_dir / "browser_profile"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.temp_resumes_dir,
            self.screenshots_dir,
            self.master_cv_dir,
            self.browser_profile_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
