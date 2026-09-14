"""Command line entry points."""

from __future__ import annotations

import argparse

import uvicorn

from cvnator.config import get_settings
from cvnator.observability import setup_logging

DEFAULT_PORT = 8088


def main() -> None:
    parser = argparse.ArgumentParser(prog="cvnator", description="Autonomous AI Job Application Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the API and UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--reload", action="store_true")

    mock = sub.add_parser("mockats", help="Run the local mock ATS used for testing")
    mock.add_argument("--port", type=int, default=8099)

    sub.add_parser("migrate", help="Apply database migrations")

    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.logs_dir / "cvnator.log")

    if args.command == "serve":
        uvicorn.run("cvnator.api.app:app", host=args.host, port=args.port, reload=args.reload)
    elif args.command == "mockats":
        uvicorn.run("cvnator.mockats.server:app", host="127.0.0.1", port=args.port,
                    log_level="warning")
    elif args.command == "migrate":
        from alembic import command
        from alembic.config import Config

        command.upgrade(Config("alembic.ini"), "head")
        print("Database is up to date.")


if __name__ == "__main__":
    main()
