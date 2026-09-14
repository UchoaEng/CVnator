"""Run the mock ATS: ``python -m cvnator.mockats``."""

from __future__ import annotations

import uvicorn

MOCK_ATS_PORT = 8099


def main() -> None:
    uvicorn.run(
        "cvnator.mockats.server:app",
        host="127.0.0.1",
        port=MOCK_ATS_PORT,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
