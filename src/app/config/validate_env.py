from __future__ import annotations

import os
from typing import Iterable, List


REQUIRED_ENV_VARS: tuple[str, ...] = (
    "GOOGLE_API_KEY",
    "TWELVEDATA_API_KEY",
    "LOG_LEVEL",
    "REDIS_URL",
    "DATABASE_URL",
)


def validate_required_env(required_vars: Iterable[str] = REQUIRED_ENV_VARS) -> None:
    missing: List[str] = [var for var in required_vars if not os.getenv(var)]
    if missing:
        missing_list = ", ".join(missing)
        raise EnvironmentError(
            "Missing required environment variables: "
            f"{missing_list}. Copy .env.example to .env and set these values "
            "before starting the orchestrator or API."
        )
