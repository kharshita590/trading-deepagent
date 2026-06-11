from __future__ import annotations

import sys

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - sandbox fallback
    import logging

    logger = logging.getLogger("trading_deepagent")

from app.config.env import getenv


def configure_logging() -> None:
    level = getenv("LOG_LEVEL", "INFO") or "INFO"
    if hasattr(logger, "remove") and hasattr(logger, "add"):
        logger.remove()
        logger.add(
            sys.stdout,
            level=level.upper(),
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:  # pragma: no cover - fallback for environments without loguru
        import logging

        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


__all__ = ["logger", "configure_logging"]
