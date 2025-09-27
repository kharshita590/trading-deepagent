import logging
from loguru import logger

def setup_logging():
    logger.remove()
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    return logger
