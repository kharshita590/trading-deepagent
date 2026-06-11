from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - sandbox fallback
    def load_dotenv(*args, **kwargs):  # type: ignore[override]
        return False

load_dotenv()


def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)
