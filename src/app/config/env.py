from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)
