from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("TWELVEDATA_API_KEY", "dummy")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("DATABASE_URL", "sqlite:///./trading_deepagent.db")
os.environ.setdefault("LOG_LEVEL", "INFO")
