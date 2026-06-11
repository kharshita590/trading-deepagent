from __future__ import annotations

from contextlib import contextmanager

from app.config.env import getenv

DATABASE_URL = getenv("DATABASE_URL", "sqlite:///./trading_deepagent.db")
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
except ModuleNotFoundError:  # pragma: no cover - sandbox fallback
    engine = None
    SessionLocal = None


@contextmanager
def get_db():
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
