from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

_MEMORY_HISTORY: List[Dict[str, Any]] = []

try:
    from sqlalchemy import select

    from .models import AnalysisRun, AnalysisResult
    from .session import get_db, engine
    from .base import Base
    _SQLALCHEMY_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - sandbox fallback
    select = None  # type: ignore[assignment]
    AnalysisRun = AnalysisResult = None  # type: ignore[assignment]
    get_db = None  # type: ignore[assignment]
    engine = None  # type: ignore[assignment]
    Base = None  # type: ignore[assignment]
    _SQLALCHEMY_AVAILABLE = False


def init_db() -> None:
    if not _SQLALCHEMY_AVAILABLE or Base is None or engine is None:
        return
    Base.metadata.create_all(bind=engine)


def save_analysis(query: Optional[str], parameters: Dict[str, Any], result: Dict[str, Any], status: str = "completed") -> str:
    if not _SQLALCHEMY_AVAILABLE or AnalysisRun is None or AnalysisResult is None or get_db is None:
        record = {
            "id": f"memory-{len(_MEMORY_HISTORY) + 1}",
            "query": query,
            "parameters": parameters,
            "status": status,
            "created_at": None,
            "result": result,
        }
        _MEMORY_HISTORY.insert(0, record)
        return record["id"]

    init_db()
    safe_result = json.loads(json.dumps(result, default=str))
    safe_parameters = json.loads(json.dumps(parameters, default=str))
    with get_db() as db:
        run = AnalysisRun(query=query, parameters=safe_parameters, status=status)
        db.add(run)
        db.flush()
        recommendations = safe_result.get("recommendations", [])
        db.add(
            AnalysisResult(
                run_id=run.id,
                recommendations=recommendations,
                full_result=safe_result,
            )
        )
        db.commit()
        return run.id


def list_analysis_history(limit: int = 50) -> List[Dict[str, Any]]:
    if not _SQLALCHEMY_AVAILABLE or get_db is None:
        return _MEMORY_HISTORY[:limit]

    init_db()
    with get_db() as db:
        stmt = select(AnalysisRun).order_by(AnalysisRun.created_at.desc()).limit(limit)
        rows = db.execute(stmt).scalars().all()
        history: List[Dict[str, Any]] = []
        for row in rows:
            history.append(
                {
                    "id": row.id,
                    "query": row.query,
                    "parameters": row.parameters,
                    "status": row.status,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "result": row.result.full_result if row.result else None,
                }
            )
        return history
