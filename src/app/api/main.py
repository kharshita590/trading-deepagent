from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.config.validate_env import validate_required_env
from app.core.cache import get_redis_client
from app.constants import DISCLAIMER_TEXT
from app.core.logging import configure_logging
from app.db.repository import init_db, list_analysis_history

from .job_store import job_store

API_KEY_SECRET = os.getenv("API_KEY_SECRET", "")

app = FastAPI(title="trading-deepagent", version="0.1.0")
_orchestrator: Optional[object] = None


class AnalyzeRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if API_KEY_SECRET and x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from app.agents.portfolio_orchestrator.main import PortfolioOrchestrator

        _orchestrator = PortfolioOrchestrator(api_key=os.getenv("GOOGLE_API_KEY", ""))
    return _orchestrator


async def _run_job(job_id: str, query: str, conversation_history: List[Dict[str, Any]]) -> None:
    await job_store.update(job_id, {"status": "running"})
    try:
        result = await get_orchestrator().run_from_query(query, conversation_history)
        await job_store.update(job_id, {"status": "completed", "result": result, "disclaimer": DISCLAIMER_TEXT})
    except Exception as exc:
        await job_store.update(job_id, {"status": "failed", "error": str(exc), "disclaimer": DISCLAIMER_TEXT})


@app.on_event("startup")
async def startup_event() -> None:
    configure_logging()
    init_db()


@app.post("/portfolio/analyze")
async def analyze_portfolio(request: AnalyzeRequest, background_tasks: BackgroundTasks, _: None = Depends(require_api_key)):
    validate_required_env()
    query_result = await get_orchestrator().process_user_query(request.query, request.conversation_history)
    if query_result["status"] == "incomplete":
        return {
            "status": "needs_clarification",
            "message": query_result["message"],
            "conversation_history": query_result["conversation_history"],
            "disclaimer": DISCLAIMER_TEXT,
        }

    job_id = await job_store.create(
        {
            "query": request.query,
            "conversation_history": request.conversation_history,
            "status": "queued",
            "disclaimer": DISCLAIMER_TEXT,
        }
    )
    background_tasks.add_task(_run_job, job_id, request.query, request.conversation_history)
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Portfolio analysis started",
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.get("/portfolio/analyze/{job_id}")
async def get_analysis(job_id: str, _: None = Depends(require_api_key)):
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    job["disclaimer"] = DISCLAIMER_TEXT
    return job


@app.get("/portfolio/history")
async def history(_: None = Depends(require_api_key)):
    return {"history": list_analysis_history(), "disclaimer": DISCLAIMER_TEXT}


@app.get("/health")
async def health():
    try:
        validate_required_env()
        client = await get_redis_client()
        redis_ok = False
        if client is not None:
            redis_ok = bool(await client.ping())
        return {"status": "ok", "redis": redis_ok, "disclaimer": DISCLAIMER_TEXT}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
