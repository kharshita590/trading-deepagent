from __future__ import annotations

import asyncio
import pickle
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.core.cache import get_redis_client


class JobStore:
    def __init__(self):
        self._memory: dict[str, Dict[str, Any]] = {}

    async def create(self, payload: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            **payload,
        }
        self._memory[job_id] = job
        await self._persist(job_id, job)
        return job_id

    async def update(self, job_id: str, updates: Dict[str, Any]) -> None:
        current = await self.get(job_id) or {"job_id": job_id}
        current.update(updates)
        current["updated_at"] = datetime.utcnow().isoformat()
        self._memory[job_id] = current
        await self._persist(job_id, current)

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self._memory:
            return self._memory[job_id]
        client = await get_redis_client()
        if client is not None:
            cached = await client.get(f"job:{job_id}")
            if cached is not None:
                job = pickle.loads(cached)
                self._memory[job_id] = job
                return job
        return None

    async def _persist(self, job_id: str, job: Dict[str, Any]) -> None:
        client = await get_redis_client()
        if client is not None:
            await client.set(f"job:{job_id}", pickle.dumps(job), ex=24 * 3600)


job_store = JobStore()
