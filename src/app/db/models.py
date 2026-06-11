from __future__ import annotations

import uuid
from datetime import datetime

try:
    from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text
    from sqlalchemy.orm import relationship

    from .base import Base

    def _uuid() -> str:
        return str(uuid.uuid4())


    class AnalysisRun(Base):
        __tablename__ = "analysis_runs"

        id = Column(String, primary_key=True, default=_uuid)
        query = Column(Text, nullable=True)
        parameters = Column(JSON, nullable=False, default=dict)
        status = Column(String, nullable=False, default="completed")
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

        result = relationship("AnalysisResult", back_populates="run", uselist=False, cascade="all, delete-orphan")


    class AnalysisResult(Base):
        __tablename__ = "analysis_results"

        run_id = Column(String, ForeignKey("analysis_runs.id"), primary_key=True)
        recommendations = Column(JSON, nullable=False, default=list)
        full_result = Column(JSON, nullable=False, default=dict)

        run = relationship("AnalysisRun", back_populates="result")
except ModuleNotFoundError:  # pragma: no cover - sandbox fallback
    Base = object  # type: ignore[assignment]

    class AnalysisRun:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.id = str(uuid.uuid4())
            self.query = kwargs.get("query")
            self.parameters = kwargs.get("parameters", {})
            self.status = kwargs.get("status", "completed")
            self.created_at = datetime.utcnow()
            self.result = None


    class AnalysisResult:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.run_id = kwargs.get("run_id")
            self.recommendations = kwargs.get("recommendations", [])
            self.full_result = kwargs.get("full_result", {})
