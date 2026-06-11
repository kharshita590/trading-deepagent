import os
import time
from typing import Dict, List, Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:  # pragma: no cover - fallback for smoke tests in constrained environments
    class ChatGoogleGenerativeAI:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

from app.config.validate_env import validate_required_env
from app.constants import DISCLAIMER_TEXT
from app.db.repository import save_analysis
from app.core.logging import logger
from .query_processor import QueryProcessor
from .workflow_builder import WorkflowBuilder
from .workflow_nodes import WorkflowNodes


class PortfolioOrchestrator:
    def __init__(self, api_key: Optional[str] = None):
        validate_required_env()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if self.api_key:
            self.gemini_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=self.api_key
            )
            self.query_processor = QueryProcessor(self.gemini_model)
        else:
            self.gemini_model = None
            self.query_processor = None
        self.nodes = None
        self.workflow = None
        logger.info("PortfolioOrchestrator initialized successfully")
    
    def _initialize_orchestrators(self):
        from ..investment_allocation_system.orchestrator import InvestmentAllocationOrchestrator
        from ..research_agent.orchestrator import ResearchOrchestrator
        from ..fundamental_agent.orchestrator import FundamentalOrchestrator
        from ..macro_agent.orchestrator import MacroAnalysisOrchestrator
        from ..technical_agent.orchestrator import TechnicalAnalysisOrchestrator
        from ..volatality.orchestrator import VolatilityLiquidityOrchestrator
        from ..behavioral_agent.orchestrator import BehavioralPsychologyOrchestrator
        from ..risk_management.orchestrator import RiskManagementOrchestrator

        orchestrators = {
            'investment': InvestmentAllocationOrchestrator(self.api_key),
            'research': ResearchOrchestrator(self.api_key),
            'fundamental': FundamentalOrchestrator(),
            'macro': MacroAnalysisOrchestrator(),
            'technical': TechnicalAnalysisOrchestrator(),
            'volatility': VolatilityLiquidityOrchestrator(),
            'behavioral': BehavioralPsychologyOrchestrator(),
            'risk': RiskManagementOrchestrator()
        }
        
        self.nodes = WorkflowNodes(orchestrators)
    
    def _build_workflow(self):
        builder = WorkflowBuilder(self.nodes)
        self.workflow = builder.build()

    def _ensure_workflow(self):
        if self.workflow is None or self.nodes is None:
            self._initialize_orchestrators()
            self._build_workflow()
    
    async def process_user_query(
        self, 
        query: str, 
        conversation_history: List[Dict] = None
    ) -> Dict:
        if not self.query_processor:
            raise ValueError("API key required for query processing")
        
        query_state = await self.query_processor.extract_parameters(
            query, 
            conversation_history
        )
        
        if not query_state.is_complete:
            clarification = await self.query_processor.generate_clarification_message(
                query_state
            )
            return {
                "status": "incomplete",
                "message": clarification,
                "extracted_parameters": query_state.extracted_parameters,
                "missing_parameters": query_state.missing_parameters,
                "conversation_history": query_state.conversation_history
            }
        
        return {
            "status": "complete",
            "parameters": query_state.extracted_parameters,
            "conversation_history": query_state.conversation_history
        }
    
    async def run_from_query(
        self,
        query: str,
        conversation_history: List[Dict] = None
    ) -> Dict:
        logger.info(f"Processing user query: {query}")
        
        query_result = await self.process_user_query(query, conversation_history)
        
        if query_result["status"] == "incomplete":
            return query_result
        
        params = query_result["parameters"]
        
        result = await self.run_complete_analysis(
            investment_amount=params.get("investment_amount"),
            risk_tolerance=params.get("risk_tolerance", "moderate"),
            investment_horizon=params.get("investment_horizon", "medium"),
            user_preferences={
                "preferred_sectors": params.get("preferred_sectors", []),
                "exclude_sectors": params.get("exclude_sectors", [])
            },
            request_context={"query": query, "conversation_history": conversation_history or []}
        )
        
        result["conversation_history"] = query_result["conversation_history"]
        result["status"] = "complete"
        return result
    
    async def run_complete_analysis(
        self,
        investment_amount: float,
        risk_tolerance: str = "moderate",
        investment_horizon: str = "medium",
        user_preferences: Dict = None,
        request_context: Optional[Dict] = None
    ) -> Dict:
        logger.info("STARTING COMPLETE PORTFOLIO ANALYSIS")
        logger.info(f"Investment Amount: ₹{investment_amount:,.2f}")
        logger.info(f"Risk Tolerance: {risk_tolerance}")
        logger.info(f"Investment Horizon: {investment_horizon}")
        self._ensure_workflow()
        start_time = time.perf_counter()
        
        initial_state = {
            "investment_amount": investment_amount,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon,
            "user_preferences": user_preferences or {},
            "allocation_decision": None,
            "recommendations": [],
            "fundamental_analysis": None,
            "fundamental_data": None,
            "macro_analysis": None,
            "macro_data": None,
            "technical_analysis": None,
            "technical_data": None,
            "volatility_liquidity_analysis": None,
            "volatility_liquidity_data": None,
            "behavioral_psychology_analysis": None,
            "behavioral_data": None,
            "risk_management_result": None,
            "messages": [],
            "errors": {}
        }
        
        final_state = await self.workflow.ainvoke(initial_state)

        elapsed = time.perf_counter() - start_time
        errors = final_state.get("errors", {}) if isinstance(final_state, dict) else {}
        recommendations = final_state.get("recommendations", []) if isinstance(final_state, dict) else []
        successful_analyses = [
            name for name, field in (
                ("fundamental", "fundamental_analysis"),
                ("macro", "macro_analysis"),
                ("technical", "technical_analysis"),
                ("volatility", "volatility_liquidity_analysis"),
                ("behavioral", "behavioral_psychology_analysis"),
                ("risk", "risk_management_result"),
            ) if final_state.get(field)
        ]
        final_state["meta"] = {
            "duration_seconds": round(elapsed, 3),
            "successful_analyses": successful_analyses,
            "failed_analyses": list(errors.keys()) if isinstance(errors, dict) else [],
            "analysis_timings": final_state.get("analysis_timings", {}),
            "gemini_api_calls_estimate": len(successful_analyses) + (1 if self.query_processor else 0),
        }
        final_state["disclaimer"] = DISCLAIMER_TEXT
        try:
            save_analysis(
                query=(request_context or {}).get("query"),
                parameters={
                    "investment_amount": investment_amount,
                    "risk_tolerance": risk_tolerance,
                    "investment_horizon": investment_horizon,
                    "user_preferences": user_preferences or {},
                    "conversation_history": (request_context or {}).get("conversation_history", []),
                },
                result=final_state,
            )
        except Exception as exc:
            logger.warning(f"Failed to persist analysis result: {exc}")
        
        logger.info("PORTFOLIO ANALYSIS COMPLETED")
        return final_state
