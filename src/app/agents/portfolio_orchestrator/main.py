import logging
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from .query_processor import QueryProcessor
from .workflow_builder import WorkflowBuilder
from .workflow_nodes import WorkflowNodes

from ..investment_allocation_system.orchestrator import InvestmentAllocationOrchestrator
from ..research_agent.orchestrator import ResearchOrchestrator
from ..fundamental_agent.orchestrator import FundamentalOrchestrator
from ..macro_agent.orchestrator import MacroAnalysisOrchestrator
from ..technical_agent.orchestrator import TechnicalAnalysisOrchestrator
from ..volatality.orchestrator import VolatilityLiquidityOrchestrator
from ..behavorial_agent.orchestrator import BehavioralPsychologyOrchestrator
from ..risk_management.orchestrator import RiskManagementOrchestrator

logger = logging.getLogger(__name__)


class PortfolioOrchestrator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key:
            self.gemini_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=self.api_key
            )
            self.query_processor = QueryProcessor(self.gemini_model)
        else:
            self.gemini_model = None
            self.query_processor = None        
        self._initialize_orchestrators()        
        self._build_workflow()
        logger.info("PortfolioOrchestrator initialized successfully")
    
    def _initialize_orchestrators(self):
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
            }
        )
        
        result["conversation_history"] = query_result["conversation_history"]
        result["status"] = "complete"
        return result
    
    async def run_complete_analysis(
        self,
        investment_amount: float,
        risk_tolerance: str = "moderate",
        investment_horizon: str = "medium",
        user_preferences: Dict = None
    ) -> Dict:
        logger.info("STARTING COMPLETE PORTFOLIO ANALYSIS")
        logger.info(f"Investment Amount: ₹{investment_amount:,.2f}")
        logger.info(f"Risk Tolerance: {risk_tolerance}")
        logger.info(f"Investment Horizon: {investment_horizon}")
        
        initial_state = {
            "investment_amount": investment_amount,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon,
            "user_preferences": user_preferences or {},
            "allocation_decision": None,
            "recommendations": [],
            "fundamental_analysis": None,
            "macro_analysis": None,
            "technical_analysis": None,
            "volatility_liquidity_analysis": None,
            "behavioral_psychology_analysis": None,
            "risk_management_result": None,
            "messages": [],
            "errors": {}
        }
        
        final_state = await self.workflow.ainvoke(initial_state)
        
        logger.info("PORTFOLIO ANALYSIS COMPLETED")
        return final_state