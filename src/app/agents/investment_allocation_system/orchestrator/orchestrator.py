import logging
from langgraph.graph import StateGraph, END

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:  # pragma: no cover - fallback for smoke tests in constrained environments
    class ChatGoogleGenerativeAI:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

from ..models.types import (
    InvestmentAllocationState,
    AllocationDecision,
    RiskLevel,
    AllocationStrategy,
)
from ..config.settings import ALLOCATION_FACTORS, LLM_CONFIG
from ..agents.amount_analysis import AmountAnalysisAgent
from ..agents.risk_assessment import RiskAssessmentAgent
from ..agents.market_conditions_agent import MarketConditionsAgent
from ..agents.strategy_agent import StrategyDeterminationAgent
from ..agents.validation_agent import ValidationAgent

logger = logging.getLogger(__name__)

class InvestmentAllocationOrchestrator:
    
    def __init__(self, api_key=None):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            google_api_key=api_key or LLM_CONFIG["api_key"]
        )        
        self.agents = {
            "amount_analysis": AmountAnalysisAgent(self.llm, ALLOCATION_FACTORS),
            "risk_assessment": RiskAssessmentAgent(self.llm),
            "market_conditions": MarketConditionsAgent(self.llm),
            "strategy_determination": StrategyDeterminationAgent(self.llm),
            "validation": ValidationAgent(self.llm)
        }
        
        logger.info("Orchestrator initialized with all agents")
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(InvestmentAllocationState)        
        workflow.add_node("amount_analysis", self._run_amount_analysis)
        workflow.add_node("risk_assessment", self._run_risk_assessment)
        workflow.add_node("market_conditions", self._run_market_conditions)
        workflow.add_node("strategy_determination", self._run_strategy_determination)
        workflow.add_node("validation", self._run_validation)      

        workflow.set_entry_point("amount_analysis")
        
        workflow.add_edge("amount_analysis", "risk_assessment")
        workflow.add_edge("risk_assessment", "market_conditions")
        workflow.add_edge("market_conditions", "strategy_determination")
        workflow.add_edge("strategy_determination", "validation")
        workflow.add_edge("validation", END)
        
        return workflow.compile()
    
    async def _run_amount_analysis(self, state: InvestmentAllocationState):
        return await self.agents["amount_analysis"].execute(state)
    
    async def _run_risk_assessment(self, state: InvestmentAllocationState):
        return await self.agents["risk_assessment"].execute(state)
    
    async def _run_market_conditions(self, state: InvestmentAllocationState):
        return await self.agents["market_conditions"].execute(state)
    
    async def _run_strategy_determination(self, state: InvestmentAllocationState):
        return await self.agents["strategy_determination"].execute(state)
    
    async def _run_validation(self, state: InvestmentAllocationState):
        return await self.agents["validation"].execute(state)
    
    async def run_allocation(self, 
                           investment_amount: float,
                           risk_tolerance: RiskLevel = RiskLevel.MODERATE,
                           investment_horizon: str = "medium",
                           user_preferences: dict = None) -> AllocationDecision:
        
        logger.info(f"Starting allocation for ₹{investment_amount}")
        
        initial_state = InvestmentAllocationState(
            investment_amount=investment_amount,
            user_risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon,
            user_preferences=user_preferences or {},
            market_conditions={},
            allocation_factors={},
            volatility_threshold=0.25,
            diversification_requirement=False,
            sector_constraints=[],
            allocation_strategy=AllocationStrategy.MULTI_STOCK,
            allocation_decision=None,
            messages=[],
            next_agent=""
        )
        
        workflow = self.create_workflow()
        result = await workflow.ainvoke(initial_state)
        
        logger.info("Allocation completed")
        return result["allocation_decision"]
