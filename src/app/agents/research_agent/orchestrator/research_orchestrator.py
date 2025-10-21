
import logging
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from ..models.types import ResearchAgentState, AllocationStrategy, StockRecommendation
from ..config.settings import LLM_CONFIG
from ..agents.stock_fetcher_agent import StockFetcherAgent
from ..agents.stock_filter_agent import StockFilterAgent
from ..agents.price_fetcher_agent import PriceFetcherAgent
from ..agents.recommendation_agent import RecommendationAgent

logger = logging.getLogger(__name__)

class ResearchOrchestrator:    
    def __init__(self, api_key=None):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            google_api_key=api_key or LLM_CONFIG["api_key"]
        )        
        self.agents = {
            "stock_fetcher": StockFetcherAgent(self.llm),
            "stock_filter": StockFilterAgent(self.llm),
            "price_fetcher": PriceFetcherAgent(self.llm),
            "recommendation": RecommendationAgent(self.llm)
        }
        
        logger.info("Research Orchestrator initialized with 4 agents")
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(ResearchAgentState)
        
        workflow.add_node("stock_fetcher", self._run_stock_fetcher)
        workflow.add_node("stock_filter", self._run_stock_filter)
        workflow.add_node("price_fetcher", self._run_price_fetcher)
        workflow.add_node("recommendation", self._run_recommendation)        
        workflow.set_entry_point("stock_fetcher")
        workflow.add_edge("stock_fetcher", "stock_filter")
        workflow.add_edge("stock_filter", "price_fetcher")
        workflow.add_edge("price_fetcher", "recommendation")
        workflow.add_edge("recommendation", END)
        
        return workflow.compile()
    
    async def _run_stock_fetcher(self, state: ResearchAgentState):
        return await self.agents["stock_fetcher"].execute(state)
    
    async def _run_stock_filter(self, state: ResearchAgentState):
        return await self.agents["stock_filter"].execute(state)
    
    async def _run_price_fetcher(self, state: ResearchAgentState):
        return await self.agents["price_fetcher"].execute(state)
    
    async def _run_recommendation(self, state: ResearchAgentState):
        return await self.agents["recommendation"].execute(state)
    
    async def generate_recommendations(
        self,
        allocation_strategy: AllocationStrategy,
        investment_amount: float,
        preferred_sectors: list,
        risk_score: float = 0.5,
        diversification_score: float = 0.7,
        reasoning: str = ""
    ) -> list[StockRecommendation]:
        
        logger.info(f"Starting research for {allocation_strategy.value} with ₹{investment_amount}")
        
        initial_state = ResearchAgentState(
            allocation_strategy=allocation_strategy,
            investment_amount=investment_amount,
            risk_score=risk_score,
            diversification_score=diversification_score,
            reasoning=reasoning,
            preferred_sectors=preferred_sectors,
            messages=[],
            stock_lists={},
            stock_prices={},
            filtered_stocks={},
            recommendations=[]
        )
        
        workflow = self.create_workflow()
        result = await workflow.ainvoke(initial_state)
        
        logger.info(f"Research completed with {len(result['recommendations'])} recommendations")
        return result["recommendations"]

