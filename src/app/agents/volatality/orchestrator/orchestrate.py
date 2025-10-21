import asyncio
from langgraph.graph import StateGraph, END
from ..models.types import VolatilityLiquidityAgentState
from ..agents.compilation_agent import CompilationAgent
from ..agents.recommendation_agent import RiskRecommendationAgent
from ..agents.liquidity_analysis import LiquidityAnalysisAgent
from ..agents.volatality_analysis import VolatilityAnalysisAgent
from ..agents.data_fetch_agent import DataFetchAgent,VolatilityLiquidityDataProvider
# from ..agents.data_provider import VolatilityLiquidityDataProvider
from ..config.settings import logger

class VolatilityLiquidityOrchestrator:
    def __init__(self):
        self.data_fetch_agent = DataFetchAgent()
        self.volatility_agent = VolatilityAnalysisAgent()
        self.liquidity_agent = LiquidityAnalysisAgent()
        self.risk_agent = RiskRecommendationAgent()
        self.compilation_agent = CompilationAgent()
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(VolatilityLiquidityAgentState)
        
        workflow.add_node("fetch_volatility_liquidity_data", self.data_fetch_agent.execute)
        workflow.add_node("analyze_volatility_metrics", self.volatility_agent.execute)
        workflow.add_node("analyze_liquidity_metrics", self.liquidity_agent.execute)
        workflow.add_node("generate_risk_recommendations", self.risk_agent.execute)
        workflow.add_node("compile_volatility_liquidity_analysis", self.compilation_agent.execute)
        
        workflow.add_edge("fetch_volatility_liquidity_data", "analyze_volatility_metrics")
        workflow.add_edge("analyze_volatility_metrics", "analyze_liquidity_metrics")
        workflow.add_edge("analyze_liquidity_metrics", "generate_risk_recommendations")
        workflow.add_edge("generate_risk_recommendations", "compile_volatility_liquidity_analysis")
        workflow.add_edge("compile_volatility_liquidity_analysis", END)
        
        workflow.set_entry_point("fetch_volatility_liquidity_data")
        
        return workflow
    
    async def run(self, recommendations):
        logger.info("Starting volatility/liquidity analysis workflow")
        
        initial_state = {
            "recommendations": recommendations,
            "volatility_liquidity_analysis": None,
            "messages": []
        }
        
        workflow = self.create_workflow()
        app = workflow.compile()
        
        final_state = await app.ainvoke(initial_state)
        
        return final_state

