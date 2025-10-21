
import logging
from typing import Dict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

from ..models.types import MacroAgentState, MacroAnalysis
from ..config.settings import AppConfig
from ..agents.compilation_agent import CompilationAgent
from ..agents.data_fetcher import DataFetcherAgent
from ..agents.economic_analysis import EconomicAnalysisAgent
from ..agents.global_event_analysis import GlobalEventsAnalysisAgent
from ..agents.interest_analysis import InterestRateAnalysisAgent


logger = logging.getLogger(__name__)


class MacroAnalysisOrchestrator:    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.config.validate()
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            google_api_key=self.config.llm.api_key
        )
        
        self.data_fetcher = DataFetcherAgent(self.config)
        self.economic_analyst = EconomicAnalysisAgent(self.llm, self.config)
        self.interest_rate_analyst = InterestRateAnalysisAgent(self.llm, self.config)
        self.global_events_analyst = GlobalEventsAnalysisAgent(self.llm, self.config)
        self.compiler = CompilationAgent(self.config)
        
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        logger.info("MacroAnalysisOrchestrator initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(MacroAgentState)
        
        workflow.add_node("fetch_macro_data", self._fetch_data_node)
        workflow.add_node("analyze_economic_conditions", self._analyze_economic_node)
        workflow.add_node("analyze_interest_rate_impact", self._analyze_interest_rate_node)
        workflow.add_node("analyze_global_events", self._analyze_global_events_node)
        workflow.add_node("compile_macro_analysis", self._compile_node)
        
        workflow.add_edge("fetch_macro_data", "analyze_economic_conditions")
        workflow.add_edge("analyze_economic_conditions", "analyze_interest_rate_impact")
        workflow.add_edge("analyze_interest_rate_impact", "analyze_global_events")
        workflow.add_edge("analyze_global_events", "compile_macro_analysis")
        workflow.add_edge("compile_macro_analysis", END)
        
        workflow.set_entry_point("fetch_macro_data")
        
        return workflow
    
    async def _fetch_data_node(self, state: MacroAgentState) -> MacroAgentState:
        return await self.data_fetcher.fetch_all_data(state)
    
    async def _analyze_economic_node(self, state: MacroAgentState) -> MacroAgentState:
        return await self.economic_analyst.analyze(state)
    
    async def _analyze_interest_rate_node(self, state: MacroAgentState) -> MacroAgentState:
        return await self.interest_rate_analyst.analyze(state)
    
    async def _analyze_global_events_node(self, state: MacroAgentState) -> MacroAgentState:
        return await self.global_events_analyst.analyze(state)
    
    async def _compile_node(self, state: MacroAgentState) -> MacroAgentState:
        return await self.compiler.compile(state)
    
    async def analyze(self, recommendations: List[Dict], portfolio_amount: float = 50000) -> MacroAnalysis:
        """
        Run the complete macro analysis workflow
        
        Args:
            recommendations: List of stock recommendations
            portfolio_amount: Total portfolio amount
            
        Returns:
            MacroAnalysis object with complete analysis
        """
        logger.info(f"Starting macro analysis for {len(recommendations)} recommendations")        
        initial_state: MacroAgentState = {
            "recommendations": recommendations,
            "macro_data": None,
            "economic_analysis": None,
            "interest_rate_analysis": None,
            "global_events_analysis": None,
            "macro_analysis": None,
            "messages": [AIMessage(content="Starting macro analysis workflow")],
            "portfolio_amount": portfolio_amount
        }
        
        try:
            final_state = await self.app.ainvoke(initial_state)            
            macro_analysis = final_state.get("macro_analysis")
            
            if macro_analysis:
                logger.info("Macro analysis completed successfully")
                return macro_analysis
            else:
                raise ValueError("Macro analysis not found in final state")
                
        except Exception as e:
            logger.error(f"Error during macro analysis: {e}")
            raise
async def run_macro_analysis(
    recommendations: List[Dict],
    portfolio_amount: float = 50000,
    config: AppConfig = None
) -> MacroAnalysis:
    """
    Convenience function to run macro analysis
    
    Args:
        recommendations: List of stock recommendations
        portfolio_amount: Total portfolio amount
        config: Optional configuration
        
    Returns:
        MacroAnalysis object
    """
    orchestrator = MacroAnalysisOrchestrator(config)
    return await orchestrator.analyze(recommendations, portfolio_amount)

