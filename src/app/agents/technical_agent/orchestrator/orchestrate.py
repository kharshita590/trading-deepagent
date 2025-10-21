import logging
import asyncio
from typing import Dict, List
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from ..models.types import TechnicalAgentState, TechnicalAnalysis
from ..agents.base_agent import (
    DataFetcherAgent,
    IndicatorAnalysisAgent,
    PatternAnalysisAgent,
    SignalGenerationAgent,
    CompilationAgent
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class TechnicalAnalysisOrchestrator:    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Technical Analysis Orchestrator")        
        self.data_fetcher = DataFetcherAgent()
        self.indicator_analyzer = IndicatorAnalysisAgent()
        self.pattern_analyzer = PatternAnalysisAgent()
        self.signal_generator = SignalGenerationAgent()
        self.compiler = CompilationAgent()        
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        
        self.logger.info("Orchestrator initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        self.logger.info("Building workflow graph")
        
        workflow = StateGraph(TechnicalAgentState)
        
        workflow.add_node("fetch_data", self._fetch_data_node)
        workflow.add_node("analyze_indicators", self._analyze_indicators_node)
        workflow.add_node("analyze_patterns", self._analyze_patterns_node)
        workflow.add_node("generate_signals", self._generate_signals_node)
        workflow.add_node("compile_analysis", self._compile_analysis_node)
        
        workflow.add_edge("fetch_data", "analyze_indicators")
        workflow.add_edge("analyze_indicators", "analyze_patterns")
        workflow.add_edge("analyze_patterns", "generate_signals")
        workflow.add_edge("generate_signals", "compile_analysis")
        workflow.add_edge("compile_analysis", END)
        
        workflow.set_entry_point("fetch_data")
        
        self.logger.info("Workflow graph built successfully")
        return workflow
    
    async def _fetch_data_node(self, state: TechnicalAgentState) -> TechnicalAgentState:
        return await self.data_fetcher.execute(state)
    
    async def _analyze_indicators_node(self, state: TechnicalAgentState) -> TechnicalAgentState:
        return await self.indicator_analyzer.execute(state)
    
    async def _analyze_patterns_node(self, state: TechnicalAgentState) -> TechnicalAgentState:
        return await self.pattern_analyzer.execute(state)
    
    async def _generate_signals_node(self, state: TechnicalAgentState) -> TechnicalAgentState:
        return await self.signal_generator.execute(state)
    
    async def _compile_analysis_node(self, state: TechnicalAgentState) -> TechnicalAgentState:
        return await self.compiler.execute(state)
    
    async def analyze(self, recommendations: List[Dict]) -> TechnicalAnalysis:
        """
        Run the complete technical analysis workflow
        
        Args:
            recommendations: List of stock recommendations with ticker, company_name, etc.
        
        Returns:
            TechnicalAnalysis object with complete analysis
        """
        self.logger.info(f"Starting technical analysis for {len(recommendations)} stocks")
        start_time = datetime.now()
        
        initial_state: TechnicalAgentState = {
            "recommendations": recommendations,
            "technical_data": None,
            "indicators_analysis": None,
            "patterns_analysis": None,
            "signals_analysis": None,
            "technical_analysis": None,
            "messages": [AIMessage(content="Starting technical analysis workflow")]
        }
        
        try:
            final_state = await self.app.ainvoke(initial_state)            
            technical_analysis = final_state.get("technical_analysis")
            if technical_analysis:
                technical_analysis.timestamp = datetime.now().isoformat()
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Technical analysis completed in {elapsed_time:.2f} seconds")
            
            return technical_analysis
        
        except Exception as e:
            self.logger.error(f"Error during technical analysis: {e}")
            raise
    
    def analyze_sync(self, recommendations: List[Dict]) -> TechnicalAnalysis:
        """
        Synchronous wrapper for analyze method
        
        Args:
            recommendations: List of stock recommendations
        
        Returns:
            TechnicalAnalysis object
        """
        return asyncio.run(self.analyze(recommendations))
    
