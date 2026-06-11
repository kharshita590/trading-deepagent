import logging
from typing import Dict, List
from datetime import datetime
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from ..models.types import MacroAgentState, MacroAnalysis, MacroData
from ..config.settings import AppConfig

logger = logging.getLogger(__name__)

class CompilationAgent:    
    def __init__(self, config: AppConfig):
        self.config = config
    
    async def compile(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Compiling macro analysis")
        macro_data = state.get("macro_data", {})
        market_sentiment = macro_data.get("economic_indicators", {}).get("market_sentiment")
        
        macro_analysis = MacroAnalysis(
            economic_conditions_summary=state.get("economic_analysis", "Analysis unavailable"),
            interest_rate_impact_summary=state.get("interest_rate_analysis", "Analysis unavailable"),
            global_events_summary=state.get("global_events_analysis", "Analysis unavailable"),
            market_sentiment=market_sentiment,
            macro_data=macro_data
        )
        
        state["macro_analysis"] = macro_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive macro-economic analysis"))
        
        logger.info("Macro analysis compilation completed")
        return state
