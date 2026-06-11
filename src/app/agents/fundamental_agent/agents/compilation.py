from typing import Dict
import json
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.types import FundamentalAnalysis,FundamentalAgentState
from ..config.settings import logger, GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SECTOR_ETFS

class FundamentalCompilationAgent:
    async def execute(self, state: FundamentalAgentState) -> FundamentalAgentState:
        logger.info("Compiling fundamental analysis")
        stock_metrics = state.get("financial_data", {})
        
        fundamental_analysis = FundamentalAnalysis(
            company_financials_summary=state.get("company_financials_analysis", "Analysis unavailable"),
            sector_strength_summary=state.get("sector_strength_analysis", "Analysis unavailable"),
            fundamental_investment_thesis=state.get("investment_thesis_analysis", "Analysis unavailable"),
            stock_metrics=stock_metrics
        )
        
        state["fundamental_analysis"] = fundamental_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive fundamental analysis"))
        
        return state
