from typing import Dict
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from scipy import stats
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.types import VolatilityLiquidityAgentState, VolatilityLiquidityAnalysis
from ..config.settings import logger, GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE

class CompilationAgent:
    async def execute(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Compiling volatility/liquidity analysis")
        vol_liq_analysis = VolatilityLiquidityAnalysis(
            volatility_assessment_summary=state.get("volatility_analysis", "Analysis unavailable"),
            liquidity_analysis_summary=state.get("liquidity_analysis", "Analysis unavailable"),
            risk_management_recommendations=state.get("risk_recommendations", "Analysis unavailable")
        )
        
        state["volatility_liquidity_analysis"] = vol_liq_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive volatility/liquidity analysis"))
        
        return state