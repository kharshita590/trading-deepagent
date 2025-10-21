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

class VolatilityAnalysisAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )
    
    async def execute(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Analyzing volatility metrics")
        
        vol_liq_data = state.get("vol_liq_data", {})
        
        volatility_prompt = f"""
        Analyze the volatility characteristics of this portfolio:
        
        {self._format_volatility_data(vol_liq_data)}
        
        Provide a comprehensive volatility assessment summary (4-5 sentences) covering:
        1. Overall portfolio volatility profile and classification
        2. Individual stock volatility analysis and risk levels
        3. Historical vs current volatility trends
        4. Volatility clustering and correlation patterns
        5. Market regime analysis (low vol vs high vol environment)
        6. Volatility-adjusted position sizing recommendations
        
        Focus on:
        - Historical volatility levels (1Y, 3M, 1M comparison)
        - ATR analysis for intraday risk assessment
        - Beta analysis for market sensitivity
        - Volatility percentiles and extremes
        - Recent volatility trend changes
        - Portfolio volatility diversification benefits
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=volatility_prompt)])
            state["volatility_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            state["volatility_analysis"] = f"Volatility analysis unavailable: {str(e)}"
        
        return state
    
    def _format_volatility_data(self, vol_liq_data: Dict) -> str:
        formatted = []
        for ticker, data in vol_liq_data.items():
            if "error" in data:
                continue
                
            vol_metrics = data.get("volatility_metrics", {})
            rec = data.get("recommendation", {})
            
            formatted.append(f"""
STOCK: {rec.get('company_name', 'N/A')} ({ticker})
- Allocation: {rec.get('allocation_percentage', 0):.1f}% (₹{rec.get('allocation_amount', 0):,.0f})
- Historical Volatility: 1Y: {vol_metrics.get('historical_vol_1y', 0):.1f}% | 3M: {vol_metrics.get('historical_vol_3m', 0):.1f}% | 1M: {vol_metrics.get('historical_vol_1m', 0):.1f}%
- Current 30D Vol: {vol_metrics.get('current_30d_vol', 0):.1f}% (Percentile: {vol_metrics.get('volatility_percentile', 50):.0f}%)
- ATR: {vol_metrics.get('atr_14', 0):.2f} ({vol_metrics.get('atr_percentage', 0):.2f}% of price)
- Volatility Class: {vol_metrics.get('volatility_class', 'Unknown')} | Trend: {vol_metrics.get('volatility_trend', 'Unknown')}
- Beta: {vol_metrics.get('beta', 1.0):.2f} | 52W Range: {vol_metrics.get('price_range_52w', 0):.1f}%
- Current vs 52W High: {vol_metrics.get('current_vs_52w_high', 0):+.1f}% | vs Low: {vol_metrics.get('current_vs_52w_low', 0):+.1f}%
""")
        
        return '\n'.join(formatted)