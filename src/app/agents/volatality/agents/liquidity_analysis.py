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

class LiquidityAnalysisAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )
    
    async def execute(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Analyzing liquidity metrics")
        
        vol_liq_data = state.get("vol_liq_data", {})
        
        liquidity_prompt = f"""
        Analyze the liquidity characteristics of this portfolio:
        
        {self._format_liquidity_data(vol_liq_data)}
        
        Provide a comprehensive liquidity analysis summary (4-5 sentences) covering:
        1. Overall portfolio liquidity profile and accessibility
        2. Individual stock liquidity analysis and trading ease
        3. Volume trends and market depth assessment
        4. Impact cost estimation for position sizes
        5. Liquidity risk during market stress scenarios
        6. Optimal execution strategies for each position
        
        Focus on:
        - Average daily volume and dollar volume analysis
        - Volume consistency and seasonal patterns
        - Market cap and float analysis
        - Bid-ask spread proxies and trading costs
        - Turnover ratios and market maker presence
        - Position size vs daily volume constraints
        - Emergency exit liquidity assessment
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=liquidity_prompt)])
            state["liquidity_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            state["liquidity_analysis"] = f"Liquidity analysis unavailable: {str(e)}"
        
        return state
    
    def _format_liquidity_data(self, vol_liq_data: Dict) -> str:
        formatted = []
        for ticker, data in vol_liq_data.items():
            if "error" in data:
                continue
                
            liq_metrics = data.get("liquidity_metrics", {})
            rec = data.get("recommendation", {})
            
            formatted.append(f"""
{ticker}: {rec.get('company_name', 'N/A')} ({rec.get('allocation_percentage', 0):.1f}% allocation)
- Average Volume: 1Y: {liq_metrics.get('avg_volume_1y', 0):,.0f} | 3M: {liq_metrics.get('avg_volume_3m', 0):,.0f} | Current: {liq_metrics.get('current_volume', 0):,.0f}
- Volume Ratios: 3M: {liq_metrics.get('volume_ratio_3m', 1):.2f}x | 1M: {liq_metrics.get('volume_ratio_1m', 1):.2f}x | Trend: {liq_metrics.get('volume_trend', 'Unknown')}
- Dollar Volume: 1Y Avg: ₹{liq_metrics.get('avg_dollar_volume_1y', 0):,.0f} | Current: ₹{liq_metrics.get('current_dollar_volume', 0):,.0f}
- Market Cap: ₹{liq_metrics.get('market_cap', 0):,.0f} | Float: {liq_metrics.get('float_percentage', 100):.1f}%
- Daily Turnover: Current: {liq_metrics.get('daily_turnover', 0):.3f}% | Average: {liq_metrics.get('avg_daily_turnover', 0):.3f}%
- Liquidity Class: {liq_metrics.get('liquidity_class', 'Unknown')} | Avg Spread: {liq_metrics.get('avg_daily_spread', 0):.2f}%
- Price vs VWAP (1M): {liq_metrics.get('price_vs_vwap_1m', 0):+.2f}% | Impact Cost Est: {liq_metrics.get('estimated_impact_cost', 0):.2f}%
""")
        
        return '\n'.join(formatted)