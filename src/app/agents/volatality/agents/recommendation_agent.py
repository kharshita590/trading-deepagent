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

class RiskRecommendationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )
    
    async def execute(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Generating risk management recommendations")
        
        vol_liq_data = state.get("vol_liq_data", {})
        
        risk_prompt = f"""
        Generate risk management recommendations for this portfolio based on volatility and liquidity analysis:
        
        {self._format_risk_data(vol_liq_data)}
        
        CONTEXT:
        Volatility Analysis: {state.get("volatility_analysis", "Not available")}
        Liquidity Analysis: {state.get("liquidity_analysis", "Not available")}
        
        Provide comprehensive risk management recommendations (4-5 sentences) covering:
        1. Position sizing adjustments based on vol/liquidity profiles
        2. Stop-loss and risk management level recommendations
        3. Portfolio rebalancing frequency suggestions
        4. Market timing and entry/exit strategy guidance
        5. Stress testing and scenario planning recommendations
        6. Emergency liquidity and exit planning
        
        Consider:
        - VaR calculations and maximum expected losses
        - Liquidity-adjusted position limits
        - Volatility-based stop losses
        - Optimal order execution strategies
        - Portfolio correlation and concentration risks
        - Market regime change preparedness
        - Cost-benefit analysis of risk vs return
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=risk_prompt)])
            state["risk_recommendations"] = response.content
        except Exception as e:
            logger.error(f"Error in risk recommendations: {e}")
            state["risk_recommendations"] = f"Risk recommendations unavailable: {str(e)}"
        
        return state
    
    def _format_risk_data(self, vol_liq_data: Dict) -> str:
        formatted = []
        total_portfolio_value = sum(data.get("recommendation", {}).get("allocation_amount", 0) for data in vol_liq_data.values() if "error" not in data)
        
        formatted.append(f"PORTFOLIO RISK SUMMARY (Total: ₹{total_portfolio_value:,.0f}):\n")
        
        for ticker, data in vol_liq_data.items():
            if "error" in data:
                continue
                
            risk_metrics = data.get("risk_metrics", {})
            rec = data.get("recommendation", {})
            
            formatted.append(f"""
{ticker}: {rec.get('company_name', 'N/A')}
- Position: ₹{rec.get('allocation_amount', 0):,.0f} ({rec.get('allocation_percentage', 0):.1f}% of portfolio)
- VaR (95%, 1-day): ₹{risk_metrics.get('var_95_1day', 0):,.0f} ({risk_metrics.get('var_95_1day_pct', 0):.2f}% of position)
- Max Expected Daily Loss: ₹{risk_metrics.get('max_expected_daily_loss', 0):,.0f} ({risk_metrics.get('max_expected_daily_loss_pct', 0):.2f}%)
- Position vs Avg Volume: {risk_metrics.get('position_vs_avg_volume_pct', 0):.1f}%
- Liquidity Risk: {risk_metrics.get('liquidity_risk', 'Unknown')} (Score: {risk_metrics.get('liquidity_risk_score', 0)}/100)
- Overall Risk Level: {risk_metrics.get('overall_risk_level', 'Unknown')} (Score: {risk_metrics.get('combined_risk_score', 0):.0f}/100)
- Recommended Position Adjustment: {risk_metrics.get('position_size_adjustment', 1.0):.0%} of current size
""")
        
        return '\n'.join(formatted)