import logging
from typing import Dict, List
from datetime import datetime
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from ..models.types import MacroAgentState, MacroAnalysis, MacroData
from ..config.settings import AppConfig

logger = logging.getLogger(__name__)

class InterestRateAnalysisAgent:    
    def __init__(self, llm: ChatGoogleGenerativeAI, config: AppConfig):
        self.llm = llm
        self.config = config
    
    async def analyze(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Analyzing interest rate impact")
        
        recommendations = state.get("recommendations", [])
        macro_data = state.get("macro_data", {})
        interest_data = macro_data.get("interest_rates", {})
        
        rate_prompt = self._build_prompt(interest_data, recommendations)
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=rate_prompt)])
            state["interest_rate_analysis"] = response.content
            logger.info("Interest rate analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in interest rate analysis: {e}")
            state["interest_rate_analysis"] = f"Interest rate analysis unavailable: {str(e)}"
        
        return state
    
    def _build_prompt(self, interest_data: Dict, recommendations: List[Dict]) -> str:
        return f"""
Analyze the interest rate environment impact on this stock portfolio:

CURRENT INTEREST RATE ENVIRONMENT:
- 10-Year Treasury Rate: {interest_data.get('treasury_10y_rate', 'N/A'):.2f}%
- Short-term Rate: {interest_data.get('short_term_rate', 'N/A'):.2f}%
- Rate Trend: {interest_data.get('rate_trend', 'N/A')}
- Last Updated: {interest_data.get('last_updated', 'N/A')}

PORTFOLIO COMPOSITION:
{self._format_recommendations(recommendations)}

Provide a detailed interest rate impact analysis (3-4 sentences) covering:
1. How current interest rates affect each sector in the portfolio
2. Impact of rate trends on valuation multiples
3. Interest-sensitive vs interest-resistant stocks in portfolio
4. Recommended positioning based on rate environment

Focus on sectors like:
- Financial Services (benefit from rising rates)
- Technology (sensitive to rate changes)
- Real Estate/Utilities (rate sensitive)
- Consumer sectors (spending impact)
"""
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(
                f"{i}. {rec.get('ticker', 'N/A')} - {rec.get('company_name', 'N/A')} "
                f"({rec.get('sector', 'Unknown')} | {rec.get('allocation_percentage', 0):.1f}%)"
            )
        return '\n'.join(formatted)
