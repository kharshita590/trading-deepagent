import logging
from typing import Dict, List
from datetime import datetime
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from ..models.types import MacroAgentState, MacroAnalysis, MacroData
from ..config.settings import AppConfig

logger = logging.getLogger(__name__)

class EconomicAnalysisAgent:    
    def __init__(self, llm: ChatGoogleGenerativeAI, config: AppConfig):
        self.llm = llm
        self.config = config
    
    async def analyze(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Analyzing economic conditions")
        
        recommendations = state.get("recommendations", [])
        macro_data = state.get("macro_data", {})
        
        portfolio_sectors = [rec.get("sector", "Unknown") for rec in recommendations]
        unique_sectors = list(set(portfolio_sectors))
        
        economic_prompt = self._build_prompt(macro_data, recommendations, unique_sectors)
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=economic_prompt)])
            state["economic_analysis"] = response.content
            logger.info("Economic analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in economic analysis: {e}")
            state["economic_analysis"] = f"Economic analysis unavailable: {str(e)}"
        
        return state
    
    def _build_prompt(self, macro_data: Dict, recommendations: List[Dict], unique_sectors: List[str]) -> str:
        return f"""
Analyze the current economic conditions and their impact on this stock portfolio:

CURRENT ECONOMIC DATA:
- S&P 500 30-day return: {macro_data.get('economic_indicators', {}).get('sp500_30d_return', 'N/A'):.2f}%
- Market Volatility (VIX): {macro_data.get('economic_indicators', {}).get('market_volatility', 'N/A')}
- Market Sentiment: {macro_data.get('economic_indicators', {}).get('market_sentiment', 'N/A')}
- Dollar Trend: {macro_data.get('economic_indicators', {}).get('dollar_trend', 'N/A')}

SECTOR PERFORMANCE (30-day):
{self._format_sector_performance(macro_data.get('sector_performance', {}))}

PORTFOLIO COMPOSITION:
- Sectors: {', '.join(unique_sectors)}
- Total Stocks: {len(recommendations)}
- Stock Details:
{self._format_recommendations(recommendations)}

Provide a comprehensive economic conditions analysis (3-4 sentences) covering:
1. How current economic conditions favor or challenge this portfolio
2. Sector-specific economic impacts
3. Overall economic environment assessment (bullish/bearish/neutral)
4. Key economic risks or opportunities
"""
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(
                f"{i}. {rec.get('ticker', 'N/A')} - {rec.get('company_name', 'N/A')} "
                f"({rec.get('sector', 'Unknown')} | {rec.get('allocation_percentage', 0):.1f}% | "
                f"₹{rec.get('allocation_amount', 0):,.0f})"
            )
        return '\n'.join(formatted)
    
    def _format_sector_performance(self, sector_data: Dict) -> str:
        if not sector_data or "error" in sector_data:
            return "Sector data unavailable"
        return '\n'.join([f"- {sector}: {perf:+.2f}%" for sector, perf in sector_data.items()])

