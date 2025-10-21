import logging
from typing import Dict, List
from datetime import datetime
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from ..models.types import MacroAgentState, MacroAnalysis, MacroData
from ..config.settings import AppConfig

logger = logging.getLogger(__name__)

class GlobalEventsAnalysisAgent:    
    def __init__(self, llm: ChatGoogleGenerativeAI, config: AppConfig):
        self.llm = llm
        self.config = config
    
    async def analyze(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Analyzing global events impact")
        
        recommendations = state.get("recommendations", [])
        macro_data = state.get("macro_data", {})
        
        global_prompt = self._build_prompt(macro_data, recommendations)
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=global_prompt)])
            state["global_events_analysis"] = response.content
            logger.info("Global events analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in global events analysis: {e}")
            state["global_events_analysis"] = f"Global events analysis unavailable: {str(e)}"
        
        return state
    
    def _build_prompt(self, macro_data: Dict, recommendations: List[Dict]) -> str:
        sector_perf = macro_data.get('sector_performance', {})
        return f"""
Analyze global events and geopolitical factors impact on this Indian stock portfolio:

GLOBAL MARKET CONDITIONS:
- Dollar Index Trend: {macro_data.get('economic_indicators', {}).get('dollar_trend', 'N/A')}
- Market Volatility: {macro_data.get('economic_indicators', {}).get('market_volatility', 'N/A')}
- Global Sector Performance:
{self._format_sector_performance(sector_perf)}

PORTFOLIO (Indian Stocks):
{self._format_recommendations(recommendations)}

Consider current global factors and provide analysis (3-4 sentences) covering:
1. Global supply chain impacts on these Indian sectors
2. Currency effects (USD/INR) on export-oriented companies
3. Global commodity price impacts (energy, metals, agricultural)
4. Geopolitical risks and opportunities for Indian markets
5. Foreign investment flows into Indian equities

Focus on how global events specifically affect:
- Technology sector (global demand, outsourcing trends)
- Energy sector (oil prices, renewable transition)
- Financial sector (capital flows, global banking)
- Manufacturing/Auto (supply chains, global demand)
"""
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(
                f"{i}. {rec.get('ticker', 'N/A')} - {rec.get('company_name', 'N/A')} "
                f"({rec.get('sector', 'Unknown')} | {rec.get('allocation_percentage', 0):.1f}%)"
            )
        return '\n'.join(formatted)
    
    def _format_sector_performance(self, sector_data: Dict) -> str:
        if not sector_data or "error" in sector_data:
            return "  Sector data unavailable"
        return '\n'.join([f"  - {sector}: {perf:+.2f}%" for sector, perf in sector_data.items()])
