from typing import TypedDict, List, Dict, Optional
import json
from dataclasses import dataclass
import asyncio
import logging
import requests
import yfinance as yf
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.graph.message import HumanMessage, AIMessage
from langchain_core.messages import AIMessage
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MacroAnalysis:
    economic_conditions_summary: str
    interest_rate_impact_summary: str
    global_events_summary: str

class MacroAgentState(TypedDict):
    recommendations: List[Dict]  
    macro_analysis: Optional[MacroAnalysis]
    messages: List

class MacroDataProvider:    
    @staticmethod
    def get_interest_rates() -> Dict:
        try:
            treasury_10y = yf.Ticker("^TNX") 
            fed_rate = yf.Ticker("^IRX")    
            
            treasury_data = treasury_10y.history(period="5d")
            fed_data = fed_rate.history(period="5d")
            
            return {
                "treasury_10y_rate": treasury_data['Close'].iloc[-1] if not treasury_data.empty else 0,
                "short_term_rate": fed_data['Close'].iloc[-1] if not fed_data.empty else 0,
                "rate_trend": "rising" if treasury_data['Close'].iloc[-1] > treasury_data['Close'].iloc[0] else "falling",
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
        except Exception as e:
            logger.error(f"Error fetching interest rate data: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_economic_indicators() -> Dict:
        try:
            spy = yf.Ticker("SPY") 
            vix = yf.Ticker("^VIX") 
            dxy = yf.Ticker("DX-Y.NYB")  
            
            spy_data = spy.history(period="30d")
            vix_data = vix.history(period="5d")
            dxy_data = dxy.history(period="30d")
            
            return {
                "sp500_30d_return": ((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1) * 100 if not spy_data.empty else 0,
                "market_volatility": vix_data['Close'].iloc[-1] if not vix_data.empty else 0,
                "dollar_strength": dxy_data['Close'].iloc[-1] if not dxy_data.empty else 0,
                "dollar_trend": "strengthening" if dxy_data['Close'].iloc[-1] > dxy_data['Close'].iloc[0] else "weakening",
                "market_sentiment": "fear" if (vix_data['Close'].iloc[-1] if not vix_data.empty else 0) > 30 else "greed"
            }
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_sector_performance() -> Dict:
        try:
            sectors = {
                "Technology": "XLK",
                "Financials": "XLF", 
                "Energy": "XLE",
                "Healthcare": "XLV",
                "Consumer Goods": "XLY"
            }
            sector_performance = {}
            for sector, etf in sectors.items():
                ticker = yf.Ticker(etf)
                data = ticker.history(period="30d")
                if not data.empty:
                    performance = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    sector_performance[sector] = performance
            
            return sector_performance
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {"error": str(e)}

class MacroAgent:
    def __init__(self, llm_model="gpt-4"):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0,
        google_api_key="")
        self.data_provider = MacroDataProvider()

    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(MacroAgentState)
        
        workflow.add_node("fetch_macro_data", self.fetch_macro_data)
        workflow.add_node("analyze_economic_conditions", self.analyze_economic_conditions)
        workflow.add_node("analyze_interest_rate_impact", self.analyze_interest_rate_impact)
        workflow.add_node("analyze_global_events", self.analyze_global_events)
        workflow.add_node("compile_macro_analysis", self.compile_macro_analysis)
        
        workflow.add_edge("fetch_macro_data", "analyze_economic_conditions")
        workflow.add_edge("analyze_economic_conditions", "analyze_interest_rate_impact")
        workflow.add_edge("analyze_interest_rate_impact", "analyze_global_events")
        workflow.add_edge("analyze_global_events", "compile_macro_analysis")
        workflow.add_edge("compile_macro_analysis", END)
        
        workflow.set_entry_point("fetch_macro_data")
        
        return workflow

    async def fetch_macro_data(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Fetching macro-economic data")
        interest_data = self.data_provider.get_interest_rates()
        economic_data = self.data_provider.get_economic_indicators()        
        sector_data = self.data_provider.get_sector_performance()
        
        state["macro_data"] = {
            "interest_rates": interest_data,
            "economic_indicators": economic_data,
            "sector_performance": sector_data,
            "fetch_timestamp": datetime.now().isoformat()
        }
        
        state["messages"].append(AIMessage(content="Fetched real-time macro-economic data"))
        return state

    async def analyze_economic_conditions(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Analyzing economic conditions")
        
        recommendations = state.get("recommendations", [])
        macro_data = state.get("macro_data", {})        
        portfolio_sectors = [rec.get("sector", "Unknown") for rec in recommendations]
        unique_sectors = list(set(portfolio_sectors))
        
        economic_prompt = f"""
        Analyze the current economic conditions and their impact on this stock portfolio:
        
        CURRENT ECONOMIC DATA:
        - S&P 500 30-day return: {macro_data.get('economic_indicators', {}).get('sp500_30d_return', 'N/A'):.2f}%
        - Market Volatility (VIX): {macro_data.get('economic_indicators', {}).get('market_volatility', 'N/A')}
        - Market Sentiment: {macro_data.get('economic_indicators', {}).get('market_sentiment', 'N/A')}
        - Dollar Trend: {macro_data.get('economic_indicators', {}).get('dollar_trend', 'N/A')}
        
        SECTOR PERFORMANCE (30-day):
        {json.dumps(macro_data.get('sector_performance', {}), indent=2)}
        
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
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=economic_prompt)])
            state["economic_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in economic analysis: {e}")
            state["economic_analysis"] = f"Economic analysis unavailable: {str(e)}"
        
        return state

    async def analyze_interest_rate_impact(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Analyzing interest rate impact")
        
        recommendations = state.get("recommendations", [])
        macro_data = state.get("macro_data", {})
        interest_data = macro_data.get("interest_rates", {})
        
        rate_prompt = f"""
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
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=rate_prompt)])
            state["interest_rate_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in interest rate analysis: {e}")
            state["interest_rate_analysis"] = f"Interest rate analysis unavailable: {str(e)}"
        
        return state

    async def analyze_global_events(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Analyzing global events impact")
        
        recommendations = state.get("recommendations", [])
        macro_data = state.get("macro_data", {})        
        global_prompt = f"""
        Analyze global events and geopolitical factors impact on this Indian stock portfolio:
        
        GLOBAL MARKET CONDITIONS:
        - Dollar Index Trend: {macro_data.get('economic_indicators', {}).get('dollar_trend', 'N/A')}
        - Market Volatility: {macro_data.get('economic_indicators', {}).get('market_volatility', 'N/A')}
        - Global Sector Performance: {json.dumps(macro_data.get('sector_performance', {}), indent=2)}
        
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
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=global_prompt)])
            state["global_events_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in global events analysis: {e}")
            state["global_events_analysis"] = f"Global events analysis unavailable: {str(e)}"
        
        return state

    async def compile_macro_analysis(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Compiling macro analysis")
        macro_analysis = MacroAnalysis(
            economic_conditions_summary=state.get("economic_analysis", "Analysis unavailable"),
            interest_rate_impact_summary=state.get("interest_rate_analysis", "Analysis unavailable"),
            global_events_summary=state.get("global_events_analysis", "Analysis unavailable")
        )
        
        state["macro_analysis"] = macro_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive macro-economic analysis")) 
        return state

    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        formatted = []
        for i, rec in enumerate(recommendations, 1):
            formatted.append(
                f"{i}. {rec.get('ticker', 'N/A')} - {rec.get('company_name', 'N/A')} "
                f"({rec.get('sector', 'Unknown')} | {rec.get('allocation_percentage', 0):.1f}% | "
                f"₹{rec.get('allocation_amount', 0):,.0f})"
            )
        return '\n'.join(formatted)
