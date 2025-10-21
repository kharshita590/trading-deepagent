from typing import Dict
import json
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.types import FundamentalAnalysis,FundamentalAgentState
from ..config.settings import logger, GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SECTOR_ETFS

class SectorStrengthAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )
    
    async def execute(self, state: FundamentalAgentState) -> FundamentalAgentState:
        logger.info("Analyzing sector strength")
        financial_data = state.get("financial_data", {})
        sectors_data = {}
        for ticker, data in financial_data.items():
            sector = data["recommendation"].get("sector", "Unknown")
            if sector not in sectors_data:
                sectors_data[sector] = {
                    "stocks": [],
                    "sector_performance": data["peer_comparison"]
                }
            sectors_data[sector]["stocks"].append({
                "ticker": ticker,
                "company": data["recommendation"].get("company_name", ""),
                "allocation": data["recommendation"].get("allocation_percentage", 0),
                "fundamental_score": data["financial_scores"].get("overall_fundamental_score", 0)
            })
        
        sector_analysis_prompt = f"""
        Analyze the sector strength and positioning for this portfolio:
        
        SECTOR BREAKDOWN:
        {json.dumps(sectors_data, indent=2, default=str)}
        
        DETAILED FINANCIAL DATA:
        {self._format_sector_analysis_data(financial_data)}
        
        Provide a comprehensive sector strength summary (4-5 sentences) covering:
        1. Sector diversification quality and concentration risks
        2. Individual sector performance and outlook
        3. Sector rotation trends and positioning
        4. Industry-specific competitive advantages
        5. Regulatory environment and sector-specific risks
        6. Cyclical vs defensive sector balance
        
        Focus on:
        - Technology sector (growth prospects, valuation levels)
        - Financial Services (interest rate sensitivity, NPA levels)
        - Energy sector (commodity price exposure, transition risks)
        - Consumer sectors (spending patterns, brand strength)
        - Sector valuations vs historical averages
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=sector_analysis_prompt)])
            state["sector_strength_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in sector strength analysis: {e}")
            state["sector_strength_analysis"] = f"Sector strength analysis unavailable: {str(e)}"
        
        return state
    
    def _format_sector_analysis_data(self, financial_data: Dict) -> str:
        formatted = []
        for ticker, data in financial_data.items():
            peer = data["peer_comparison"]
            metrics = data["financial_metrics"]
            
            formatted.append(f"""
{ticker}: Sector Performance: {peer.get('sector_1y_return', 0):.1f}% (1Y) | {peer.get('sector_30d_return', 0):.1f}% (30D)
- Company P/E: {metrics.get('pe_ratio', 0):.2f} vs Sector P/E: {peer.get('sector_pe', 0):.2f}
- Beta: {metrics.get('beta', 0):.2f} | Market Cap: ₹{metrics.get('market_cap', 0):,.0f}
""")
        
        return '\n'.join(formatted)
