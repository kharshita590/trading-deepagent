from typing import Dict
import json
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.types import FundamentalAnalysis,FundamentalAgentState
from ..config.settings import logger, GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SECTOR_ETFS

class CompanyFinancialsAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )
    
    async def execute(self, state: FundamentalAgentState) -> FundamentalAgentState:
        logger.info("Analyzing company financials")
        
        financial_data = state.get("financial_data", {})        
        company_analysis_prompt = f"""
        Analyze the financial health and fundamentals of these companies:
        
        {self._format_financial_data_for_analysis(financial_data)}
        
        Provide a comprehensive company financials summary (4-5 sentences) covering:
        1. Overall financial health assessment for each company
        2. Key financial strengths and weaknesses
        3. Profitability, growth, and valuation analysis
        4. Financial stability and debt management
        5. Cash flow generation and dividend sustainability
        6. Relative financial performance ranking within the portfolio
        
        Focus on:
        - P/E ratios, ROE, ROA, profit margins
        - Revenue and earnings growth trends
        - Balance sheet strength (debt levels, liquidity)
        - Cash flow generation capabilities
        - Valuation attractiveness vs fundamentals
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=company_analysis_prompt)])
            state["company_financials_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in company financials analysis: {e}")
            state["company_financials_analysis"] = f"Company financials analysis unavailable: {str(e)}"
        
        return state
    
    def _format_financial_data_for_analysis(self, financial_data: Dict) -> str:
        formatted = []
        for ticker, data in financial_data.items():
            metrics = data["financial_metrics"]
            scores = data["financial_scores"]
            rec = data["recommendation"]
            
            formatted.append(f"""
COMPANY: {rec.get('company_name', 'N/A')} ({ticker})
- Sector: {metrics.get('sector', 'Unknown')} | Allocation: {rec.get('allocation_percentage', 0):.1f}%
- Valuation: P/E: {metrics.get('pe_ratio', 0):.2f} | P/B: {metrics.get('pb_ratio', 0):.2f} | EV/EBITDA: {metrics.get('ev_ebitda', 0):.2f}
- Profitability: ROE: {metrics.get('roe', 0)*100:.1f}% | ROA: {metrics.get('roa', 0)*100:.1f}% | Net Margin: {metrics.get('profit_margin', 0)*100:.1f}%
- Growth: Revenue: {metrics.get('revenue_growth', 0)*100:.1f}% | Earnings: {metrics.get('earnings_growth', 0)*100:.1f}%
- Financial Strength: Current Ratio: {metrics.get('current_ratio', 0):.2f} | D/E: {metrics.get('debt_to_equity', 0):.2f}
- Scores: Overall: {scores.get('overall_fundamental_score', 0):.1f}/100 | Profitability: {scores.get('profitability_score', 0):.1f}/100
""")
        
        return '\n'.join(formatted)