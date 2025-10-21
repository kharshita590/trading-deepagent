from typing import Dict
import json
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.types import FundamentalAnalysis,FundamentalAgentState
from ..config.settings import logger, GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SECTOR_ETFS

class InvestmentThesisAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )
    
    async def execute(self, state: FundamentalAgentState) -> FundamentalAgentState:
        logger.info("Generating fundamental investment thesis")
        
        financial_data = state.get("financial_data", {})
        thesis_prompt = f"""
        Generate a fundamental investment thesis for this portfolio based on detailed financial analysis:
        
        PORTFOLIO FINANCIAL SUMMARY:
        {self._format_investment_thesis_data(financial_data)}
        
        COMPANY FINANCIALS CONTEXT:
        {state.get("company_financials_analysis", "No analysis available")}
        
        SECTOR STRENGTH CONTEXT:
        {state.get("sector_strength_analysis", "No analysis available")}
        
        Provide a comprehensive fundamental investment thesis (4-5 sentences) covering:
        1. Overall portfolio fundamental attractiveness
        2. Key investment strengths based on financial metrics
        3. Risk factors from fundamental perspective
        4. Expected fundamental performance drivers
        5. Long-term fundamental outlook and conviction level
        6. Price targets and fundamental fair value assessment
        
        Consider:
        - Quality of earnings and revenue streams
        - Balance sheet strength and capital allocation
        - Competitive moats and market positioning
        - Management effectiveness and corporate governance
        - Fundamental valuation vs market price
        - Catalyst events and fundamental inflection points
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=thesis_prompt)])
            state["investment_thesis_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in investment thesis generation: {e}")
            state["investment_thesis_analysis"] = f"Investment thesis unavailable: {str(e)}"
        
        return state
    
    def _format_investment_thesis_data(self, financial_data: Dict) -> str:
        total_allocation = sum(data["recommendation"].get("allocation_amount", 0) for data in financial_data.values())
        avg_fundamental_score = np.mean([data["financial_scores"].get("overall_fundamental_score", 0) for data in financial_data.values()])
        
        formatted = [f"""
PORTFOLIO OVERVIEW:
- Total Investment: ₹{total_allocation:,.0f}
- Average Fundamental Score: {avg_fundamental_score:.1f}/100
- Stock Count: {len(financial_data)}

TOP PERFORMERS BY FUNDAMENTAL SCORE:
"""]
        
        sorted_stocks = sorted(financial_data.items(), key=lambda x: x[1]["financial_scores"].get("overall_fundamental_score", 0), reverse=True)
        
        for ticker, data in sorted_stocks[:3]:
            score = data["financial_scores"].get("overall_fundamental_score", 0)
            company = data["recommendation"].get("company_name", "")
            allocation = data["recommendation"].get("allocation_percentage", 0)
            formatted.append(f"- {company} ({ticker}): {score:.1f}/100 ({allocation:.1f}% allocation)")
        
        return '\n'.join(formatted)