from typing import TypedDict, List, Dict, Optional
import json
from dataclasses import dataclass
import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from langgraph import StateGraph, END
from langgraph.graph.message import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FundamentalAnalysis:
    company_financials_summary: str
    sector_strength_summary: str
    fundamental_investment_thesis: str

class FundamentalAgentState(TypedDict):
    recommendations: List[Dict]  
    fundamental_analysis: Optional[FundamentalAnalysis]
    messages: List

class FinancialDataProvider:
    
    @staticmethod
    def get_financial_metrics(ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            metrics = {
                "pe_ratio": info.get('trailingPE', 0),
                "forward_pe": info.get('forwardPE', 0),
                "pb_ratio": info.get('priceToBook', 0),
                "ps_ratio": info.get('priceToSalesTrailing12Months', 0),
                "peg_ratio": info.get('pegRatio', 0),
                "ev_ebitda": info.get('enterpriseToEbitda', 0),
                
                "roe": info.get('returnOnEquity', 0),
                "roa": info.get('returnOnAssets', 0),
                "gross_margin": info.get('grossMargins', 0),
                "operating_margin": info.get('operatingMargins', 0),
                "profit_margin": info.get('profitMargins', 0),
                
                "revenue_growth": info.get('revenueGrowth', 0),
                "earnings_growth": info.get('earningsGrowth', 0),
                "revenue_growth_quarterly": info.get('revenueQuarterlyGrowth', 0),
                "earnings_growth_quarterly": info.get('earningsQuarterlyGrowth', 0),
                
                "current_ratio": info.get('currentRatio', 0),
                "quick_ratio": info.get('quickRatio', 0),
                "debt_to_equity": info.get('debtToEquity', 0),
                "interest_coverage": info.get('interestCoverage', 0),
                "free_cash_flow": info.get('freeCashflow', 0),
                "operating_cash_flow": info.get('operatingCashflow', 0),
                
                "market_cap": info.get('marketCap', 0),
                "enterprise_value": info.get('enterpriseValue', 0),
                "shares_outstanding": info.get('sharesOutstanding', 0),
                "float_shares": info.get('floatShares', 0),
                "beta": info.get('beta', 1.0),
                
                "dividend_yield": info.get('dividendYield', 0),
                "payout_ratio": info.get('payoutRatio', 0),
                "dividend_rate": info.get('dividendRate', 0),
                
                "52_week_high": info.get('fiftyTwoWeekHigh', 0),
                "52_week_low": info.get('fiftyTwoWeekLow', 0),
                "50_day_avg": info.get('fiftyDayAverage', 0),
                "200_day_avg": info.get('twoHundredDayAverage', 0),
                
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "business_summary": info.get('businessSummary', ''),
                "employee_count": info.get('fullTimeEmployees', 0),
                
                "has_financials": not financials.empty,
                "has_balance_sheet": not balance_sheet.empty,
                "has_cashflow": not cashflow.empty,
                "last_fiscal_year_end": info.get('lastFiscalYearEnd', 'Unknown')
            }
            
            if metrics["free_cash_flow"] and metrics["market_cap"]:
                metrics["fcf_yield"] = (metrics["free_cash_flow"] / metrics["market_cap"]) * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    @staticmethod
    def get_peer_comparison(ticker: str, sector: str) -> Dict:
        try:
            sector_etfs = {
                "Technology": "XLK",
                "Financial Services": "XLF", 
                "Energy": "XLE",
                "Healthcare": "XLV",
                "Consumer Cyclical": "XLY",
                "Consumer Defensive": "XLP",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Communication Services": "XLC"
            }
            
            sector_etf = sector_etfs.get(sector, "SPY")  
            etf_ticker = yf.Ticker(sector_etf)
            etf_info = etf_ticker.info
            
            etf_history = etf_ticker.history(period="1y")
            sector_performance = {
                "sector_1y_return": ((etf_history['Close'].iloc[-1] / etf_history['Close'].iloc[0]) - 1) * 100 if not etf_history.empty else 0,
                "sector_30d_return": ((etf_history['Close'].iloc[-1] / etf_history['Close'].iloc[-22]) - 1) * 100 if len(etf_history) > 22 else 0,
                "sector_ytd_return": ((etf_history['Close'].iloc[-1] / etf_history['Close'].iloc[0]) - 1) * 100 if not etf_history.empty else 0,
                "sector_pe": etf_info.get('trailingPE', 0),
                "sector_etf": sector_etf,
                "sector_expense_ratio": etf_info.get('totalExpenseRatio', 0)
            }
            
            return sector_performance
            
        except Exception as e:
            logger.error(f"Error fetching peer comparison for {ticker}: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_financial_scores(metrics: Dict) -> Dict:
        """Calculate composite financial health scores"""
        scores = {}
        
        try:
            profitability_components = [
                min(max(metrics.get("roe", 0) * 5, 0), 100),  
                min(max(metrics.get("roa", 0) * 10, 0), 100), 
                min(max(metrics.get("gross_margin", 0) * 100, 0), 100),  
                min(max(metrics.get("operating_margin", 0) * 100, 0), 100),  
                min(max(metrics.get("profit_margin", 0) * 100, 0), 100)  
            ]
            scores["profitability_score"] = np.mean([x for x in profitability_components if x > 0]) if any(x > 0 for x in profitability_components) else 0
            
            growth_components = [
                min(max((metrics.get("revenue_growth", 0) * 100 + 100) / 2, 0), 100),  # Revenue growth normalized
                min(max((metrics.get("earnings_growth", 0) * 100 + 100) / 2, 0), 100)  # Earnings growth normalized
            ]
            scores["growth_score"] = np.mean([x for x in growth_components if x > 0]) if any(x > 0 for x in growth_components) else 0
            
            strength_components = [
                min(max(metrics.get("current_ratio", 0) * 33, 0), 100),  # Current ratio scaled
                min(max((5 - metrics.get("debt_to_equity", 5)) * 20, 0), 100),  # Lower debt is better
                min(max(metrics.get("interest_coverage", 0) * 10, 0), 100)  # Interest coverage scaled
            ]
            scores["financial_strength_score"] = np.mean([x for x in strength_components if x >= 0]) if strength_components else 0
            
            valuation_components = []
            if metrics.get("pe_ratio", 0) > 0:
                valuation_components.append(min(max((50 - metrics.get("pe_ratio", 0)) * 2, 0), 100))
            if metrics.get("pb_ratio", 0) > 0:
                valuation_components.append(min(max((10 - metrics.get("pb_ratio", 0)) * 10, 0), 100))
            scores["valuation_score"] = np.mean(valuation_components) if valuation_components else 50
            
            scores["overall_fundamental_score"] = (
                scores["profitability_score"] * 0.3 +
                scores["growth_score"] * 0.25 +
                scores["financial_strength_score"] * 0.25 +
                scores["valuation_score"] * 0.2
            )
            
        except Exception as e:
            logger.error(f"Error calculating financial scores: {e}")
            scores = {"error": str(e)}
        
        return scores

class FundamentalAgent:
    def __init__(self, llm_model="gpt-4"):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)
        self.data_provider = FinancialDataProvider()

    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(FundamentalAgentState)
        
        workflow.add_node("fetch_financial_data", self.fetch_financial_data)
        workflow.add_node("analyze_company_financials", self.analyze_company_financials)
        workflow.add_node("analyze_sector_strength", self.analyze_sector_strength)
        workflow.add_node("generate_investment_thesis", self.generate_investment_thesis)
        workflow.add_node("compile_fundamental_analysis", self.compile_fundamental_analysis)
        
        workflow.add_edge("fetch_financial_data", "analyze_company_financials")
        workflow.add_edge("analyze_company_financials", "analyze_sector_strength")
        workflow.add_edge("analyze_sector_strength", "generate_investment_thesis")
        workflow.add_edge("generate_investment_thesis", "compile_fundamental_analysis")
        workflow.add_edge("compile_fundamental_analysis", END)
        
        workflow.set_entry_point("fetch_financial_data")
        
        return workflow

    async def fetch_financial_data(self, state: FundamentalAgentState) -> FundamentalAgentState:
        logger.info("Fetching financial data for all recommendations")
        
        recommendations = state.get("recommendations", [])
        financial_data = {}
        
        for rec in recommendations:
            ticker = rec.get("ticker", "")
            sector = rec.get("sector", "Unknown")
            
            metrics = self.data_provider.get_financial_metrics(ticker)            
            peer_data = self.data_provider.get_peer_comparison(ticker, sector)            
            scores = self.data_provider.calculate_financial_scores(metrics)
            financial_data[ticker] = {
                "recommendation": rec,
                "financial_metrics": metrics,
                "peer_comparison": peer_data,
                "financial_scores": scores
            }
        
        state["financial_data"] = financial_data
        state["messages"].append(AIMessage(content=f"Fetched financial data for {len(recommendations)} stocks"))
        
        return state

    async def analyze_company_financials(self, state: FundamentalAgentState) -> FundamentalAgentState:
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

    async def analyze_sector_strength(self, state: FundamentalAgentState) -> FundamentalAgentState:
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

    async def generate_investment_thesis(self, state: FundamentalAgentState) -> FundamentalAgentState:
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

    async def compile_fundamental_analysis(self, state: FundamentalAgentState) -> FundamentalAgentState:
        """Compile all fundamental analysis into final summary"""
        logger.info("Compiling fundamental analysis")
        
        fundamental_analysis = FundamentalAnalysis(
            company_financials_summary=state.get("company_financials_analysis", "Analysis unavailable"),
            sector_strength_summary=state.get("sector_strength_analysis", "Analysis unavailable"),
            fundamental_investment_thesis=state.get("investment_thesis_analysis", "Analysis unavailable")
        )
        
        state["fundamental_analysis"] = fundamental_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive fundamental analysis"))
        
        return state

    def _format_financial_data_for_analysis(self, financial_data: Dict) -> str:
        """Format financial data for company analysis prompt"""
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

    def _format_sector_analysis_data(self, financial_data: Dict) -> str:
        """Format data for sector analysis"""
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

    def _format_investment_thesis_data(self, financial_data: Dict) -> str:
        """Format data for investment thesis"""
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
        
        for ticker, data in sorted_stocks[:3]:  # Top 3
            score = data["financial_scores"].get("overall_fundamental_score", 0)
            company = data["recommendation"].get("company_name", "")
            allocation = data["recommendation"].get("allocation_percentage", 0)
            formatted.append(f"- {company} ({ticker}): {score:.1f}/100 ({allocation:.1f}% allocation)")
        
        return '\n'.join(formatted)
