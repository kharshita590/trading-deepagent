from typing import Dict
import json
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.types import FundamentalAnalysis,FundamentalAgentState
from ..config.settings import logger, GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SECTOR_ETFS

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
            sector_etf = SECTOR_ETFS.get(sector, "SPY")  
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
                min(max((metrics.get("revenue_growth", 0) * 100 + 100) / 2, 0), 100),
                min(max((metrics.get("earnings_growth", 0) * 100 + 100) / 2, 0), 100)
            ]
            scores["growth_score"] = np.mean([x for x in growth_components if x > 0]) if any(x > 0 for x in growth_components) else 0
            
            strength_components = [
                min(max(metrics.get("current_ratio", 0) * 33, 0), 100),
                min(max((5 - metrics.get("debt_to_equity", 5)) * 20, 0), 100),
                min(max(metrics.get("interest_coverage", 0) * 10, 0), 100)
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

class FinancialDataFetchAgent:
    def __init__(self):
        self.data_provider = FinancialDataProvider()
    
    async def execute(self, state: FundamentalAgentState) -> FundamentalAgentState:
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
