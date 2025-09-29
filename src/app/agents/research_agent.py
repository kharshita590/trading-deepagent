from typing import TypedDict, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging
import aiohttp

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

from app.agents.investment_allocation import AllocationStrategy  
import yfinance as yf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockRecommendation:
    ticker: str
    company_name: str
    price: float
    allocation_percentage: float
    allocation_amount: float
    sector: str
    market_cap: str
    reasoning: str

class ResearchAgentState(TypedDict):
    allocation_strategy: AllocationStrategy
    investment_amount: float
    risk_score: float
    diversification_score: float
    reasoning: str
    preferred_sectors: List[str]
    messages: List
    recommendations: List[StockRecommendation]

class ResearchAgent:
    def __init__(self, llm_model: str = "gpt-4", api_key: str = ""):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0,
            google_api_key="AIzaSyDl0-DuUoAmjs4hjM8E7TnRL7qazQ2Bq8w"
        )
        self.api_key = "3f515b7af72944d582da4926e99accbf"

    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(ResearchAgentState)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.set_entry_point("generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        return workflow.compile()

    async def fetch_stock_list(self, sector: str) -> List[Dict]:
        url = f"https://api.twelvedata.com/stocks"
        params = {"sector": sector,"country": "India","apikey": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
                else:
                    logger.error(f"Failed to fetch stocks for sector {sector}: {resp.status}")
                    return []

    async def fetch_multiple_prices_batch(self, tickers: List[str], batch_size: int = 50) -> Dict[str, float]:
        all_prices = {}        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            nse_tickers = [f"{ticker}.NS" for ticker in batch_tickers]
            try:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    data = await loop.run_in_executor(
                        executor,
                        lambda: yf.download(nse_tickers, period="1d", interval="1d", progress=False, threads=True)
                    )                    
                    if not data.empty:
                        if len(batch_tickers) == 1:
                            ticker = batch_tickers[0]
                            if 'Close' in data.columns and not data['Close'].empty:
                                all_prices[ticker] = float(data['Close'].iloc[-1])
                            else:
                                all_prices[ticker] = 0.0
                        else:
                            for ticker in batch_tickers:
                                nse_ticker = f"{ticker}.NS"
                                try:
                                    if ('Close', nse_ticker) in data.columns:
                                        close_series = data[('Close', nse_ticker)]
                                        if not close_series.empty and not pd.isna(close_series.iloc[-1]):
                                            all_prices[ticker] = float(close_series.iloc[-1])
                                        else:
                                            all_prices[ticker] = 0.0
                                    else:
                                        all_prices[ticker] = 0.0
                                except Exception as e:
                                    logger.warning(f"Error processing {ticker}: {e}")
                                    all_prices[ticker] = 0.0
                    else:
                        for ticker in batch_tickers:
                            all_prices[ticker] = 0.0
                            
            except Exception as e:
                logger.error(f"Batch fetch failed for batch starting at {i}: {e}")
                for ticker in batch_tickers:
                    all_prices[ticker] = 0.0
            
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.1)
        
        logger.info(f"Fetched prices for {len(all_prices)} stocks in {len(range(0, len(tickers), batch_size))} batches")
        return all_prices

    async def generate_recommendations(self, state: ResearchAgentState) -> ResearchAgentState:
        logger.info("Generating stock recommendations")
        strategy = state["allocation_strategy"]
        amount = state["investment_amount"]
        preferred_sectors = state["preferred_sectors"]
        recommendations: List[StockRecommendation] = []
        print(f"Strategy: {strategy}, Amount: {amount}, Sectors: {preferred_sectors}")
        for sector in preferred_sectors:
            stocks_in_sector = await self.fetch_stock_list(sector)
            print(f"Found {len(stocks_in_sector)} stocks in {sector}")
            if not stocks_in_sector:
                continue
            max_stocks_to_process = 200             
            filtered_stocks = []
            for stock in stocks_in_sector:
                if stock.get("symbol") and stock.get("name"):
                    market_cap = stock.get("market_cap", "")
                    filtered_stocks.append({
                        "symbol": stock.get("symbol"),
                        "name": stock.get("name"),
                        "market_cap": market_cap,
                        "original": stock
                    })            
            if filtered_stocks:
                try:
                    def parse_market_cap(mc_str):
                        if not mc_str or mc_str == "":
                            return 0
                        mc_str = str(mc_str).upper().replace(",", "")
                        if "B" in mc_str:
                            return float(mc_str.replace("B", "")) * 1000000000
                        elif "M" in mc_str:
                            return float(mc_str.replace("M", "")) * 1000000
                        elif "K" in mc_str:
                            return float(mc_str.replace("K", "")) * 1000
                        else:
                            return float(mc_str) if mc_str.isdigit() else 0
                    
                    filtered_stocks.sort(key=lambda x: parse_market_cap(x["market_cap"]), reverse=True)
                    print(f"Sorted {len(filtered_stocks)} stocks by market cap")
                except:
                    print(f"Could not sort by market cap, using original order")                
                selected_stocks = filtered_stocks[:max_stocks_to_process]
                print(f"Processing top {len(selected_stocks)} stocks from {sector}")
            else:
                selected_stocks = []
            if not selected_stocks:
                continue            
            tickers = [s["symbol"] for s in selected_stocks]
            print(f"Fetching prices for {len(tickers)} selected tickers in {sector}...")            
            prices = await self.fetch_multiple_prices_batch(tickers, batch_size=50)            
            price_filtered = []
            for selected_stock in selected_stocks:
                ticker = selected_stock["symbol"]
                price = prices.get(ticker, 0.0)
                if price > 0 and price <= amount:
                    price_filtered.append({
                        "ticker": ticker,
                        "name": selected_stock["name"],
                        "sector": sector,
                        "price": price,
                        "market_cap": selected_stock["market_cap"]
                    })
            
            print(f"After price filtering: {len(price_filtered)} valid stocks in {sector}")
            if not price_filtered:
                continue
            if strategy == AllocationStrategy.SINGLE_STOCK:
                best = max(price_filtered, key=lambda x: x["price"])
                recommendations.append(StockRecommendation(
                    ticker=best["ticker"],
                    company_name=best["name"],
                    price=best["price"],
                    allocation_percentage=100.0,
                    allocation_amount=amount,
                    sector=sector,
                    market_cap=best["market_cap"],
                    reasoning=f"Single stock pick from sector {sector}"
                ))
                break 
            else:
                sorted_by_price = sorted(price_filtered, key=lambda x: x["price"], reverse=True)
                picks = sorted_by_price[:3]
                if picks:
                    equal_alloc = amount / len(picks)
                    for p in picks:
                        allocation_pct = 100.0 / len(picks)
                        recommendations.append(StockRecommendation(
                            ticker=p["ticker"],
                            company_name=p["name"],
                            price=p["price"],
                            allocation_percentage=allocation_pct,
                            allocation_amount=equal_alloc,
                            sector=p["sector"],
                            market_cap=p["market_cap"],
                            reasoning=f"Pick from {sector} sector"
                        ))

        state["recommendations"] = recommendations
        total_allocated = sum(rec.allocation_amount for rec in recommendations)
        summary = f"Generated {len(recommendations)} recommendations for {strategy.value}. Total allocated: ₹{total_allocated:,.0f}"
        state["messages"].append(AIMessage(content=summary))
        return state