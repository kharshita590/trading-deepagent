from typing import TypedDict, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging
import aiohttp

from langgraph import StateGraph, END
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.allocation_agent import AllocationStrategy  

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
    strategy: AllocationStrategy
    total_amount: float
    risk_score: float
    diversification_score: float
    reasoning: str
    target_stocks: int
    preferred_sectors: List[str]
    messages: List
    recommendations: List[StockRecommendation]


class ResearchAgent:
    def __init__(self, llm_model: str = "gpt-4", api_key: str = ""):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.api_key = api_key 

    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(ResearchAgentState)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.set_entry_point("generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        return workflow.compile()

    async def fetch_stock_list(self, sector: str) -> List[Dict]:
        url = f"https://api.twelvedata.com/stocks?sector={sector}&apikey={self.api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
                else:
                    logger.error(f"Failed to fetch stocks for sector {sector}: {resp.status}")
                    return []

    async def fetch_price_info(self, ticker: str) -> float:
        url = f"https://api.twelvedata.com/price?symbol={ticker}&apikey={self.api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("price", 0.0))
                else:
                    logger.error(f"Failed to fetch price for {ticker}: {resp.status}")
                    return 0.0

    async def generate_recommendations(self, state: ResearchAgentState) -> ResearchAgentState:
        logger.info("Generating stock recommendations")

        strategy = state["strategy"]
        amount = state["total_amount"]
        target_stocks = state["target_stocks"]
        preferred_sectors = state["preferred_sectors"]

        recommendations: List[StockRecommendation] = []

        for sector in preferred_sectors:
            stocks_in_sector = await self.fetch_stock_list(sector)
            filtered = []
            for s in stocks_in_sector:
                ticker = s.get("ticker")
                name = s.get("name", "")
                price = await self.fetch_price_info(ticker)
                if price > 0 and price <= amount:
                    filtered.append({
                        "ticker": ticker,
                        "name": name,
                        "sector": sector,
                        "price": price,
                        "market_cap": s.get("market_cap", "")
                    })

            if not filtered:
                continue

            if strategy == AllocationStrategy.SINGLE_STOCK:
                best = max(filtered, key=lambda x: x["price"])
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
            sorted_by_price = sorted(filtered, key=lambda x: x["price"], reverse=True)
            picks = sorted_by_price[:target_stocks]

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
        summary = f"Generated {len(recommendations)} recommendations for {strategy.value}"
        state["messages"].append(AIMessage(content=summary))

        return state
