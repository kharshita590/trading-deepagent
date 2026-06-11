from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

from langchain_core.messages import AIMessage

from app.core.yfinance_utils import fetch_history

from ..config.settings import AppConfig
from ..models.types import MacroAgentState

logger = logging.getLogger(__name__)


class DataFetcherAgent:
    def __init__(self, config: AppConfig):
        self.config = config

    async def get_interest_rates(self) -> Dict:
        try:
            treasury_data = await fetch_history(self.config.MARKET_INDICES["treasury_10y"], period=self.config.data_provider.interest_rate_period)
            fed_data = await fetch_history(self.config.MARKET_INDICES["fed_rate"], period=self.config.data_provider.interest_rate_period)
            return {
                "treasury_10y_rate": treasury_data["Close"].iloc[-1] if not treasury_data.empty else 0,
                "short_term_rate": fed_data["Close"].iloc[-1] if not fed_data.empty else 0,
                "rate_trend": "rising" if not treasury_data.empty and treasury_data["Close"].iloc[-1] > treasury_data["Close"].iloc[0] else "falling",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
            }
        except Exception as e:
            logger.error("Error fetching interest rate data: %s", e)
            return {"error": str(e)}

    async def get_economic_indicators(self) -> Dict:
        try:
            spy_data = await fetch_history(self.config.MARKET_INDICES["sp500"], period=self.config.data_provider.economic_indicator_period)
            vix_data = await fetch_history(self.config.MARKET_INDICES["volatility"], period=self.config.data_provider.interest_rate_period)
            dxy_data = await fetch_history(self.config.MARKET_INDICES["dollar_index"], period=self.config.data_provider.economic_indicator_period)
            vix_value = vix_data["Close"].iloc[-1] if not vix_data.empty else 0
            sentiment = self._determine_sentiment(vix_value)
            return {
                "sp500_30d_return": ((spy_data["Close"].iloc[-1] / spy_data["Close"].iloc[0]) - 1) * 100 if not spy_data.empty else 0,
                "market_volatility": vix_value,
                "dollar_strength": dxy_data["Close"].iloc[-1] if not dxy_data.empty else 0,
                "dollar_trend": "strengthening" if not dxy_data.empty and dxy_data["Close"].iloc[-1] > dxy_data["Close"].iloc[0] else "weakening",
                "market_sentiment": sentiment,
            }
        except Exception as e:
            logger.error("Error fetching economic indicators: %s", e)
            return {"error": str(e)}

    async def get_sector_performance(self) -> Dict:
        try:
            sector_performance: Dict[str, float] = {}
            for sector, etf in self.config.SECTOR_ETFS.items():
                data = await fetch_history(etf, period=self.config.data_provider.sector_performance_period)
                if not data.empty:
                    performance = ((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100
                    sector_performance[sector] = round(performance, 2)
            return sector_performance
        except Exception as e:
            logger.error("Error fetching sector performance: %s", e)
            return {"error": str(e)}

    def _determine_sentiment(self, vix_value: float) -> str:
        thresholds = self.config.VOLATILITY_THRESHOLDS
        if vix_value >= thresholds["extreme"]:
            return "extreme_fear"
        if vix_value >= thresholds["high"]:
            return "fear"
        if vix_value >= thresholds["moderate"]:
            return "neutral"
        return "greed"

    async def fetch_all_data(self, state: MacroAgentState) -> MacroAgentState:
        logger.info("Fetching macro-economic data")
        interest_data = await self.get_interest_rates()
        economic_data = await self.get_economic_indicators()
        sector_data = await self.get_sector_performance()
        state["macro_data"] = {
            "interest_rates": interest_data,
            "economic_indicators": economic_data,
            "sector_performance": sector_data,
            "fetch_timestamp": datetime.now().isoformat(),
            "market_sentiment": economic_data.get("market_sentiment"),
        }
        state["market_sentiment"] = economic_data.get("market_sentiment")
        state["messages"].append(AIMessage(content="Fetched real-time macro-economic data"))
        return state
