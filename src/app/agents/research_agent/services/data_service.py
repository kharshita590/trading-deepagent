from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

import pandas as pd

from app.core.yfinance_utils import fetch_download

logger = logging.getLogger(__name__)


class StockDataService:
    @staticmethod
    async def fetch_prices_batch(tickers: List[str], batch_size: int = 50) -> Dict[str, float]:
        """Fetch stock prices in batches using cached yfinance downloads."""
        all_prices: Dict[str, float] = {}

        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            nse_tickers = [f"{ticker}.NS" for ticker in batch_tickers]

            try:
                data = await fetch_download(nse_tickers, period="1d", interval="1d")

                if not data.empty:
                    if len(batch_tickers) == 1:
                        ticker = batch_tickers[0]
                        if "Close" in data.columns and not data["Close"].empty:
                            all_prices[ticker] = float(data["Close"].iloc[-1])
                        else:
                            all_prices[ticker] = 0.0
                    else:
                        for ticker in batch_tickers:
                            nse_ticker = f"{ticker}.NS"
                            try:
                                if ("Close", nse_ticker) in data.columns:
                                    close_series = data[("Close", nse_ticker)]
                                    all_prices[ticker] = float(close_series.iloc[-1]) if not close_series.empty and not pd.isna(close_series.iloc[-1]) else 0.0
                                else:
                                    all_prices[ticker] = 0.0
                            except Exception as exc:
                                logger.warning("Error processing %s: %s", ticker, exc)
                                all_prices[ticker] = 0.0
                else:
                    for ticker in batch_tickers:
                        all_prices[ticker] = 0.0
            except Exception as exc:
                logger.error("Batch fetch failed for batch starting at %s: %s", i, exc)
                for ticker in batch_tickers:
                    all_prices[ticker] = 0.0

            if i + batch_size < len(tickers):
                await asyncio.sleep(0.1)

        logger.info("Fetched prices for %s stocks in batches", len(all_prices))
        return all_prices

    @staticmethod
    def parse_market_cap(mc_str: str) -> float:
        """Parse market cap string to numeric value."""
        if not mc_str:
            return 0

        mc_str = str(mc_str).upper().replace(",", "")

        try:
            if "B" in mc_str:
                return float(mc_str.replace("B", "")) * 1_000_000_000
            if "M" in mc_str:
                return float(mc_str.replace("M", "")) * 1_000_000
            if "K" in mc_str:
                return float(mc_str.replace("K", "")) * 1_000
            return float(mc_str) if mc_str.isdigit() else 0
        except Exception:
            return 0
