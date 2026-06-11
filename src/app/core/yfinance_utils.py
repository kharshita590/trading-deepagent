from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import yfinance as yf

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:  # pragma: no cover
    def retry(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

from .cache import get_or_set


def _period_ttl(period: str, interval: Optional[str] = None) -> int:
    if interval in {"1m", "2m", "5m", "15m", "30m", "60m"}:
        return 300
    if period in {"1d", "5d", "1wk"}:
        return 300
    if period in {"1mo", "3mo", "6mo", "1y", "2y"}:
        return 3600
    return 3600


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _run_blocking(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def fetch_history(ticker: str, period: str = "1y", interval: Optional[str] = None) -> pd.DataFrame:
    key = f"yf:history:{ticker}:{period}:{interval or ''}"

    async def _fetch():
        stock = yf.Ticker(ticker)
        return await _run_blocking(stock.history, period=period, interval=interval) if interval else await _run_blocking(stock.history, period=period)

    data = await get_or_set(key, _fetch, _period_ttl(period, interval))
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


async def fetch_info(ticker: str) -> Dict[str, Any]:
    key = f"yf:info:{ticker}"

    async def _fetch():
        stock = yf.Ticker(ticker)
        return await _run_blocking(lambda: stock.info)

    data = await get_or_set(key, _fetch, 86400)
    return data if isinstance(data, dict) else {}


async def fetch_financial_bundle(ticker: str) -> Dict[str, Any]:
    key = f"yf:bundle:{ticker}"

    async def _fetch():
        stock = yf.Ticker(ticker)
        info = await _run_blocking(lambda: stock.info)
        financials = await _run_blocking(lambda: stock.financials)
        balance_sheet = await _run_blocking(lambda: stock.balance_sheet)
        cashflow = await _run_blocking(lambda: stock.cashflow)
        return {
            "info": info,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cashflow": cashflow,
        }

    bundle = await get_or_set(key, _fetch, 86400)
    return bundle if isinstance(bundle, dict) else {}


async def fetch_download(tickers: Iterable[str], period: str, interval: str = "1d") -> pd.DataFrame:
    tickers_list = list(tickers)
    key = f"yf:download:{','.join(sorted(tickers_list))}:{period}:{interval}"

    async def _fetch():
        return await _run_blocking(
            yf.download,
            tickers_list,
            period=period,
            interval=interval,
            progress=False,
            threads=True,
        )

    data = await get_or_set(key, _fetch, _period_ttl(period, interval))
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)
