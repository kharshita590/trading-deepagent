from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from typing import Dict, List

import pandas as pd
import vectorbt as vbt

from app.agents.technical_agent.services.technical_data import TechnicalDataProvider
from app.core.yfinance_utils import fetch_history


async def _build_signal_series(ticker: str, period: str) -> pd.DataFrame:
    price_data = await fetch_history(f"{ticker}.NS" if not ticker.endswith(".NS") else ticker, period=period)
    if price_data.empty:
        return pd.DataFrame()

    provider = TechnicalDataProvider()
    records: List[Dict] = []
    window = max(provider.config.indicators.ema_periods + provider.config.indicators.sma_periods) + 20

    for idx in range(window, len(price_data)):
        window_df = price_data.iloc[: idx + 1]
        indicators = provider.calculate_technical_indicators(window_df)
        patterns = provider.detect_chart_patterns(window_df)
        signals = provider.generate_trading_signals(indicators, patterns, float(window_df["Close"].iloc[-1]))
        records.append(
            {
                "index": window_df.index[-1],
                "close": float(window_df["Close"].iloc[-1]),
                "signal": signals.get("overall_signal", "HOLD"),
            }
        )

    return pd.DataFrame(records).set_index("index") if records else pd.DataFrame()


async def run_backtest(ticker: str, period: str) -> Dict:
    signals = await _build_signal_series(ticker, period)
    if signals.empty:
        return {"ticker": ticker, "period": period, "error": "No data available"}

    closes = signals["close"]
    entries = signals["signal"].isin(["BUY", "STRONG BUY"])
    exits = signals["signal"].isin(["SELL", "STRONG SELL"])
    portfolio = vbt.Portfolio.from_signals(closes, entries, exits, init_cash=100000)
    stats = portfolio.stats()
    return {
        "ticker": ticker,
        "period": period,
        "win_rate": float(stats.get("Win Rate [%]", 0)),
        "sharpe": float(stats.get("Sharpe Ratio", 0)),
        "max_drawdown": float(stats.get("Max Drawdown [%]", 0)),
        "total_return": float(stats.get("Total Return [%]", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest technical signals")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", default="2y")
    args = parser.parse_args()

    result = asyncio.run(run_backtest(args.ticker, args.period))
    if result.get("error"):
        print(f"Backtest failed: {result['error']}")
        raise SystemExit(1)

    print(f"Ticker: {result['ticker']}")
    print(f"Period: {result['period']}")
    print(f"Win rate: {result['win_rate']:.2f}%")
    print(f"Sharpe: {result['sharpe']:.2f}")
    print(f"Max drawdown: {result['max_drawdown']:.2f}%")
    print(f"Total return: {result['total_return']:.2f}%")


if __name__ == "__main__":
    main()
