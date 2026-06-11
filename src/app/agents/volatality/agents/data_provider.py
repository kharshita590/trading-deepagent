from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import talib
from scipy import stats

from app.core.yfinance_utils import fetch_history
from ..config.settings import logger


class VolatilityLiquidityDataProvider:
    @staticmethod
    async def get_extended_price_data(ticker: str) -> Dict:
        try:
            data_1y = await fetch_history(ticker, period="1y")
            data_3m = await fetch_history(ticker, period="3mo")
            data_1m = await fetch_history(ticker, period="1mo")
            data_5d = await fetch_history(ticker, period="5d", interval="1h")
            return {
                "data_1y": data_1y,
                "data_3m": data_3m,
                "data_1m": data_1m,
                "data_5d": data_5d,
                "info": {},
                "ticker": ticker,
            }
        except Exception as e:
            logger.error("Error fetching extended data for %s: %s", ticker, e)
            return {"error": str(e), "ticker": ticker}

    @staticmethod
    def calculate_volatility_metrics(data_dict: Dict) -> Dict:
        try:
            metrics = {}
            data_1y = data_dict.get("data_1y", pd.DataFrame())
            data_3m = data_dict.get("data_3m", pd.DataFrame())
            data_1m = data_dict.get("data_1m", pd.DataFrame())
            info = data_dict.get("info", {})
            if data_1y.empty:
                return {"error": "No price data available for volatility calculation"}
            returns_1y = data_1y["Close"].pct_change().dropna()
            returns_3m = data_3m["Close"].pct_change().dropna() if not data_3m.empty else returns_1y[-63:]
            returns_1m = data_1m["Close"].pct_change().dropna() if not data_1m.empty else returns_1y[-22:]
            positive_returns = returns_1y[returns_1y > 0]
            negative_returns = returns_1y[returns_1y < 0]
            metrics["historical_vol_1y"] = returns_1y.std() * np.sqrt(252) * 100
            metrics["historical_vol_3m"] = returns_3m.std() * np.sqrt(252) * 100
            metrics["historical_vol_1m"] = returns_1m.std() * np.sqrt(252) * 100
            metrics["historical_win_rate"] = float((returns_1y > 0).mean()) if len(returns_1y) else 0.0
            metrics["avg_win"] = float(positive_returns.mean()) if not positive_returns.empty else 0.0
            metrics["avg_loss"] = float(abs(negative_returns.mean())) if not negative_returns.empty else 0.0
            rolling_vol_30d = returns_1y.rolling(window=30).std() * np.sqrt(252) * 100
            metrics["current_30d_vol"] = rolling_vol_30d.iloc[-1] if not rolling_vol_30d.empty else 0
            metrics["avg_30d_vol"] = rolling_vol_30d.mean() if not rolling_vol_30d.empty else 0
            metrics["max_30d_vol"] = rolling_vol_30d.max() if not rolling_vol_30d.empty else 0
            metrics["min_30d_vol"] = rolling_vol_30d.min() if not rolling_vol_30d.empty else 0
            if len(data_1y) > 14:
                atr = talib.ATR(data_1y["High"].values, data_1y["Low"].values, data_1y["Close"].values, timeperiod=14)
                metrics["atr_14"] = atr[-1] if not np.isnan(atr[-1]) else 0
                metrics["atr_percentage"] = (atr[-1] / data_1y["Close"].iloc[-1]) * 100 if atr[-1] > 0 else 0
            current_price = data_1y["Close"].iloc[-1]
            high_52w = data_1y["High"].max()
            low_52w = data_1y["Low"].min()
            metrics["price_range_52w"] = ((high_52w - low_52w) / low_52w) * 100
            metrics["current_vs_52w_high"] = ((current_price - high_52w) / high_52w) * 100
            metrics["current_vs_52w_low"] = ((current_price - low_52w) / low_52w) * 100
            metrics["volatility_percentile"] = stats.percentileofscore(rolling_vol_30d.dropna(), metrics["current_30d_vol"]) if len(rolling_vol_30d) > 1 else 50
            metrics["beta"] = info.get("beta", 1.0)
            if metrics["historical_vol_1y"] < 15:
                metrics["volatility_class"] = "Low"
            elif metrics["historical_vol_1y"] < 30:
                metrics["volatility_class"] = "Moderate"
            elif metrics["historical_vol_1y"] < 50:
                metrics["volatility_class"] = "High"
            else:
                metrics["volatility_class"] = "Very High"
            recent_vol = metrics["current_30d_vol"]
            avg_vol = metrics["avg_30d_vol"]
            metrics["volatility_trend"] = "Increasing" if recent_vol > avg_vol * 1.2 else "Decreasing" if recent_vol < avg_vol * 0.8 else "Stable"
            return metrics
        except Exception as e:
            logger.error("Error calculating volatility metrics: %s", e)
            return {"error": str(e)}

    @staticmethod
    def calculate_liquidity_metrics(data_dict: Dict) -> Dict:
        try:
            metrics = {}
            data_1y = data_dict.get("data_1y", pd.DataFrame())
            data_3m = data_dict.get("data_3m", pd.DataFrame())
            data_1m = data_dict.get("data_1m", pd.DataFrame())
            data_5d = data_dict.get("data_5d", pd.DataFrame())
            info = data_dict.get("info", {})
            if data_1y.empty:
                return {"error": "No data available for liquidity calculation"}
            volume_1y = data_1y["Volume"]
            metrics["avg_volume_1y"] = volume_1y.mean()
            metrics["avg_volume_3m"] = data_3m["Volume"].mean() if not data_3m.empty else volume_1y[-63:].mean()
            metrics["avg_volume_1m"] = data_1m["Volume"].mean() if not data_1m.empty else volume_1y[-22:].mean()
            current_volume = volume_1y.iloc[-1]
            metrics["current_volume"] = current_volume
            metrics["volume_ratio_3m"] = current_volume / metrics["avg_volume_3m"] if metrics["avg_volume_3m"] > 0 else 1
            metrics["volume_ratio_1m"] = current_volume / metrics["avg_volume_1m"] if metrics["avg_volume_1m"] > 0 else 1
            recent_avg_volume = volume_1y[-10:].mean()
            older_avg_volume = volume_1y[-30:-10].mean()
            metrics["volume_trend"] = "Increasing" if recent_avg_volume > older_avg_volume * 1.15 else "Decreasing" if recent_avg_volume < older_avg_volume * 0.85 else "Stable"
            market_cap = info.get("marketCap", 0)
            shares_outstanding = info.get("sharesOutstanding", 0)
            float_shares = info.get("floatShares", shares_outstanding)
            metrics["market_cap"] = market_cap
            metrics["shares_outstanding"] = shares_outstanding
            metrics["float_shares"] = float_shares
            metrics["float_percentage"] = (float_shares / shares_outstanding * 100) if shares_outstanding > 0 else 100
            current_price = data_1y["Close"].iloc[-1]
            metrics["avg_dollar_volume_1y"] = (volume_1y * data_1y["Close"]).mean()
            metrics["avg_dollar_volume_3m"] = (data_3m["Volume"] * data_3m["Close"]).mean() if not data_3m.empty else (volume_1y[-63:] * data_1y["Close"][-63:]).mean()
            metrics["current_dollar_volume"] = current_volume * current_price
            daily_spreads = ((data_1y["High"] - data_1y["Low"]) / data_1y["Close"]) * 100
            metrics["avg_daily_spread"] = daily_spreads.mean()
            metrics["current_daily_spread"] = daily_spreads.iloc[-1]
            metrics["daily_turnover"] = (current_volume / float_shares) * 100 if float_shares > 0 else 0
            metrics["avg_daily_turnover"] = (metrics["avg_volume_1y"] / float_shares) * 100 if float_shares > 0 else 0
            if not data_1m.empty:
                vwap_1m = (data_1m["Close"] * data_1m["Volume"]).sum() / data_1m["Volume"].sum()
                metrics["price_vs_vwap_1m"] = ((current_price - vwap_1m) / vwap_1m) * 100
            else:
                metrics["price_vs_vwap_1m"] = 0
            if not data_5d.empty and len(data_5d) > 20:
                hourly_volumes = data_5d["Volume"]
                hourly_ranges = ((data_5d["High"] - data_5d["Low"]) / data_5d["Close"]) * 100
                metrics["intraday_vol_consistency"] = 1 - (hourly_volumes.std() / hourly_volumes.mean()) if hourly_volumes.mean() > 0 else 0
                metrics["intraday_spread_avg"] = hourly_ranges.mean()
            else:
                metrics["intraday_vol_consistency"] = 0.5
                metrics["intraday_spread_avg"] = metrics["avg_daily_spread"] / 4
            if metrics["avg_dollar_volume_1y"] > 100000000:
                metrics["liquidity_class"] = "Very High"
            elif metrics["avg_dollar_volume_1y"] > 50000000:
                metrics["liquidity_class"] = "High"
            elif metrics["avg_dollar_volume_1y"] > 10000000:
                metrics["liquidity_class"] = "Moderate"
            elif metrics["avg_dollar_volume_1y"] > 1000000:
                metrics["liquidity_class"] = "Low"
            else:
                metrics["liquidity_class"] = "Very Low"
            metrics["estimated_impact_cost"] = min(1000000 / (market_cap / 1000000000 * metrics["avg_volume_1y"] / 1000000) * 0.1, 5.0) if market_cap > 0 and metrics["avg_volume_1y"] > 0 else 2.0
            return metrics
        except Exception as e:
            logger.error("Error calculating liquidity metrics: %s", e)
            return {"error": str(e)}

    @staticmethod
    def calculate_risk_metrics(volatility_metrics: Dict, liquidity_metrics: Dict, allocation_amount: float) -> Dict:
        try:
            risk_metrics = {}
            daily_vol = volatility_metrics.get("historical_vol_1y", 20) / np.sqrt(252) / 100
            var_95_1d = allocation_amount * 1.645 * daily_vol
            var_99_1d = allocation_amount * 2.326 * daily_vol
            risk_metrics["var_95_1day"] = var_95_1d
            risk_metrics["var_99_1day"] = var_99_1d
            risk_metrics["var_95_1day_pct"] = (var_95_1d / allocation_amount) * 100
            atr_pct = volatility_metrics.get("atr_percentage", 2)
            max_daily_loss = allocation_amount * (atr_pct / 100) * 2
            risk_metrics["max_expected_daily_loss"] = max_daily_loss
            risk_metrics["max_expected_daily_loss_pct"] = (max_daily_loss / allocation_amount) * 100
            avg_dollar_volume = liquidity_metrics.get("avg_dollar_volume_1y", 1000000)
            position_vs_volume = (allocation_amount / avg_dollar_volume) * 100
            risk_metrics["position_vs_avg_volume_pct"] = position_vs_volume
            if position_vs_volume > 20:
                risk_metrics["liquidity_risk"] = "Very High"
                risk_metrics["liquidity_risk_score"] = 90
            elif position_vs_volume > 10:
                risk_metrics["liquidity_risk"] = "High"
                risk_metrics["liquidity_risk_score"] = 75
            elif position_vs_volume > 5:
                risk_metrics["liquidity_risk"] = "Moderate"
                risk_metrics["liquidity_risk_score"] = 50
            elif position_vs_volume > 1:
                risk_metrics["liquidity_risk"] = "Low"
                risk_metrics["liquidity_risk_score"] = 25
            else:
                risk_metrics["liquidity_risk"] = "Very Low"
                risk_metrics["liquidity_risk_score"] = 10
            vol_score = min(volatility_metrics.get("historical_vol_1y", 20) * 2, 100)
            liq_score = risk_metrics["liquidity_risk_score"]
            risk_metrics["combined_risk_score"] = (vol_score * 0.6 + liq_score * 0.4)
            if risk_metrics["combined_risk_score"] > 75:
                risk_metrics["overall_risk_level"] = "Very High"
            elif risk_metrics["combined_risk_score"] > 60:
                risk_metrics["overall_risk_level"] = "High"
            elif risk_metrics["combined_risk_score"] > 40:
                risk_metrics["overall_risk_level"] = "Moderate"
            elif risk_metrics["combined_risk_score"] > 25:
                risk_metrics["overall_risk_level"] = "Low"
            else:
                risk_metrics["overall_risk_level"] = "Very Low"
            risk_metrics["position_size_adjustment"] = 0.5 if risk_metrics["combined_risk_score"] > 75 else 0.7 if risk_metrics["combined_risk_score"] > 60 else 0.85 if risk_metrics["combined_risk_score"] > 40 else 1.0
            return risk_metrics
        except Exception as e:
            logger.error("Error calculating risk metrics: %s", e)
            return {"error": str(e)}


class DataFetchAgent:
    def __init__(self):
        self.data_provider = VolatilityLiquidityDataProvider()

    async def execute(self, state: Dict) -> Dict:
        logger.info("Fetching volatility and liquidity data")
        recommendations = state.get("recommendations", [])
        vol_liq_data = {}
        for rec in recommendations:
            ticker = rec.get("ticker", "")
            allocation_amount = rec.get("allocation_amount", 0)
            data_dict = await self.data_provider.get_extended_price_data(ticker)
            if "error" not in data_dict:
                volatility_metrics = self.data_provider.calculate_volatility_metrics(data_dict)
                liquidity_metrics = self.data_provider.calculate_liquidity_metrics(data_dict)
                risk_metrics = self.data_provider.calculate_risk_metrics(volatility_metrics, liquidity_metrics, allocation_amount)
                vol_liq_data[ticker] = {
                    "recommendation": rec,
                    "volatility_metrics": volatility_metrics,
                    "liquidity_metrics": liquidity_metrics,
                    "risk_metrics": risk_metrics,
                }
            else:
                logger.warning("No data available for %s", ticker)
                vol_liq_data[ticker] = {"error": data_dict.get("error", "Unknown error")}
        state["vol_liq_data"] = vol_liq_data
        state["messages"].append(AIMessage(content=f"Fetched volatility/liquidity data for {len(recommendations)} stocks"))
        return state
