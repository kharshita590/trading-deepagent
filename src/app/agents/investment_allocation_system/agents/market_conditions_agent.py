import yfinance as yf
import pandas as pd
import numpy as np
from .base_agents import BaseAgent
from ..models.types import InvestmentAllocationState
import logging

logger = logging.getLogger(__name__)

class MarketConditionsAgent(BaseAgent):    
    async def execute(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        self.log_message(state, "Evaluating market conditions")
        
        default_conditions = {
            "market_volatility": "moderate",
            "sector_rotation": False,
            "correlation_levels": "moderate",
            "liquidity_conditions": "good",
            "recommended_sectors": ["technology", "healthcare"],
            "sectors_to_avoid": []
        }
        
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="6mo", interval="1d")
            
            if hist is None or hist.empty:
                raise RuntimeError("SPY history unavailable")
            
            volatility = self._calculate_volatility(hist)
            market_volatility = self._categorize_volatility(volatility)
            
            sector_data = self._analyze_sectors()
            
            market_conditions = {
                "market_volatility": market_volatility,
                "sector_rotation": sector_data.get("sector_rotation", False),
                "correlation_levels": sector_data.get("correlation_levels", "moderate"),
                "liquidity_conditions": "good" if hist["Volume"].mean() > 50_000_000 else "poor",
                "recommended_sectors": sector_data.get("top_sectors", ["technology", "healthcare"]),
                "sectors_to_avoid": sector_data.get("bottom_sectors", [])
            }
            
            volatility_adj = {"high": -0.2, "low": 0.1, "moderate": 0.0}
            state["allocation_factors"]["volatility_adjustment"] = volatility_adj.get(market_volatility, 0.0)
            
            state["market_conditions"] = market_conditions
            state["sector_constraints"] = market_conditions["sectors_to_avoid"]
            
            message = f"Market: {market_volatility} vol, Top sectors: {market_conditions['recommended_sectors']}"
            self.log_message(state, message)
            
        except Exception as e:
            logger.exception(f"Market analysis failed: {e}")
            state["market_conditions"] = default_conditions
            state["allocation_factors"]["volatility_adjustment"] = 0.0
            self.log_message(state, "Using default market conditions")
        
        return state
    
    def _calculate_volatility(self, hist):
        if isinstance(hist.columns, pd.MultiIndex):
            close = hist.xs("Close", axis=1, level=-1).squeeze()
        else:
            close = hist["Close"]
        
        returns = close.pct_change()
        vol_series = returns.rolling(21).std()
        return float(vol_series.dropna().iloc[-1]) * np.sqrt(252) if not vol_series.dropna().empty else 0.25
    
    def _categorize_volatility(self, volatility):
        if volatility < 0.15:
            return "low"
        elif volatility < 0.25:
            return "moderate"
        else:
            return "high"
    
    def _analyze_sectors(self):
        sector_etfs = {
            "technology": "XLK",
            "healthcare": "XLV",
            "financial_services": "XLF",
            "consumer_goods": "XLP",
        }
        
        sector_returns = {}
        
        for sector, etf in sector_etfs.items():
            try:
                data = yf.download(etf, period="1mo", progress=False)
                if data is not None and not data.empty:
                    close = data["Close"] if "Close" in data.columns else None
                    if close is not None and len(close) > 1:
                        ret = float(close.iloc[-1] / close.iloc[0] - 1)
                        sector_returns[sector] = ret
            except Exception as e:
                logger.debug(f"Failed to fetch {etf}: {e}")
        
        if sector_returns:
            sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
            return {
                "top_sectors": [s[0] for s in sorted_sectors[:2]],
                "bottom_sectors": [s[0] for s in sorted_sectors[-1:]],
                "sector_rotation": (max(sector_returns.values()) - min(sector_returns.values())) > 0.05,
                "correlation_levels": "moderate"
            }
        
        return {"top_sectors": ["technology"], "bottom_sectors": [], "sector_rotation": False}
