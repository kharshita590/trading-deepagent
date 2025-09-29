from typing import TypedDict, List, Dict, Optional, Literal
import json
from dataclasses import dataclass
from enum import Enum
import asyncio
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllocationStrategy(Enum):
    SINGLE_STOCK = "single_stock"
    MULTI_STOCK = "multi_stock"
    HYBRID = "hybrid"

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"


@dataclass
class AllocationDecision:
    strategy: AllocationStrategy
    total_amount: float
    risk_score: float
    diversification_score: float
    reasoning: str

class InvestmentAllocationState(TypedDict):
    investment_amount: float
    user_risk_tolerance: RiskLevel
    investment_horizon: str 
    user_preferences: Dict 

    market_conditions: Dict
    allocation_factors: Dict
    
    volatility_threshold: float
    diversification_requirement: bool
    sector_constraints: List[str]
    
    allocation_strategy: AllocationStrategy
    allocation_decision: Optional[AllocationDecision]
    
    messages: List
    next_agent: str

class InvestmentAllocationAgent:
    def __init__(self, llm_model="gemini-1.5-pro", indian_stock_tickers_file=None):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0,
        google_api_key="AIzaSyDl0-DuUoAmjs4hjM8E7TnRL7qazQ2Bq8w")
        self.indian_tickers_file = indian_stock_tickers_file
        
        self.allocation_factors = {
            "diversification_benefits": {
                "single_stock_risk": 0.8, 
                "multi_stock_protection": 0.6,  
                "optimal_diversification": 0.3   
            },
            "amount_thresholds": {
                "small_investment": 10000,  
                "medium_investment": 50000, 
                "large_investment": 100000 
            },
            "volatility_considerations": {
                "high_vol_sectors": ["tech", "crypto", "small_cap"],
                "stable_sectors": ["utilities", "consumer_staples", "healthcare"],
                "defensive_sectors": ["government_bonds", "dividend_stocks"]
            }
        }
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(InvestmentAllocationState)
        
        workflow.add_node("analyze_investment_amount", self.analyze_investment_amount)
        workflow.add_node("assess_risk_factors", self.assess_risk_factors)
        workflow.add_node("evaluate_market_conditions", self.evaluate_market_conditions)
        workflow.add_node("determine_allocation_strategy", self.determine_allocation_strategy)
        workflow.add_node("validate_allocation", self.validate_allocation)
        
        workflow.set_entry_point("analyze_investment_amount")
        workflow.add_edge("analyze_investment_amount", "assess_risk_factors")
        workflow.add_edge("assess_risk_factors", "evaluate_market_conditions")
        workflow.add_edge("evaluate_market_conditions", "determine_allocation_strategy")
        workflow.add_edge("determine_allocation_strategy", "validate_allocation")
        workflow.add_edge("validate_allocation", END)
        
        return workflow.compile()
    
    async def analyze_investment_amount(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        logger.info(f"Analyzing investment amount: ₹{state['investment_amount']}")
        
        amount = state['investment_amount']
        
        if amount < self.allocation_factors["amount_thresholds"]["small_investment"]:
            initial_strategy_lean = "single_stock"
            diversification_requirement = False
            message = f"Small investment amount (₹{amount}). Single stock strategy may be acceptable."
        elif amount < self.allocation_factors["amount_thresholds"]["medium_investment"]:
            initial_strategy_lean = "multi_stock"
            diversification_requirement = True
            message = f"Medium investment amount (₹{amount}). 2-5 stock diversification recommended."
        else:
            initial_strategy_lean = "multi_stock"
            diversification_requirement = True
            message = f"Large investment amount (₹{amount}). Strong diversification with 5+ stocks recommended."
        
        state["allocation_factors"] = {
            "initial_strategy_lean": initial_strategy_lean,
            "amount_category": "small" if amount < 10000 else "medium" if amount < 100000 else "large"
        }
        state["diversification_requirement"] = diversification_requirement
        state["messages"] = state.get("messages", []) + [AIMessage(content=message)]
        
        return state
    
    async def assess_risk_factors(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        logger.info("Assessing risk factors")
        
        risk_tolerance = state.get('user_risk_tolerance', RiskLevel.MODERATE)
        investment_horizon = state.get('investment_horizon', 'medium')
        
        volatility_thresholds = {
            RiskLevel.LOW: 0.15,     
            RiskLevel.MODERATE: 0.25,  
            RiskLevel.HIGH: 0.40      
        }
        risk_factors = {
            "volatility_threshold": volatility_thresholds[risk_tolerance],
            "time_horizon_factor": {
                "short": 0.8,  
                "medium": 1.0,  
                "long": 1.2    
            }.get(investment_horizon, 1.0),
            "risk_capacity": risk_tolerance.value
        }
        
        single_stock_acceptable = (
            risk_tolerance in [RiskLevel.MODERATE, RiskLevel.HIGH] and
            investment_horizon in ['medium', 'long'] and
            state['investment_amount'] < 25000
        )
        
        state["allocation_factors"]["risk_assessment"] = risk_factors
        state["allocation_factors"]["single_stock_acceptable"] = single_stock_acceptable
        state["volatility_threshold"] = volatility_thresholds[risk_tolerance]
        
        message = f"Risk assessment: {risk_tolerance.value} risk tolerance, {investment_horizon} horizon. Single stock acceptable: {single_stock_acceptable}"
        state["messages"].append(AIMessage(content=message))
        
        return state

    async def evaluate_market_conditions(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        logger.info("Evaluating market conditions (real data)")
        default_market_conditions = {
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
            if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
                raise RuntimeError("SPY history empty or invalid")

            if isinstance(hist.columns, pd.MultiIndex):
                try:
                    hist_close = hist.xs("Close", axis=1, level=-1).squeeze()
                except Exception:
                    hist_close = hist["Close"] if "Close" in hist.columns else None
            else:
                hist_close = hist["Close"]

            hist_returns = hist_close.pct_change()
            vol_series = hist_returns.rolling(21).std()
            if vol_series.dropna().empty:
                volatility = float("nan")
            else:
                volatility = float(vol_series.dropna().iloc[-1]) * np.sqrt(252)

            if np.isnan(volatility):
                market_volatility = "moderate"
            elif volatility < 0.15:
                market_volatility = "low"
            elif volatility < 0.25:
                market_volatility = "moderate"
            else:
                market_volatility = "high"

            sector_etfs = {
                "technology": "XLK",
                "healthcare": "XLV",
                "financial_services": "XLF",
                "consumer_goods": "XLP",
                "real_estate": "XLRE",
                "energy": "XLE",
                "industrials": "XLI"
            }

            sector_returns = {}
            series_list = []

            for sector, etf in sector_etfs.items():
                try:
                    data = yf.download(
                        etf,
                        period="1mo",
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        threads=False
                    )
                except Exception as e:
                    logger.warning(f"yf.download failed for {etf} (1mo): {e}", exc_info=True)
                    data = None

                if data is None or getattr(data, "empty", True):
                    logger.info(f"No price data for ETF {etf} ({sector}); skipping")
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        close = data.xs("Close", axis=1, level=-1).squeeze()
                    except Exception:
                        close = data["Close"] if "Close" in data.columns else None
                else:
                    close = data["Close"] if "Close" in data.columns else None

                if close is None or len(close.dropna()) < 2:
                    logger.info(f"Not enough Close data for {etf}; skipping")
                    continue

                try:
                    last_price = close.iloc[-1]
                    first_price = close.iloc[0]
                    if hasattr(last_price, 'iloc'):
                        last_price = float(last_price.iloc[0])
                    else:
                        last_price = float(last_price)
                        
                    if hasattr(first_price, 'iloc'):
                        first_price = float(first_price.iloc[0])
                    else:
                        first_price = float(first_price)
                    
                    ret = last_price / first_price - 1
                    sector_returns[sector] = ret
                except Exception as e:
                    logger.warning(f"Failed computing return for {etf}: {e}", exc_info=True)
                    continue

                try:
                    data3 = yf.download(
                        etf,
                        period="3mo",
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        threads=False
                    )
                except Exception as e:
                    logger.debug(f"yf.download failed for {etf} (3mo): {e}", exc_info=True)
                    data3 = None

                if data3 is None or getattr(data3, "empty", True):
                    continue

                if isinstance(data3.columns, pd.MultiIndex):
                    try:
                        close3 = data3.xs("Close", axis=1, level=-1).squeeze()
                    except Exception:
                        close3 = data3["Close"] if "Close" in data3.columns else None
                else:
                    close3 = data3["Close"] if "Close" in data3.columns else None

                if close3 is None or len(close3.dropna()) < 3:
                    continue

                try:
                    pct_changes = close3.pct_change().dropna()                    
                    if pct_changes.empty or len(pct_changes) == 0:
                        continue
                    if isinstance(pct_changes, pd.Series):
                        close_pct = pd.to_numeric(pct_changes, errors="coerce").dropna()
                    elif isinstance(pct_changes, pd.DataFrame):
                        close_pct = pct_changes.iloc[:, 0] if pct_changes.shape[1] == 1 else pct_changes.mean(axis=1)
                        close_pct = pd.to_numeric(close_pct, errors="coerce").dropna()
                    else:
                        if hasattr(pct_changes, 'shape') and len(pct_changes.shape) == 2 and pct_changes.shape[1] == 1:
                            pct_changes = pct_changes.flatten()
                        close_pct = pd.Series(pct_changes, index=close3.pct_change().dropna().index if hasattr(close3.pct_change().dropna(), 'index') else None)
                        close_pct = pd.to_numeric(close_pct, errors="coerce").dropna()
                    if close_pct.empty:
                        continue
                    close_pct.name = sector
                    series_list.append(close_pct)
                    
                except Exception as e:
                    logger.warning(f"Failed processing 3mo data for {etf}: {e}", exc_info=True)
                    continue
            if sector_returns:
                top_sectors = sorted(sector_returns, key=sector_returns.get, reverse=True)[:4]
                bottom_sectors = sorted(sector_returns, key=sector_returns.get)[:2]
                if len(sector_returns) >= 2:
                    try:
                        sector_rotation = (max(sector_returns.values()) - min(sector_returns.values())) > 0.05
                    except Exception:
                        sector_rotation = False
                else:
                    sector_rotation = False
            else:
                logger.warning("No sector returns computed; using fallback sectors")
                top_sectors = default_market_conditions["recommended_sectors"]
                bottom_sectors = default_market_conditions["sectors_to_avoid"]
                sector_rotation = False

            if series_list:
                try:
                    sector_hist = pd.concat(series_list, axis=1)
                    rows_with_two = sector_hist.dropna(thresh=2)
                    if rows_with_two.empty or sector_hist.shape[1] < 2:
                        avg_corr = 0.5
                    else:
                        corr_matrix = rows_with_two.corr()
                        avg_corr = float(corr_matrix.mean().mean())
                except Exception as e:
                    logger.exception(f"Failed computing sector correlation: {e} — using default avg_corr=0.5")
                    avg_corr = 0.5
            else:
                avg_corr = 0.5

            correlation_levels = "low" if avg_corr < 0.3 else "moderate" if avg_corr < 0.6 else "high"

            try:
                avg_volume = float(hist["Volume"].mean()) if "Volume" in hist.columns else 0
            except Exception:
                avg_volume = 0
            liquidity_conditions = "good" if (avg_volume and avg_volume > 50_000_000) else "poor"

            market_conditions = {
                "market_volatility": market_volatility,
                "sector_rotation": sector_rotation,
                "correlation_levels": correlation_levels,
                "liquidity_conditions": liquidity_conditions,
                "recommended_sectors": top_sectors,
                "sectors_to_avoid": bottom_sectors
            }

            if market_volatility == "high":
                state["allocation_factors"]["volatility_adjustment"] = -0.2
            elif market_volatility == "low":
                state["allocation_factors"]["volatility_adjustment"] = 0.1
            else:
                state["allocation_factors"]["volatility_adjustment"] = 0.0

            state["market_conditions"] = market_conditions
            state["sector_constraints"] = market_conditions["sectors_to_avoid"]

            message = (
                f"Market conditions: {market_conditions['market_volatility']} volatility, "
                f"sector rotation: {market_conditions['sector_rotation']}, "
                f"top sectors: {market_conditions['recommended_sectors']}"
            )
            state["messages"].append(AIMessage(content=message))

        except Exception as e:
            logger.exception(f"Error fetching market conditions (outer): {e} — using defaults")
            state["allocation_factors"]["volatility_adjustment"] = 0.0
            state["market_conditions"] = default_market_conditions
            state["sector_constraints"] = default_market_conditions["sectors_to_avoid"]
            state["messages"].append(AIMessage(content="Market conditions fetch failed; using defaults"))

        return state

    async def determine_allocation_strategy(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        logger.info("Determining allocation strategy")
        
        factors = state["allocation_factors"]
        amount = state['investment_amount']
        
        single_stock_score = 0
        multi_stock_score = 0
        
        if amount < 15000:
            single_stock_score += 2
        elif amount < 50000:
            multi_stock_score += 1
        else:
            multi_stock_score += 3
        
        if factors["risk_assessment"]["risk_capacity"] == "high":
            single_stock_score += 1
        else:
            multi_stock_score += 2
        
        if state["market_conditions"]["market_volatility"] == "high":
            multi_stock_score += 2
        elif state["market_conditions"]["market_volatility"] == "low":
            single_stock_score += 1
        
        if state["diversification_requirement"]:
            multi_stock_score += 3
        
        if single_stock_score > multi_stock_score:
            strategy = AllocationStrategy.SINGLE_STOCK
            target_stocks = 1
        elif multi_stock_score > single_stock_score + 1:
            strategy = AllocationStrategy.MULTI_STOCK
            if amount < 50000:
                target_stocks = min(3, max(2, amount // 15000))
            else:
                target_stocks = min(8, max(4, amount // 25000))
        else:
            strategy = AllocationStrategy.HYBRID
            target_stocks = 2
        
        state["allocation_strategy"] = strategy
        state["allocation_factors"]["target_stocks"] = target_stocks
        state["allocation_factors"]["decision_scores"] = {
            "single_stock": single_stock_score,
            "multi_stock": multi_stock_score
        }
        
        message = f"Strategy decision: {strategy.value} with {target_stocks} target stocks (scores: single={single_stock_score}, multi={multi_stock_score})"
        state["messages"].append(AIMessage(content=message))
        
        return state
    
    async def validate_allocation(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        logger.info("Validating allocation decision")
        strategy=state["allocation_strategy"]
        total_allocated=state["investment_amount"]
        risk_score = 0.0
        if strategy == AllocationStrategy.SINGLE_STOCK:
            risk_score = 0.8  
        else:
            risk_score = 0.5
        diversification_score = 0.2 if strategy == AllocationStrategy.SINGLE_STOCK else 0.7 
        
        decision = AllocationDecision(
            strategy=strategy,
            total_amount=total_allocated,
            risk_score=risk_score,
            diversification_score=diversification_score,
            reasoning=f"{strategy.value} strategy with stocks across sectors. Risk score: {risk_score:.2f}, Diversification: {diversification_score:.2f}"
        )
        
        state["allocation_decision"] = decision
        state["next_agent"] = "research_phase_agent" 
        
        validation_msg = f"Allocation validated: ₹{total_allocated:.2f} allocated across stocks. Risk: {risk_score:.2f}"
        state["messages"].append(AIMessage(content=validation_msg))
        
        return state
    
    async def run_allocation(self, investment_amount: float, risk_tolerance: RiskLevel = RiskLevel.MODERATE, 
                           investment_horizon: str = "medium", user_preferences: Dict = None) -> AllocationDecision:
        
        initial_state = InvestmentAllocationState(
            investment_amount=investment_amount,
            user_risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon,
            user_preferences=user_preferences or {},
            market_conditions={},
            allocation_factors={},
            volatility_threshold=0.25,
            diversification_requirement=False,
            sector_constraints=[],
            allocation_strategy=AllocationStrategy.MULTI_STOCK,
            allocation_decision=None,
            messages=[],
            next_agent=""
        )
        
        workflow = self.create_workflow()
        result = await workflow.ainvoke(initial_state)
        
        return result["allocation_decision"]
