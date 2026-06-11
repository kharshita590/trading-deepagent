import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Optional

from ..models.types import PortfolioState
from ..investment_allocation_system.models.types import RiskLevel

logger = logging.getLogger(__name__)


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _risk_level_from_string(risk_tolerance: str) -> RiskLevel:
    normalized = (risk_tolerance or "").strip().lower()
    if normalized == "low":
        return RiskLevel.LOW
    if normalized == "high":
        return RiskLevel.HIGH
    return RiskLevel.MODERATE


def _extract_market_sentiment(macro_analysis: Any, macro_data: Any) -> Optional[str]:
    macro_analysis_dict = _to_dict(macro_analysis)
    if macro_analysis_dict.get("market_sentiment"):
        return macro_analysis_dict.get("market_sentiment")
    macro_data_dict = _to_dict(macro_data)
    if macro_data_dict.get("market_sentiment"):
        return macro_data_dict.get("market_sentiment")
    return macro_data_dict.get("economic_indicators", {}).get("market_sentiment")


def _extract_signals(entry: Any) -> Dict[str, Any]:
    entry_dict = _to_dict(entry)
    if "overall_signal" in entry_dict and "key_levels" in entry_dict:
        return entry_dict
    signals = entry_dict.get("signals", {})
    return _to_dict(signals)


class WorkflowNodes:    
    def __init__(self, orchestrators: Dict):
        self.investment_orchestrator = orchestrators['investment']
        self.research_orchestrator = orchestrators['research']
        self.fundamental_orchestrator = orchestrators['fundamental']
        self.macro_orchestrator = orchestrators['macro']
        self.technical_orchestrator = orchestrators['technical']
        self.volatility_orchestrator = orchestrators['volatility']
        self.behavioral_orchestrator = orchestrators['behavioral']
        self.risk_orchestrator = orchestrators['risk']

    def _timings(self, state: PortfolioState) -> Dict[str, float]:
        return dict(getattr(state, "analysis_timings", {}) or {})
    async def run_investment_allocation(self, state: PortfolioState) -> Dict:
        logger.info("Running investment allocation")
        start = time.perf_counter()
        try:
            allocation_decision = await self.investment_orchestrator.run_allocation(
                investment_amount=state.investment_amount,
                risk_tolerance=_risk_level_from_string(state.risk_tolerance),
                investment_horizon=state.investment_horizon,
                user_preferences=state.user_preferences
            )
            logger.info(f"Allocation completed: {allocation_decision.strategy.value}")
            elapsed = round(time.perf_counter() - start, 3)
            logger.info("investment_allocation completed in %.3fs", elapsed)
            timings = self._timings(state)
            timings["investment_allocation"] = elapsed
            return {"allocation_decision": allocation_decision, "analysis_timings": timings}
        
        except Exception as e:
            logger.error(f"Investment allocation failed: {e}")
            return {"errors": {"investment_allocation": str(e)}}
    
    async def run_research(self, state: PortfolioState) -> Dict:
        logger.info("Running Research Agent")
        start = time.perf_counter()
        try:
            preferences = state.user_preferences or {}
            recommendations = await self.research_orchestrator.generate_recommendations(
                allocation_strategy=state.allocation_decision.strategy,
                investment_amount=state.investment_amount,
                preferred_sectors=preferences.get("preferred_sectors", []),
                risk_score=state.allocation_decision.risk_score,
                diversification_score=state.allocation_decision.diversification_score
            )
            recommendations_dict = [
                {
                    "ticker": rec.ticker,
                    "company_name": rec.company_name,
                    "sector": rec.sector,
                    "price": rec.price,
                    "allocation_percentage": rec.allocation_percentage,
                    "allocation_amount": rec.allocation_amount,
                    "reasoning": rec.reasoning
                }
                for rec in recommendations
            ]
            
            logger.info(f"Research completed: {len(recommendations_dict)} recommendations")
            elapsed = round(time.perf_counter() - start, 3)
            logger.info("research completed in %.3fs", elapsed)
            timings = self._timings(state)
            timings["research"] = elapsed
            return {"recommendations": recommendations_dict, "analysis_timings": timings}
        
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {"errors": {"research": str(e)}}
    
    async def run_fundamental_analysis(self, state: PortfolioState) -> Dict:
        logger.info("Running Fundamental Analysis (Parallel)")
        start = time.perf_counter()
        
        try:
            result = await self.fundamental_orchestrator.run(state.recommendations)
            logger.info("Fundamental analysis completed")
            return {
                "fundamental_analysis": result.get("fundamental_analysis"),
                "fundamental_data": result.get("financial_data"),
                "analysis_timings": {**self._timings(state), "fundamental": round(time.perf_counter() - start, 3)}
            }
        
        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            return {"errors": {"fundamental": str(e)}}
    
    async def run_macro_analysis(self, state: PortfolioState) -> Dict:
        logger.info("Running Macro Analysis (Parallel)")
        start = time.perf_counter()
        
        try:
            macro_result = await self.macro_orchestrator.analyze(
                recommendations=state.recommendations,
                portfolio_amount=state.investment_amount
            )
            logger.info("Macro analysis completed")
            return {
                "macro_analysis": macro_result,
                "macro_data": getattr(macro_result, "macro_data", None),
                "analysis_timings": {**self._timings(state), "macro": round(time.perf_counter() - start, 3)}
            }
        
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return {"errors": {"macro": str(e)}}
    
    async def run_technical_analysis(self, state: PortfolioState) -> Dict:
        logger.info("Running Technical Analysis (Parallel)")
        start = time.perf_counter()
        try:
            technical_result = await self.technical_orchestrator.analyze(
                recommendations=state.recommendations
            )
            logger.info("Technical analysis completed")
            return {
                "technical_analysis": technical_result,
                "technical_data": getattr(technical_result, "signal_by_ticker", None),
                "analysis_timings": {**self._timings(state), "technical": round(time.perf_counter() - start, 3)}
            }
        
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"errors": {"technical": str(e)}}
    
    async def run_volatility_liquidity(self, state: PortfolioState) -> Dict:
        logger.info("Running Volatility/Liquidity Analysis (Parallel)")
        start = time.perf_counter()
        
        try:
            vol_liq_result = await self.volatility_orchestrator.run(
                recommendations=state.recommendations
            )
            logger.info("Volatility/Liquidity analysis completed")
            return {
                "volatility_liquidity_analysis": vol_liq_result.get("volatility_liquidity_analysis"),
                "volatility_liquidity_data": vol_liq_result.get("vol_liq_data"),
                "analysis_timings": {**self._timings(state), "volatility": round(time.perf_counter() - start, 3)}
            }
        
        except Exception as e:
            logger.error(f"Volatility/Liquidity analysis failed: {e}")
            return {"errors": {"volatility_liquidity": str(e)}}
    
    async def run_behavioral_psychology(self, state: PortfolioState) -> Dict:
        logger.info("Running Behavioral Psychology Analysis (Parallel)")
        start = time.perf_counter()
        
        try:
            volatility_analysis = _to_dict(state.volatility_liquidity_analysis)
            fundamental_analysis = _to_dict(state.fundamental_analysis)
            technical_analysis = _to_dict(state.technical_analysis)
            volatility_data = _to_dict(state.volatility_liquidity_data) or _to_dict(volatility_analysis.get("stock_metrics"))
            fundamental_data = _to_dict(state.fundamental_data) or _to_dict(fundamental_analysis.get("stock_metrics"))
            technical_data = _to_dict(state.technical_data) or _to_dict(technical_analysis.get("signal_by_ticker"))
            behavioral_technical = _to_dict(state.technical_analysis)
            behavioral_fundamental = _to_dict(state.fundamental_analysis)

            selected_stocks = []
            for rec in state.recommendations:
                ticker = rec["ticker"]
                vol_entry = _to_dict(volatility_data.get(ticker)) or _to_dict(volatility_data.get("stock_metrics", {}).get(ticker))
                vol_metrics = _to_dict(vol_entry.get("volatility_metrics"))
                liq_metrics = _to_dict(vol_entry.get("liquidity_metrics"))
                fundamental_entry = _to_dict(fundamental_data.get(ticker)) or _to_dict(behavioral_fundamental.get("stock_metrics", {}).get(ticker))
                technical_entry = _to_dict(technical_data.get(ticker)) or _to_dict(behavioral_technical.get("signal_by_ticker", {}).get(ticker))
                signals = _extract_signals(technical_entry)
                key_levels = signals.get("key_levels", {}) or technical_entry.get("key_levels", {})
                historical_vol = vol_metrics.get("historical_vol_1y", 25.0)
                selected_stocks.append(
                    {
                        "symbol": ticker,
                        "current_price": rec["price"],
                        "volatility": float(historical_vol) / 100.0 if historical_vol else 0.25,
                        "historical_win_rate": float(vol_metrics.get("historical_win_rate", 0.5)),
                        "avg_win": float(vol_metrics.get("avg_win", 0.05)),
                        "avg_loss": float(vol_metrics.get("avg_loss", 0.03)),
                        "fundamental_score": float(_to_dict(fundamental_entry.get("financial_scores")).get("overall_fundamental_score", 0.0)),
                        "liquidity_class": liq_metrics.get("liquidity_class"),
                        "estimated_impact_cost": liq_metrics.get("estimated_impact_cost"),
                        "technical_levels": {
                            "key_levels": key_levels,
                            "overall_signal": signals.get("overall_signal"),
                            "strength": signals.get("strength"),
                            "bullish_signals": signals.get("bullish_signals", []),
                            "bearish_signals": signals.get("bearish_signals", [])
                        }
                    }
                )

            behavioral_state = {
                "investment_amount": state.investment_amount,
                "selected_stocks": selected_stocks,
                "technical_analysis": behavioral_technical,
                "fundamental_analysis": behavioral_fundamental,
                "volatility_data": volatility_data,
                "fundamental_data": fundamental_data
            }
            
            behavioral_result = await self.behavioral_orchestrator.run(behavioral_state)
            logger.info("Behavioral psychology analysis completed")
            return {
                "behavioral_psychology_analysis": behavioral_result,
                "behavioral_data": behavioral_state,
                "analysis_timings": {**self._timings(state), "behavioral": round(time.perf_counter() - start, 3)}
            }
        
        except Exception as e:
            logger.error(f"Behavioral psychology analysis failed: {e}")
            return {"errors": {"behavioral": str(e)}}
    
    async def join_parallel_analyses(self, state: PortfolioState) -> Dict:
        logger.info("JOINING PARALLEL ANALYSES")
        completed = []
        failed = []
        if state.fundamental_analysis:
            completed.append("Fundamental")
        if state.macro_analysis:
            completed.append("Macro")
        if state.technical_analysis:
            completed.append("Technical")
        if state.volatility_liquidity_analysis:
            completed.append("Volatility/Liquidity")
        if state.behavioral_psychology_analysis:
            completed.append("Behavioral")
        
        if state.errors:
            failed = list(state.errors.keys())
        
        logger.info(f"Completed analyses: {', '.join(completed)}")
        if failed:
            logger.warning(f"Failed analyses: {', '.join(failed)}")
        
        return {"analysis_timings": self._timings(state)}
    
    async def run_risk_management(self, state: PortfolioState) -> Dict:
        logger.info("Running Risk Management")
        start = time.perf_counter()
        
        try:
            volatility_analysis = _to_dict(state.volatility_liquidity_analysis)
            fundamental_analysis = _to_dict(state.fundamental_analysis)
            technical_analysis = _to_dict(state.technical_analysis)
            volatility_data = _to_dict(state.volatility_liquidity_data) or _to_dict(volatility_analysis.get("stock_metrics"))
            fundamental_data = _to_dict(state.fundamental_data) or _to_dict(fundamental_analysis.get("stock_metrics"))
            technical_data = _to_dict(state.technical_data) or _to_dict(technical_analysis.get("signal_by_ticker"))
            macro_sentiment = _extract_market_sentiment(state.macro_analysis, state.macro_data)
            macro_analysis = _to_dict(state.macro_analysis)

            risk_state = {
                "investment_amount": state.investment_amount,
                "selected_stocks": [
                    {
                        "symbol": rec["ticker"],
                        "current_price": rec["price"],
                        "volatility": float(
                            _to_dict(
                                _to_dict(volatility_data.get(rec["ticker"])).get("volatility_metrics", {})
                            ).get("historical_vol_1y", 25.0)
                        ) / 100.0,
                        "sector": rec["sector"],
                        "weight": rec["allocation_percentage"] / 100.0,
                        "technical_levels": _extract_signals(technical_data.get(rec["ticker"]))
                    }
                    for rec in state.recommendations
                ],
                "macro_analysis": {
                    **macro_analysis,
                    "market_sentiment": macro_sentiment or macro_analysis.get("market_sentiment", "neutral")
                },
                "behavioral_biases": state.behavioral_psychology_analysis.get("behavioral_biases", []) if state.behavioral_psychology_analysis else [],
                "position_sizing": state.behavioral_psychology_analysis.get("position_sizing", {}) if state.behavioral_psychology_analysis else {},
                "fundamental_data": fundamental_data,
                "technical_analysis": _to_dict(state.technical_analysis),
                "volatility_data": volatility_data
            }
            
            risk_result = await self.risk_orchestrator.run(risk_state)
            logger.info("Risk management completed")
            
            return {
                "risk_management_result": risk_result,
                "analysis_timings": {**self._timings(state), "risk": round(time.perf_counter() - start, 3)}
            }
        
        except Exception as e:
            logger.error(f"Risk management failed: {e}")
            return {"errors": {"risk_management": str(e)}}
