import logging
from typing import Dict
from ..models.types import PortfolioState
from ..investement_allocation_system.models.types import RiskLevel
logger = logging.getLogger(__name__)
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
    async def run_investment_allocation(self, state: PortfolioState) -> Dict:
        logger.info("Running investment allocation")
        try:
            allocation_decision = await self.investment_orchestrator.run_allocation(
                investment_amount=state.investment_amount,
                risk_tolerance=RiskLevel.MODERATE,
                investment_horizon=state.investment_horizon,
                user_preferences=state.user_preferences
            )
            logger.info(f"Allocation completed: {allocation_decision.strategy.value}")
            return {"allocation_decision": allocation_decision}
        
        except Exception as e:
            logger.error(f"Investment allocation failed: {e}")
            return {"errors": {"investment_allocation": str(e)}}
    
    async def run_research(self, state: PortfolioState) -> Dict:
        logger.info("Running Research Agent")
        try:
            recommendations = await self.research_orchestrator.generate_recommendations(
                allocation_strategy=state.allocation_decision.strategy,
                investment_amount=state.investment_amount,
                preferred_sectors=state.user_preferences,
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
            return {"recommendations": recommendations_dict}
        
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {"errors": {"research": str(e)}}
    
    async def run_fundamental_analysis(self, state: PortfolioState) -> Dict:
        logger.info("Running Fundamental Analysis (Parallel)")
        
        try:
            result = await self.fundamental_orchestrator.run(state.recommendations)
            logger.info("Fundamental analysis completed")
            return {"fundamental_analysis": result.get("fundamental_analysis")}
        
        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            return {"errors": {"fundamental": str(e)}}
    
    async def run_macro_analysis(self, state: PortfolioState) -> Dict:
        logger.info("Running Macro Analysis (Parallel)")
        
        try:
            macro_result = await self.macro_orchestrator.analyze(
                recommendations=state.recommendations,
                portfolio_amount=state.investment_amount
            )
            logger.info("Macro analysis completed")
            return {"macro_analysis": macro_result}
        
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return {"errors": {"macro": str(e)}}
    
    async def run_technical_analysis(self, state: PortfolioState) -> Dict:
        logger.info("Running Technical Analysis (Parallel)")
        try:
            technical_result = await self.technical_orchestrator.analyze(
                recommendations=state.recommendations
            )
            logger.info("Technical analysis completed")
            return {"technical_analysis": technical_result}
        
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"errors": {"technical": str(e)}}
    
    async def run_volatility_liquidity(self, state: PortfolioState) -> Dict:
        logger.info("Running Volatility/Liquidity Analysis (Parallel)")
        
        try:
            vol_liq_result = await self.volatility_orchestrator.run(
                recommendations=state.recommendations
            )
            logger.info("Volatility/Liquidity analysis completed")
            return {"volatility_liquidity_analysis": vol_liq_result.get("volatility_liquidity_analysis")}
        
        except Exception as e:
            logger.error(f"Volatility/Liquidity analysis failed: {e}")
            return {"errors": {"volatility_liquidity": str(e)}}
    
    async def run_behavioral_psychology(self, state: PortfolioState) -> Dict:
        logger.info("Running Behavioral Psychology Analysis (Parallel)")
        
        try:
            behavioral_state = {
                "investment_amount": state.investment_amount,
                "selected_stocks": [
                    {
                        "symbol": rec["ticker"],
                        "current_price": rec["price"],
                        "volatility": 0.25,
                        "historical_win_rate": 0.60,
                        "avg_win": 0.10,
                        "avg_loss": 0.05
                    }
                    for rec in state.recommendations
                ],
                "technical_analysis": {},
                "volatility_data": {},
                "fundamental_data": {}
            }
            
            behavioral_result = await self.behavioral_orchestrator.run(behavioral_state)
            logger.info("Behavioral psychology analysis completed")
            return {"behavioral_psychology_analysis": behavioral_result}
        
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
        
        return {}
    
    async def run_risk_management(self, state: PortfolioState) -> Dict:
        logger.info("Running Risk Management")
        
        try:
            risk_state = {
                "investment_amount": state.investment_amount,
                "selected_stocks": [
                    {
                        "symbol": rec["ticker"],
                        "current_price": rec["price"],
                        "volatility": 0.25,
                        "sector": rec["sector"],
                        "weight": rec["allocation_percentage"] / 100.0,
                        "technical_levels": {}
                    }
                    for rec in state.recommendations
                ],
                "macro_analysis": {"market_sentiment": "neutral"},
                "behavioral_biases": state.behavioral_psychology_analysis.get("behavioral_biases", []) if state.behavioral_psychology_analysis else [],
                "position_sizing": state.behavioral_psychology_analysis.get("position_sizing", {}) if state.behavioral_psychology_analysis else {},
                "fundamental_data": state.fundamental_analysis or {},
                "technical_analysis": state.technical_analysis or {},
                "volatility_data": state.volatility_liquidity_analysis or {}
            }
            
            risk_result = await self.risk_orchestrator.run(risk_state)
            logger.info("Risk management completed")
            
            return {"risk_management_result": risk_result}
        
        except Exception as e:
            logger.error(f"Risk management failed: {e}")
            return {"errors": {"risk_management": str(e)}}

