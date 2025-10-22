
from .base_agents import BaseAgent
from ..models.types import InvestmentAllocationState, RiskLevel

class RiskAssessmentAgent(BaseAgent):    
    async def execute(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        self.log_message(state, "Assessing risk factors")
        
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
        
        message = f"Risk: {risk_tolerance.value}, Horizon: {investment_horizon}, Single stock OK: {single_stock_acceptable}"
        self.log_message(state, message)
        
        return state
