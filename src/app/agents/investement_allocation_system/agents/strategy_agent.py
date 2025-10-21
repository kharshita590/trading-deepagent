from .base_agents import BaseAgent
from ..models.types import InvestmentAllocationState, AllocationStrategy

class StrategyDeterminationAgent(BaseAgent):    
    async def execute(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        self.log_message(state, "Determining allocation strategy")
        
        factors = state["allocation_factors"]
        amount = state['investment_amount']
        
        single_score, multi_score = self._calculate_strategy_scores(state, factors, amount)
        
        strategy, target_stocks = self._determine_strategy(single_score, multi_score, amount)
        
        state["allocation_strategy"] = strategy
        state["allocation_factors"]["target_stocks"] = target_stocks
        state["allocation_factors"]["decision_scores"] = {
            "single_stock": single_score,
            "multi_stock": multi_score
        }
        
        message = f"Strategy: {strategy.value}, Target stocks: {target_stocks} (single={single_score}, multi={multi_score})"
        self.log_message(state, message)
        
        return state
    
    def _calculate_strategy_scores(self, state, factors, amount):
        single_score = 0
        multi_score = 0
        
        if amount < 15000:
            single_score += 2
        elif amount < 50000:
            multi_score += 1
        else:
            multi_score += 3
        
        if factors["risk_assessment"]["risk_capacity"] == "high":
            single_score += 1
        else:
            multi_score += 2
        
        if state["market_conditions"]["market_volatility"] == "high":
            multi_score += 2
        elif state["market_conditions"]["market_volatility"] == "low":
            single_score += 1
        
        if state["diversification_requirement"]:
            multi_score += 3
        
        return single_score, multi_score
    
    def _determine_strategy(self, single_score, multi_score, amount):
        if single_score > multi_score:
            return AllocationStrategy.SINGLE_STOCK, 1
        elif multi_score > single_score + 1:
            if amount < 50000:
                target_stocks = min(3, max(2, amount // 15000))
            else:
                target_stocks = min(8, max(4, amount // 25000))
            return AllocationStrategy.MULTI_STOCK, target_stocks
        else:
            return AllocationStrategy.HYBRID, 2