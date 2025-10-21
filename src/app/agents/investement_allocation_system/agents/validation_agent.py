
from .base_agents import BaseAgent
from ..models.types import InvestmentAllocationState, AllocationDecision, AllocationStrategy

class ValidationAgent(BaseAgent):    
    async def execute(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        self.log_message(state, "Validating allocation decision")
        strategy = state["allocation_strategy"]
        total_allocated = state["investment_amount"]        
        risk_score = 0.8 if strategy == AllocationStrategy.SINGLE_STOCK else 0.5
        diversification_score = 0.2 if strategy == AllocationStrategy.SINGLE_STOCK else 0.7        
        decision = AllocationDecision(
            strategy=strategy,
            total_amount=total_allocated,
            risk_score=risk_score,
            diversification_score=diversification_score,
            reasoning=f"{strategy.value} strategy. Risk: {risk_score:.2f}, Diversification: {diversification_score:.2f}"
        )
        
        state["allocation_decision"] = decision
        state["next_agent"] = "research_phase_agent"
        
        message = f"Validated: ₹{total_allocated:.2f}, Risk: {risk_score:.2f}"
        self.log_message(state, message)
        
        return state