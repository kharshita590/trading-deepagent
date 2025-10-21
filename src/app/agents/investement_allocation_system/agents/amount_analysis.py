from .base_agents import BaseAgent
from ..models.types import InvestmentAllocationState
from langchain_core.messages import AIMessage

class AmountAnalysisAgent(BaseAgent):    
    def __init__(self, llm, allocation_factors):
        super().__init__(llm)
        self.allocation_factors = allocation_factors
    
    async def execute(self, state: InvestmentAllocationState) -> InvestmentAllocationState:
        amount = state['investment_amount']
        self.log_message(state, f"Analyzing investment amount: ₹{amount}")
        
        thresholds = self.allocation_factors["amount_thresholds"]
        
        if amount < thresholds["small_investment"]:
            initial_strategy_lean = "single_stock"
            diversification_requirement = False
            message = f"Small investment (₹{amount}). Single stock acceptable."
        elif amount < thresholds["medium_investment"]:
            initial_strategy_lean = "multi_stock"
            diversification_requirement = True
            message = f"Medium investment (₹{amount}). 2-5 stocks recommended."
        else:
            initial_strategy_lean = "multi_stock"
            diversification_requirement = True
            message = f"Large investment (₹{amount}). 5+ stocks recommended."
        
        state["allocation_factors"] = {
            "initial_strategy_lean": initial_strategy_lean,
            "amount_category": "small" if amount < 10000 else "medium" if amount < 100000 else "large"
        }
        state["diversification_requirement"] = diversification_requirement
        state["messages"] = state.get("messages", []) + [AIMessage(content=message)]
        
        return state
