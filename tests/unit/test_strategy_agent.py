from app.agents.investment_allocation_system.agents.strategy_agent import StrategyDeterminationAgent
from app.agents.investment_allocation_system.models.types import AllocationStrategy


def test_strategy_scores_and_determination():
    agent = StrategyDeterminationAgent()
    state = {
        "investment_amount": 75000,
        "allocation_factors": {"risk_assessment": {"risk_capacity": "low"}},
        "market_conditions": {"market_volatility": "high"},
        "diversification_requirement": True,
    }
    single_score, multi_score = agent._calculate_strategy_scores(state, state["allocation_factors"], state["investment_amount"])
    assert multi_score > single_score
    strategy, target_stocks = agent._determine_strategy(single_score, multi_score, state["investment_amount"])
    assert strategy in {AllocationStrategy.MULTI_STOCK, AllocationStrategy.HYBRID}
    assert target_stocks >= 2
