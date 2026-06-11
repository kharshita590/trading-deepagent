from typing import Dict, List, Any
import numpy as np
from langchain_core.messages import AIMessage
from ..models.types import BehavioralPsychologyState, BiasType
from ..config.settings import (
    logger, 
    BEHAVIORAL_PATTERNS, 
    RISK_THRESHOLDS, 
    BASE_RISK_RULES
)

class TradePlanGenerationAgent:
    async def execute(self, state: BehavioralPsychologyState) -> BehavioralPsychologyState:
        logger.info("Generating trade plan")
        
        biases = state.get('behavioral_biases', [])
        
        trade_plan = {
            "strategy": "behavioral_risk_adjusted",
            "total_positions": len(state.get('selected_stocks', [])),
            "max_risk_per_trade": RISK_THRESHOLDS["max_risk_per_trade"],
            "overall_portfolio_risk": RISK_THRESHOLDS["overall_portfolio_risk"],
            "rebalancing_frequency": "weekly",
            "review_triggers": [
                "10% unrealized loss on any position",
                "Major market volatility spike (VIX > 30)",
                "Significant bias pattern changes"
            ]
        }
        
        state['trade_plan'] = trade_plan
        state['messages'].append(AIMessage(content="Generated comprehensive trade plan"))
        
        return state