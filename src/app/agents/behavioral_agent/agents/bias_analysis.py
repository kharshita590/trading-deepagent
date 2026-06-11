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

class BiasAnalysisAgent:
    def __init__(self):
        self.behavioral_patterns = BEHAVIORAL_PATTERNS
    
    async def execute(self, state: BehavioralPsychologyState) -> BehavioralPsychologyState:
        logger.info("Analyzing behavioral biases")
        
        biases = []
        
        technical_confidence = state.get('technical_analysis', {}).get('confidence_score', 0.5)
        if technical_confidence > 0.8:
            biases.append({
                "type": BiasType.OVERCONFIDENCE,
                "severity": min(technical_confidence, 1.0),
                "description": "High technical confidence may lead to overconfidence bias",
                "mitigation_strategy": "Reduce position size by 20%, implement tighter stops"
            })
        
        avg_volatility = np.mean([stock.get('volatility', 0) 
                                for stock in state.get('selected_stocks', [])])
        if avg_volatility > 0.3:
            biases.append({
                "type": BiasType.LOSS_AVERSION,
                "severity": min(avg_volatility, 1.0),
                "description": "High volatility may trigger loss aversion behavior",
                "mitigation_strategy": "Use wider stops, smaller position sizes"
            })
        
        momentum_score = state.get('technical_analysis', {}).get('momentum_score', 0.5)
        if momentum_score > 0.75:
            biases.append({
                "type": BiasType.FOMO,
                "severity": momentum_score,
                "description": "Strong momentum may trigger FOMO trading",
                "mitigation_strategy": "Wait for 5-10% pullback before entry"
            })
        
        state['behavioral_biases'] = biases
        if 'messages' not in state:
            state['messages'] = []
        state['messages'].append(AIMessage(content=f"Identified {len(biases)} behavioral biases"))
        
        return state