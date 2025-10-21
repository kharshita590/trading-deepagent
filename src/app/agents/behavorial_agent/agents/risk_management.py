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

class RiskManagementRulesAgent:
    def __init__(self):
        self.base_rules = BASE_RISK_RULES
    
    async def execute(self, state: BehavioralPsychologyState) -> BehavioralPsychologyState:
        logger.info("Generating risk management rules")
        
        rules = list(self.base_rules)
        biases = state.get('behavioral_biases', [])
        
        for bias in biases:
            if bias['type'] == BiasType.OVERCONFIDENCE:
                rules.append({
                    "rule_id": "overconfidence_mitigation",
                    "description": "Mandatory profit booking at 15% gains due to overconfidence bias",
                    "type": "profit_booking",
                    "threshold": 0.15,
                    "enforcement": "partial_booking"
                })
            
            elif bias['type'] == BiasType.LOSS_AVERSION:
                rules.append({
                    "rule_id": "loss_aversion_mitigation",
                    "description": "Systematic stop-loss execution to prevent holding losers",
                    "type": "stop_loss_discipline",
                    "threshold": "as_calculated",
                    "enforcement": "automatic"
                })
            
            elif bias['type'] == BiasType.FOMO:
                rules.append({
                    "rule_id": "fomo_prevention",
                    "description": "Cooling-off period of 24 hours for momentum-driven trades",
                    "type": "entry_timing",
                    "threshold": "24_hours",
                    "enforcement": "delayed_entry"
                })
        
        state['risk_management_rules'] = rules
        state['messages'].append(AIMessage(content=f"Generated {len(rules)} risk management rules"))
        
        return state
