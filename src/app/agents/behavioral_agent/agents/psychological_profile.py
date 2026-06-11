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

class PsychologicalProfileAgent:
    async def execute(self, state: BehavioralPsychologyState) -> BehavioralPsychologyState:
        logger.info("Compiling psychological profile")
        
        biases = state.get('behavioral_biases', [])
        
        state['psychological_profile'] = {
            'analysis_timestamp': 'current',
            'bias_count': len(biases),
            'primary_concerns': [bias['type'].value for bias in biases[:2]],
            'risk_adjustment_applied': True
        }
        
        state['messages'].append(AIMessage(content="Completed psychological profile analysis"))
        
        return state