from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

class BiasType(Enum):
    OVERCONFIDENCE = "overconfidence"
    LOSS_AVERSION = "loss_aversion"
    ANCHORING = "anchoring"
    CONFIRMATION = "confirmation"
    HERDING = "herding"
    FOMO = "fomo"

@dataclass
class PsychologicalProfile:
    risk_tolerance: RiskLevel
    bias_score: Dict[BiasType, float]
    emotional_state: str
    past_behavior_pattern: Dict[str, Any]

class BehavioralPsychologyState(TypedDict):
    investment_amount: float
    selected_stocks: List[Dict[str, Any]]
    technical_analysis: Dict[str, Any]
    volatility_data: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    
    psychological_profile: Dict[str, Any]
    behavioral_biases: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    trade_plan: Dict[str, Any]
    stop_loss_levels: Dict[str, float]
    take_profit_levels: Dict[str, float]
    position_sizing: Dict[str, float]
    
    risk_management_rules: List[Dict[str, Any]]
    emotional_triggers: Dict[str, Any]
    messages: Optional[List]