from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class AllocationStrategy(Enum):
    SINGLE_STOCK = "single_stock"
    MULTI_STOCK = "multi_stock"
    HYBRID = "hybrid"

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"

@dataclass
class AllocationDecision:
    strategy: AllocationStrategy
    total_amount: float
    risk_score: float
    diversification_score: float
    reasoning: str

class InvestmentAllocationState(TypedDict):
    investment_amount: float
    user_risk_tolerance: RiskLevel
    investment_horizon: str 
    user_preferences: Dict 
    market_conditions: Dict
    allocation_factors: Dict
    volatility_threshold: float
    diversification_requirement: bool
    sector_constraints: List[str]
    allocation_strategy: AllocationStrategy
    allocation_decision: Optional[AllocationDecision]
    messages: List
    next_agent: str

