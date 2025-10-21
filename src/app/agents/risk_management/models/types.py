from typing import Dict, List, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass
from enum import Enum

class StopLossType(Enum):
    FIXED = "fixed"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    PSYCHOLOGICAL = "psychological"
    HYBRID = "hybrid"

class RiskLevel(Enum):
    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3
    VERY_AGGRESSIVE = 4

@dataclass
class RiskMetrics:
    portfolio_var_95: float
    portfolio_var_99: float  
    max_drawdown_limit: float
    correlation_risk_score: float
    sector_concentration_risk: float
    volatility_risk_score: float
    behavioral_risk_adjustment: float
    overall_risk_score: float

@dataclass
class PositionRisk:
    symbol: str
    entry_price: float
    position_size: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_distance: float
    max_position_risk: float  
    atr_multiplier: float
    volatility_adjustment: float

class RiskManagementState(TypedDict):
    investment_amount: float
    selected_stocks: List[Dict[str, Any]]
    macro_analysis: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    volatility_data: Dict[str, Any]
    behavioral_biases: List[Dict[str, Any]]
    position_sizing: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    stop_loss_levels: Dict[str, Dict[str, Any]]
    take_profit_levels: Dict[str, Dict[str, Any]]
    position_risks: Dict[str, Dict[str, Any]]
    portfolio_risk_limits: Dict[str, Any]
    risk_monitoring_rules: List[Dict[str, Any]]
    emergency_exit_conditions: List[Dict[str, Any]]
    risk_adjusted_positions: Dict[str, Any]