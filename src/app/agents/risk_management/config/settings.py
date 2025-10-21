from typing import Dict, Any
from ..models.types import RiskLevel

class RiskConfig:
    ATR_PERIODS = 14
    
    RISK_PARAMETERS = {
        RiskLevel.CONSERVATIVE: {
            "max_position_risk": 0.01,  
            "max_portfolio_risk": 0.05, 
            "atr_multiplier": 3.0,  
            "take_profit_ratio": 2.0,  
            "trailing_stop_activation": 0.02,  
            "correlation_limit": 0.7,  
            "sector_concentration_limit": 0.25  
        },
        RiskLevel.MODERATE: {
            "max_position_risk": 0.02,
            "max_portfolio_risk": 0.08,  
            "atr_multiplier": 2.5,
            "take_profit_ratio": 2.5, 
            "trailing_stop_activation": 0.03,  
            "correlation_limit": 0.75,
            "sector_concentration_limit": 0.35
        },
        RiskLevel.AGGRESSIVE: {
            "max_position_risk": 0.03,  
            "max_portfolio_risk": 0.12,  
            "atr_multiplier": 2.0,
            "take_profit_ratio": 3.0, 
            "trailing_stop_activation": 0.05,  
            "correlation_limit": 0.8,
            "sector_concentration_limit": 0.45
        },
        RiskLevel.VERY_AGGRESSIVE: {
            "max_position_risk": 0.05,  
            "max_portfolio_risk": 0.20,  
            "atr_multiplier": 1.5,
            "take_profit_ratio": 4.0, 
            "trailing_stop_activation": 0.08, 
            "correlation_limit": 0.85,
            "sector_concentration_limit": 0.60
        }
    }
    
    @staticmethod
    def get_risk_parameters(risk_level: RiskLevel) -> Dict[str, Any]:
        return RiskConfig.RISK_PARAMETERS[risk_level]