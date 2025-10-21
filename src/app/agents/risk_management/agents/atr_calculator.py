from typing import Dict, List, Any, Tuple
import numpy as np
import math
from ..models.types import RiskManagementState, RiskMetrics, RiskLevel, StopLossType
from ..config.settings import RiskConfig

class ATRCalculatorAgent:
    def __init__(self, risk_parameters: Dict[str, Any]):
        self.risk_parameters = risk_parameters
        self.atr_periods = RiskConfig.ATR_PERIODS
    
    def calculate_atr_based_stops(self, stock_data: Dict[str, Any], 
                                price_history: List[float] = None) -> Dict[str, float]:
        if not price_history:
            volatility = stock_data.get('volatility', 0.2)
            estimated_atr = stock_data.get('current_price', 100) * volatility / math.sqrt(252)
        else:
            estimated_atr = self._calculate_atr(price_history)
        
        current_price = stock_data.get('current_price', 100)
        atr_multiplier = self.risk_parameters["atr_multiplier"]        
        volatility_adjustment = self._get_volatility_adjustment(stock_data)
        adjusted_multiplier = atr_multiplier * volatility_adjustment
        stop_loss_distance = estimated_atr * adjusted_multiplier
        stop_loss_price = current_price - stop_loss_distance
        
        return {
            "atr_value": estimated_atr,
            "atr_multiplier": adjusted_multiplier,
            "stop_loss_distance": stop_loss_distance,
            "stop_loss_price": stop_loss_price,
            "stop_loss_percent": (stop_loss_distance / current_price) * 100
        }
    
    def _calculate_atr(self, price_history: List[float], period: int = 14) -> float:
        if len(price_history) < period + 1:
            return price_history[-1] * 0.02 
        
        true_ranges = []
        for i in range(1, len(price_history)):
            high_low = abs(price_history[i] - price_history[i-1])
            true_ranges.append(high_low)
        
        atr = np.mean(true_ranges[-period:]) if len(true_ranges) >= period else np.mean(true_ranges)
        return atr
    
    def _get_volatility_adjustment(self, stock_data: Dict[str, Any]) -> float:
        volatility = stock_data.get('volatility', 0.2)        
        if volatility < 0.15:  
            return 0.8  
        elif volatility > 0.35:  
            return 1.3  
        else: 
            return 1.0