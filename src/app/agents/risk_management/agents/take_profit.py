from typing import Dict, List, Any, Tuple
import numpy as np
import math
from ..models.types import RiskManagementState, RiskMetrics, RiskLevel, StopLossType
from ..config.settings import RiskConfig

class TakeProfitAgent:
    def __init__(self, risk_parameters: Dict[str, Any]):
        self.risk_parameters = risk_parameters
    
    def calculate_take_profit_levels(self, stock_data: Dict[str, Any], 
                                   stop_loss_data: Dict[str, Any]) -> Dict[str, Any]:
        current_price = stock_data.get('current_price', 100)
        stop_loss_price = stop_loss_data.get('stop_loss_price', current_price * 0.95)
        risk_per_share = current_price - stop_loss_price        
        target_ratio = self.risk_parameters["take_profit_ratio"]
        rr_take_profit = current_price + (risk_per_share * target_ratio)        
        technical_data = stock_data.get('technical_levels', {})
        resistance_level = technical_data.get('resistance', current_price * 1.1)        
        volatility = stock_data.get('volatility', 0.2)
        bb_upper = current_price * (1 + 2 * volatility / math.sqrt(252) * 20)        
        support_level = technical_data.get('support', current_price * 0.9)
        fib_extension = current_price + 1.618 * (current_price - support_level)        
        take_profit_candidates = [rr_take_profit, resistance_level, bb_upper, fib_extension]
        primary_take_profit = min([tp for tp in take_profit_candidates if tp > current_price])        
        partial_levels = {
            "25_percent": current_price + (primary_take_profit - current_price) * 0.25,
            "50_percent": current_price + (primary_take_profit - current_price) * 0.50,
            "75_percent": current_price + (primary_take_profit - current_price) * 0.75,
            "final": primary_take_profit
        }
        
        return {
            "primary_take_profit": primary_take_profit,
            "risk_reward_ratio": (primary_take_profit - current_price) / risk_per_share,
            "partial_levels": partial_levels,
            "alternative_targets": {
                "resistance_based": resistance_level,
                "volatility_based": bb_upper,
                "fibonacci_based": fib_extension
            },
            "profit_booking_strategy": "25%-50%-75%-remainder"
        }