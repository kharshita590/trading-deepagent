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

class RiskCalculator:
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return RISK_THRESHOLDS["kelly_min"]
        win_loss_ratio = avg_win / avg_loss
        kelly_percent = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio        
        return min(max(kelly_percent, RISK_THRESHOLDS["kelly_min"]), RISK_THRESHOLDS["kelly_max"])
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
        if not returns:
            return 0.05
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index])
    
    @staticmethod
    def psychological_risk_adjustment(base_risk: float, biases: List[Dict[str, Any]]) -> float:
        adjustment_factor = 1.0
        
        for bias in biases:
            if bias['type'] == BiasType.OVERCONFIDENCE:
                adjustment_factor *= (1 - bias['severity'] * 0.3)
            elif bias['type'] == BiasType.LOSS_AVERSION:
                adjustment_factor *= (1 - bias['severity'] * 0.2)
            elif bias['type'] == BiasType.FOMO:
                adjustment_factor *= (1 - bias['severity'] * 0.25)
        
        return base_risk * adjustment_factor
class StopLossCalculationAgent:
    def __init__(self):
        self.risk_calculator = RiskCalculator()
    
    async def execute(self, state: BehavioralPsychologyState) -> BehavioralPsychologyState:
        logger.info("Calculating stop-loss levels")
        
        stop_levels = {}
        biases = state.get('behavioral_biases', [])
        
        for stock in state.get('selected_stocks', []):
            symbol = stock.get('symbol')
            current_price = stock.get('current_price', 0)
            volatility = stock.get('volatility', 0.2)
            
            base_stop_percent = volatility * 2
            psychological_adjustment = 1.0
            for bias in biases:
                if bias['type'] == BiasType.LOSS_AVERSION:
                    psychological_adjustment *= 1.2
                elif bias['type'] == BiasType.OVERCONFIDENCE:
                    psychological_adjustment *= 0.8
            
            final_stop_percent = base_stop_percent * psychological_adjustment
            stop_price = current_price * (1 - final_stop_percent)
            
            stop_levels[symbol] = {
                'stop_price': stop_price,
                'stop_percent': final_stop_percent,
                'psychological_factor': psychological_adjustment
            }
        
        state['stop_loss_levels'] = stop_levels
        state['messages'].append(AIMessage(content=f"Calculated stop-loss levels for {len(stop_levels)} stocks"))
        
        return state

class PositionSizingAgent:
    def __init__(self):
        self.risk_calculator = RiskCalculator()
    
    async def execute(self, state: BehavioralPsychologyState) -> BehavioralPsychologyState:
        logger.info("Calculating position sizing")
        
        position_sizes = {}
        total_amount = state.get('investment_amount', 0)
        selected_stocks = state.get('selected_stocks', [])
        biases = state.get('behavioral_biases', [])
        
        if not selected_stocks:
            state['position_sizing'] = position_sizes
            return state
        
        base_allocation = total_amount / len(selected_stocks)
        
        for stock in selected_stocks:
            symbol = stock.get('symbol')
            
            win_rate = stock.get('historical_win_rate', 0.55)
            avg_win = stock.get('avg_win', 0.08)
            avg_loss = stock.get('avg_loss', 0.04)
            
            kelly_size = self.risk_calculator.kelly_criterion(win_rate, avg_win, avg_loss) * total_amount
            
            psychological_size = self.risk_calculator.psychological_risk_adjustment(kelly_size, biases)
            
            final_size = min(base_allocation, psychological_size)
            position_sizes[symbol] = {
                'amount': final_size,
                'percent_of_total': final_size / total_amount,
                'kelly_optimal': kelly_size,
                'psychological_adjusted': psychological_size
            }
        
        state['position_sizing'] = position_sizes
        state['messages'].append(AIMessage(content=f"Calculated position sizing for {len(position_sizes)} stocks"))
        
        return state