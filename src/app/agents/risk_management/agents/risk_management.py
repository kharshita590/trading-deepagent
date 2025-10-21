from typing import Dict, List, Any, Tuple
import numpy as np
import math
from ..models.types import RiskManagementState, RiskMetrics, RiskLevel, StopLossType
from ..config.settings import RiskConfig
from .atr_calculator import ATRCalculatorAgent
from .portfolio_risk import PortfolioRiskAssessmentAgent
from .risk_monitor import RiskMonitoringAgent
from .trailing_stop import TrailingStopAgent
from .take_profit import TakeProfitAgent

class RiskManagementAgent:
    def __init__(self, base_risk_level: RiskLevel = RiskLevel.MODERATE):
        self.base_risk_level = base_risk_level
        self.risk_parameters = RiskConfig.get_risk_parameters(base_risk_level)
        
        self.atr_agent = ATRCalculatorAgent(self.risk_parameters)
        self.take_profit_agent = TakeProfitAgent(self.risk_parameters)
        self.trailing_stop_agent = TrailingStopAgent(self.risk_parameters)
        self.portfolio_risk_agent = PortfolioRiskAssessmentAgent(self.risk_parameters)
        self.monitoring_agent = RiskMonitoringAgent(self.risk_parameters)
    
    def __call__(self, state: RiskManagementState) -> RiskManagementState:
        risk_metrics = self.portfolio_risk_agent.assess_portfolio_risk(state)        
        stop_loss_levels = {}
        take_profit_levels = {}
        position_risks = {}
        
        selected_stocks = state.get('selected_stocks', [])
        
        for stock in selected_stocks:
            symbol = stock.get('symbol')            
            atr_stops = self.atr_agent.calculate_atr_based_stops(stock)            
            take_profits = self.take_profit_agent.calculate_take_profit_levels(stock, atr_stops)            
            trailing_params = self.trailing_stop_agent.calculate_trailing_stop_parameters(stock)            
            behavioral_biases = state.get('behavioral_biases', [])
            psychological_adjustment = 1.0
            
            for bias in behavioral_biases:
                if bias.get('type') == 'loss_aversion':
                    psychological_adjustment *= 1.2  
                elif bias.get('type') == 'overconfidence':
                    psychological_adjustment *= 0.9  
            
            final_stop_loss = atr_stops['stop_loss_price'] * psychological_adjustment
            
            stop_loss_levels[symbol] = {
                "final_stop_loss": final_stop_loss,
                "atr_based": atr_stops,
                "psychological_adjustment": psychological_adjustment,
                "trailing_parameters": trailing_params,
                "stop_type": StopLossType.HYBRID.value
            }
            
            take_profit_levels[symbol] = take_profits            
            current_price = stock.get('current_price', 100)
            position_size = state.get('position_sizing', {}).get(symbol, {}).get('amount', 0)
            position_risk_percent = abs(current_price - final_stop_loss) / current_price
            position_dollar_risk = position_size * position_risk_percent
            
            position_risks[symbol] = {
                "position_size": position_size,
                "risk_per_share": abs(current_price - final_stop_loss),
                "risk_percent": position_risk_percent * 100,
                "dollar_risk": position_dollar_risk,
                "max_acceptable_risk": self.risk_parameters["max_position_risk"] * state.get('investment_amount', 100000)
            }        
        portfolio_risk_limits = {
            "max_portfolio_risk": risk_metrics.max_drawdown_limit,
            "max_daily_loss": risk_metrics.portfolio_var_95,
            "max_position_correlation": self.risk_parameters["correlation_limit"],
            "max_sector_concentration": self.risk_parameters["sector_concentration_limit"],
            "rebalancing_threshold": 0.05,
            "risk_parity_target": True  
        }
        
        risk_monitoring_rules = self.monitoring_agent.generate_risk_monitoring_rules(state)
        
        emergency_exit_conditions = [
            {
                "condition": "portfolio_loss_exceeds_limit",
                "threshold": risk_metrics.max_drawdown_limit,
                "action": "liquidate_all_positions",
                "priority": "critical"
            },
            {
                "condition": "individual_position_loss_exceeds_stop",
                "threshold": "calculated_stop_loss",
                "action": "exit_position_immediately",
                "priority": "high"
            },
            {
                "condition": "market_crash_detected",
                "threshold": "vix_above_40_or_spy_down_10_percent",
                "action": "reduce_exposure_by_50_percent",
                "priority": "high"
            }
        ]        
        risk_adjusted_positions = {}
        total_portfolio_risk = sum(pos['dollar_risk'] for pos in position_risks.values())
        investment_amount = state.get('investment_amount', 100000)
        
        if total_portfolio_risk > risk_metrics.max_drawdown_limit * investment_amount:
            risk_reduction_factor = (risk_metrics.max_drawdown_limit * investment_amount) / total_portfolio_risk
            for symbol, pos_risk in position_risks.items():
                original_size = pos_risk['position_size']
                adjusted_size = original_size * risk_reduction_factor
                
                risk_adjusted_positions[symbol] = {
                    "original_size": original_size,
                    "adjusted_size": adjusted_size,
                    "reduction_factor": risk_reduction_factor,
                    "reason": "portfolio_risk_limit_exceeded"
                }        
        state.update({
            'risk_metrics': {
                'portfolio_var_95': risk_metrics.portfolio_var_95,
                'portfolio_var_99': risk_metrics.portfolio_var_99,
                'max_drawdown_limit': risk_metrics.max_drawdown_limit,
                'correlation_risk_score': risk_metrics.correlation_risk_score,
                'sector_concentration_risk': risk_metrics.sector_concentration_risk,
                'volatility_risk_score': risk_metrics.volatility_risk_score,
                'behavioral_risk_adjustment': risk_metrics.behavioral_risk_adjustment,
                'overall_risk_score': risk_metrics.overall_risk_score
            },
            'stop_loss_levels': stop_loss_levels,
            'take_profit_levels': take_profit_levels,
            'position_risks': position_risks,
            'portfolio_risk_limits': portfolio_risk_limits,
            'risk_monitoring_rules': risk_monitoring_rules,
            'emergency_exit_conditions': emergency_exit_conditions,
            'risk_adjusted_positions': risk_adjusted_positions,
            'risk_management_summary': {
                'total_positions': len(selected_stocks),
                'total_portfolio_risk_dollars': total_portfolio_risk,
                'total_portfolio_risk_percent': (total_portfolio_risk / investment_amount) * 100,
                'risk_level': self.base_risk_level.name,
                'primary_stop_method': 'ATR_HYBRID_WITH_PSYCHOLOGICAL_ADJUSTMENT',
                'take_profit_strategy': 'PARTIAL_PROFIT_BOOKING',
                'monitoring_frequency': 'REAL_TIME'
            }
        })
        
        return state