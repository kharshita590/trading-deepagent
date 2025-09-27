from typing import Dict, List, Any, Optional, Tuple, TypedDict
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math

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
    # backtest_results: Dict[str, Any]
    position_sizing: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    stop_loss_levels: Dict[str, Dict[str, Any]]
    take_profit_levels: Dict[str, Dict[str, Any]]
    position_risks: Dict[str, Dict[str, Any]]
    portfolio_risk_limits: Dict[str, Any]
    risk_monitoring_rules: List[Dict[str, Any]]
    emergency_exit_conditions: List[Dict[str, Any]]
    risk_adjusted_positions: Dict[str, Any]

class RiskManagementAgent:
    def __init__(self, base_risk_level: RiskLevel = RiskLevel.MODERATE):
        self.base_risk_level = base_risk_level
        self.risk_parameters = self._initialize_risk_parameters()
        self.atr_periods = 14  
        self.max_portfolio_risk = self._get_max_portfolio_risk()
        
    def _initialize_risk_parameters(self) -> Dict[str, Any]:
        risk_configs = {
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
        return risk_configs[self.base_risk_level]
    
    def _get_max_portfolio_risk(self) -> float:
        return self.risk_parameters["max_portfolio_risk"]
    
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
    
    def calculate_trailing_stop_parameters(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        current_price = stock_data.get('current_price', 100)
        volatility = stock_data.get('volatility', 0.2)
        atr_value = stock_data.get('atr_value', current_price * 0.02)        
        base_trailing_distance = atr_value * 2.0  
        if volatility > 0.3: 
            trailing_multiplier = 2.5
        elif volatility < 0.15:
            trailing_multiplier = 1.5
        else: 
            trailing_multiplier = 2.0
        
        trailing_distance = base_trailing_distance * trailing_multiplier
        activation_threshold = self.risk_parameters["trailing_stop_activation"]
        
        return {
            "trailing_distance": trailing_distance,
            "trailing_distance_percent": (trailing_distance / current_price) * 100,
            "activation_price": current_price * (1 + activation_threshold),
            "activation_threshold_percent": activation_threshold * 100,
            "update_frequency": "real_time",
            "step_size": atr_value * 0.25  
        }
    
    def assess_portfolio_risk(self, state: RiskManagementState) -> RiskMetrics:
        selected_stocks = state.get('selected_stocks', [])
        investment_amount = state.get('investment_amount', 100000)        
        portfolio_var_95, portfolio_var_99 = self._calculate_portfolio_var(selected_stocks, investment_amount)        
        correlation_risk = self._assess_correlation_risk(selected_stocks)        
        sector_risk = self._calculate_sector_concentration_risk(selected_stocks)        
        volatility_risk = self._calculate_volatility_risk(selected_stocks)        
        behavioral_adjustment = self._calculate_behavioral_risk_adjustment(state)        
        # backtest_results = state.get('backtest_results', {})
        max_drawdown_limit = min(
            self.risk_parameters["max_portfolio_risk"] * 1.5,  
            # backtest_results.get('max_historical_drawdown', 0.15) * 1.2  
        )        
        risk_components = [correlation_risk, sector_risk, volatility_risk, behavioral_adjustment]
        overall_risk = np.mean(risk_components)
        
        return RiskMetrics(
            portfolio_var_95=portfolio_var_95,
            portfolio_var_99=portfolio_var_99,
            max_drawdown_limit=max_drawdown_limit,
            correlation_risk_score=correlation_risk,
            sector_concentration_risk=sector_risk,
            volatility_risk_score=volatility_risk,
            behavioral_risk_adjustment=behavioral_adjustment,
            overall_risk_score=overall_risk
        )
    
    def _calculate_portfolio_var(self, selected_stocks: List[Dict[str, Any]], 
                               investment_amount: float) -> Tuple[float, float]:
        if not selected_stocks:
            return 0.05 * investment_amount, 0.10 * investment_amount        
        portfolio_volatility = 0
        total_weight = 0
        for stock in selected_stocks:
            weight = stock.get('weight', 1.0 / len(selected_stocks))
            volatility = stock.get('volatility', 0.2)
            portfolio_volatility += (weight * volatility) ** 2
            total_weight += weight
        portfolio_volatility = math.sqrt(portfolio_volatility)        
        daily_var_95 = investment_amount * portfolio_volatility * 1.65 / math.sqrt(252)  
        daily_var_99 = investment_amount * portfolio_volatility * 2.33 / math.sqrt(252)  
        return daily_var_95, daily_var_99
    
    def _assess_correlation_risk(self, selected_stocks: List[Dict[str, Any]]) -> float:
        if len(selected_stocks) <= 1:
            return 0.0        
        sectors = [stock.get('sector', 'Unknown') for stock in selected_stocks]
        unique_sectors = set(sectors)        
        diversification_score = len(unique_sectors) / len(selected_stocks)
        correlation_risk = 1.0 - diversification_score
        return min(correlation_risk, 1.0)
    
    def _calculate_sector_concentration_risk(self, selected_stocks: List[Dict[str, Any]]) -> float:
        if not selected_stocks:
            return 0.0
        sector_weights = {}
        for stock in selected_stocks:
            sector = stock.get('sector', 'Unknown')
            weight = stock.get('weight', 1.0 / len(selected_stocks))
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        concentration_limit = self.risk_parameters["sector_concentration_limit"]        
        if max_sector_weight > concentration_limit:
            return min((max_sector_weight - concentration_limit) / concentration_limit, 1.0)
        else:
            return 0.0
    
    def _calculate_volatility_risk(self, selected_stocks: List[Dict[str, Any]]) -> float:
        if not selected_stocks:
            return 0.0
        
        volatilities = [stock.get('volatility', 0.2) for stock in selected_stocks]
        avg_volatility = np.mean(volatilities)        
        if avg_volatility < 0.15:  
            return 0.2
        elif avg_volatility > 0.35:  
            return 0.8
        else: 
            return 0.5
    
    def _calculate_behavioral_risk_adjustment(self, state: RiskManagementState) -> float:
        behavioral_biases = state.get('behavioral_biases', [])
        
        if not behavioral_biases:
            return 0.0
        
        risk_adjustment = 0.0
        for bias in behavioral_biases:
            severity = bias.get('severity', 0.5)
            bias_type = bias.get('type')            
            if bias_type == 'overconfidence':
                risk_adjustment += severity * 0.3
            elif bias_type == 'loss_aversion':
                risk_adjustment += severity * 0.2
            elif bias_type == 'fomo':
                risk_adjustment += severity * 0.4
            else:
                risk_adjustment += severity * 0.25
        
        return min(risk_adjustment, 1.0)
    
    def generate_risk_monitoring_rules(self, state: RiskManagementState) -> List[Dict[str, Any]]:
        rules = []
        risk_metrics = state.get('risk_metrics', {})        
        rules.extend([
            {
                "rule_id": "portfolio_var_limit",
                "description": f"Alert if daily VaR exceeds {risk_metrics.get('portfolio_var_95', 0):.0f}",
                "type": "portfolio_limit",
                "threshold": risk_metrics.get('portfolio_var_95', 0),
                "action": "reduce_position_sizes",
                "priority": "high"
            },
            {
                "rule_id": "max_drawdown_monitoring",
                "description": f"Stop trading if drawdown exceeds {risk_metrics.get('max_drawdown_limit', 0.1)*100:.1f}%",
                "type": "drawdown_limit",
                "threshold": risk_metrics.get('max_drawdown_limit', 0.1),
                "action": "emergency_exit",
                "priority": "critical"
            },
            {
                "rule_id": "correlation_spike_detection",
                "description": "Alert if inter-position correlation exceeds 0.85",
                "type": "correlation_monitoring",
                "threshold": 0.85,
                "action": "diversify_positions",
                "priority": "medium"
            }
        ])        
        for stock in state.get('selected_stocks', []):
            symbol = stock.get('symbol')
            rules.append({
                "rule_id": f"stop_loss_monitoring_{symbol}",
                "description": f"Execute stop loss for {symbol} at calculated level",
                "type": "stop_loss_execution",
                "symbol": symbol,
                "threshold": "calculated_stop_loss",
                "action": "market_sell",
                "priority": "high"
            })        
        macro_data = state.get('macro_analysis', {})
        if macro_data.get('market_sentiment') == 'bearish':
            rules.append({
                "rule_id": "bear_market_protection",
                "description": "Reduce position sizes by 50% in confirmed bear market",
                "type": "market_condition",
                "threshold": "bear_market_confirmed",
                "action": "reduce_exposure",
                "priority": "high"
            })
        
        return rules
    
    def __call__(self, state: RiskManagementState) -> RiskManagementState:
        risk_metrics = self.assess_portfolio_risk(state)        
        stop_loss_levels = {}
        take_profit_levels = {}
        position_risks = {}
        
        selected_stocks = state.get('selected_stocks', [])
        
        for stock in selected_stocks:
            symbol = stock.get('symbol')            
            atr_stops = self.calculate_atr_based_stops(stock)            
            take_profits = self.calculate_take_profit_levels(stock, atr_stops)            
            trailing_params = self.calculate_trailing_stop_parameters(stock)            
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
        risk_monitoring_rules = self.generate_risk_monitoring_rules(state)
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

def create_risk_management_node(risk_level: RiskLevel = RiskLevel.MODERATE):
    agent = RiskManagementAgent(risk_level)
    return agent

def add_risk_management_to_workflow(workflow: StateGraph, risk_level: RiskLevel = RiskLevel.MODERATE):
    risk_agent = create_risk_management_node(risk_level)    
    workflow.add_node("risk_management", risk_agent)    
    # workflow.add_edge("backtest_agent", "risk_management")
    workflow.add_edge("risk_management", "execution_agent") 
    return workflow