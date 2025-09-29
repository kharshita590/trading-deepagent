from typing import Dict, List, Any, TypedDict
import json
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from dataclasses import dataclass
from enum import Enum
import numpy as np

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
    
class BehavioralPsychologyAgent:
    def __init__(self):
        self.behavioral_patterns = self._load_behavioral_patterns()
        self.risk_models = self._initialize_risk_models()
        
    def _load_behavioral_patterns(self) -> Dict[str, Any]:
        return {
            "market_sentiment_impact": {
                "bull_market": {"overconfidence_boost": 1.3, "risk_tolerance_increase": 0.2},
                "bear_market": {"fear_boost": 1.5, "loss_aversion_increase": 0.3},
                "sideways": {"uncertainty_boost": 1.1, "anchoring_increase": 0.15}
            },
            "common_biases": {
                BiasType.OVERCONFIDENCE: {
                    "triggers": ["recent_wins", "bull_market", "high_technical_confidence"],
                    "impact": "increase_position_size",
                    "mitigation": "enforce_strict_stop_loss"
                },
                BiasType.LOSS_AVERSION: {
                    "triggers": ["recent_losses", "high_volatility", "bear_market"],
                    "impact": "premature_exit",
                    "mitigation": "systematic_profit_booking"
                },
                BiasType.FOMO: {
                    "triggers": ["trending_stocks", "social_media_buzz", "momentum_signals"],
                    "impact": "chase_expensive_entries",
                    "mitigation": "wait_for_pullback"
                }
            }
        }
    
    def _initialize_risk_models(self) -> Dict[str, Any]:
        return {
            "kelly_criterion": self._kelly_criterion,
            "var_model": self._calculate_var,
            "psychological_adjustment": self._psychological_risk_adjustment
        }
    
    def analyze_behavioral_biases(self, state: BehavioralPsychologyState) -> List[Dict[str, Any]]:
        biases = []
        
        technical_confidence = state.get('technical_analysis', {}).get('confidence_score', 0.5)
        if technical_confidence > 0.8:
            biases.append({
                "type": BiasType.OVERCONFIDENCE,
                "severity": min(technical_confidence, 1.0),
                "description": "High technical confidence may lead to overconfidence bias",
                "mitigation_strategy": "Reduce position size by 20%, implement tighter stops"
            })
        
        avg_volatility = np.mean([stock.get('volatility', 0) 
                                for stock in state.get('selected_stocks', [])])
        if avg_volatility > 0.3: 
            biases.append({
                "type": BiasType.LOSS_AVERSION,
                "severity": min(avg_volatility, 1.0),
                "description": "High volatility may trigger loss aversion behavior",
                "mitigation_strategy": "Use wider stops, smaller position sizes"
            })        
        momentum_score = state.get('technical_analysis', {}).get('momentum_score', 0.5)
        if momentum_score > 0.75:
            biases.append({
                "type": BiasType.FOMO,
                "severity": momentum_score,
                "description": "Strong momentum may trigger FOMO trading",
                "mitigation_strategy": "Wait for 5-10% pullback before entry"
            })
        
        return biases
    
    def _kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 0.1  
        win_loss_ratio = avg_win / avg_loss
        kelly_percent = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio        
        return min(max(kelly_percent, 0.02), 0.25)
    
    def _calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        if not returns:
            return 0.05 
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index])
    
    def _psychological_risk_adjustment(self, base_risk: float, biases: List[Dict[str, Any]]) -> float:
        """Adjust risk based on psychological factors"""
        adjustment_factor = 1.0
        
        for bias in biases:
            if bias['type'] == BiasType.OVERCONFIDENCE:
                adjustment_factor *= (1 - bias['severity'] * 0.3)  
            elif bias['type'] == BiasType.LOSS_AVERSION:
                adjustment_factor *= (1 - bias['severity'] * 0.2)  
            elif bias['type'] == BiasType.FOMO:
                adjustment_factor *= (1 - bias['severity'] * 0.25)  
        
        return base_risk * adjustment_factor
    
    def calculate_stop_loss_levels(self, state: BehavioralPsychologyState) -> Dict[str, float]:
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
        
        return stop_levels
    
    def calculate_position_sizing(self, state: BehavioralPsychologyState) -> Dict[str, float]:
        position_sizes = {}
        total_amount = state.get('investment_amount', 0)
        selected_stocks = state.get('selected_stocks', [])
        biases = state.get('behavioral_biases', [])
        
        if not selected_stocks:
            return position_sizes
        
        base_allocation = total_amount / len(selected_stocks)
        
        for stock in selected_stocks:
            symbol = stock.get('symbol')            
            win_rate = stock.get('historical_win_rate', 0.55)
            avg_win = stock.get('avg_win', 0.08)  
            avg_loss = stock.get('avg_loss', 0.04) 
            
            kelly_size = self._kelly_criterion(win_rate, avg_win, avg_loss) * total_amount            
            psychological_size = self._psychological_risk_adjustment(kelly_size, biases)            
            final_size = min(base_allocation, psychological_size)
            position_sizes[symbol] = {
                'amount': final_size,
                'percent_of_total': final_size / total_amount,
                'kelly_optimal': kelly_size,
                'psychological_adjusted': psychological_size
            }
        
        return position_sizes
    
    def generate_risk_management_rules(self, state: BehavioralPsychologyState) -> List[Dict[str, Any]]:
        rules = []
        biases = state.get('behavioral_biases', [])        
        rules.extend([
            {
                "rule_id": "max_single_position",
                "description": "No single position should exceed 20% of total portfolio",
                "type": "position_limit",
                "threshold": 0.20,
                "enforcement": "automatic"
            },
            {
                "rule_id": "daily_loss_limit",
                "description": "Stop trading if daily losses exceed 2% of total capital",
                "type": "daily_limit",
                "threshold": 0.02,
                "enforcement": "mandatory_break"
            }
        ])        
        for bias in biases:
            if bias['type'] == BiasType.OVERCONFIDENCE:
                rules.append({
                    "rule_id": "overconfidence_mitigation",
                    "description": "Mandatory profit booking at 15% gains due to overconfidence bias",
                    "type": "profit_booking",
                    "threshold": 0.15,
                    "enforcement": "partial_booking"
                })
            
            elif bias['type'] == BiasType.LOSS_AVERSION:
                rules.append({
                    "rule_id": "loss_aversion_mitigation",
                    "description": "Systematic stop-loss execution to prevent holding losers",
                    "type": "stop_loss_discipline",
                    "threshold": "as_calculated",
                    "enforcement": "automatic"
                })
            
            elif bias['type'] == BiasType.FOMO:
                rules.append({
                    "rule_id": "fomo_prevention",
                    "description": "Cooling-off period of 24 hours for momentum-driven trades",
                    "type": "entry_timing",
                    "threshold": "24_hours",
                    "enforcement": "delayed_entry"
                })
        
        return rules
    
    def __call__(self, state: BehavioralPsychologyState) -> BehavioralPsychologyState:
        behavioral_biases = self.analyze_behavioral_biases(state)        
        stop_loss_levels = self.calculate_stop_loss_levels(state)        
        position_sizing = self.calculate_position_sizing(state)        
        risk_management_rules = self.generate_risk_management_rules(state)        
        trade_plan = {
            "strategy": "behavioral_risk_adjusted",
            "total_positions": len(state.get('selected_stocks', [])),
            "max_risk_per_trade": 0.02, 
            "overall_portfolio_risk": 0.08,  
            "rebalancing_frequency": "weekly",
            "review_triggers": [
                "10% unrealized loss on any position",
                "Major market volatility spike (VIX > 30)",
                "Significant bias pattern changes"
            ]
        }
        
        state.update({
            'behavioral_biases': behavioral_biases,
            'stop_loss_levels': stop_loss_levels,
            'position_sizing': position_sizing,
            'risk_management_rules': risk_management_rules,
            'trade_plan': trade_plan,
            'psychological_profile': {
                'analysis_timestamp': 'current',
                'bias_count': len(behavioral_biases),
                'primary_concerns': [bias['type'].value for bias in behavioral_biases[:2]],
                'risk_adjustment_applied': True
            }
        })
        
        return state

def create_behavioral_psychology_node():
    agent = BehavioralPsychologyAgent()
    return agent

def add_to_langgraph_workflow(workflow: StateGraph):
    behavioral_agent = create_behavioral_psychology_node()    
    workflow.add_node("behavioral_psychology", behavioral_agent)    
    workflow.add_edge("volatility_agent", "behavioral_psychology")    
    workflow.add_edge("behavioral_psychology", "backtest_agent")
    return workflow