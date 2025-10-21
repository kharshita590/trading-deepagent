from typing import Dict, List, Any, Tuple
import numpy as np
import math
from ..models.types import RiskManagementState, RiskMetrics, RiskLevel, StopLossType
from ..config.settings import RiskConfig

class RiskMonitoringAgent:
    def __init__(self, risk_parameters: Dict[str, Any]):
        self.risk_parameters = risk_parameters
    
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
