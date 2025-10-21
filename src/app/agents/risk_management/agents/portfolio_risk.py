from typing import Dict, List, Any, Tuple
import numpy as np
import math
from ..models.types import RiskManagementState, RiskMetrics, RiskLevel, StopLossType
from ..config.settings import RiskConfig
class PortfolioRiskAssessmentAgent:
    def __init__(self, risk_parameters: Dict[str, Any]):
        self.risk_parameters = risk_parameters
    
    def assess_portfolio_risk(self, state: RiskManagementState) -> RiskMetrics:
        selected_stocks = state.get('selected_stocks', [])
        investment_amount = state.get('investment_amount', 100000)        
        portfolio_var_95, portfolio_var_99 = self._calculate_portfolio_var(selected_stocks, investment_amount)        
        correlation_risk = self._assess_correlation_risk(selected_stocks)        
        sector_risk = self._calculate_sector_concentration_risk(selected_stocks)        
        volatility_risk = self._calculate_volatility_risk(selected_stocks)        
        behavioral_adjustment = self._calculate_behavioral_risk_adjustment(state)        
        backtest_results = state.get('backtest_results', {})
        base_risk_limit = self.risk_parameters["max_portfolio_risk"] * 1.5
        historical_drawdown_limit = backtest_results.get('max_historical_drawdown', 0.15) * 1.2
        max_drawdown_limit = min(base_risk_limit, historical_drawdown_limit)
        
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