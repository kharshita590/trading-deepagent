from app.agents.risk_management.agents.atr_calculator import ATRCalculatorAgent
from app.agents.risk_management.agents.take_profit import TakeProfitAgent
from app.agents.risk_management.config.settings import RiskConfig
from app.agents.risk_management.models.types import RiskLevel


def test_atr_stop_loss_math():
    risk_params = RiskConfig.get_risk_parameters(RiskLevel.MODERATE)
    atr = ATRCalculatorAgent(risk_params)
    stop = atr.calculate_atr_based_stops({"current_price": 100, "volatility": 0.2})
    assert stop["stop_loss_price"] < 100
    assert stop["stop_loss_percent"] > 0


def test_take_profit_math_uses_technical_levels():
    risk_params = RiskConfig.get_risk_parameters(RiskLevel.MODERATE)
    tp = TakeProfitAgent(risk_params)
    result = tp.calculate_take_profit_levels(
        {"current_price": 100, "volatility": 0.2, "technical_levels": {"resistance": 108, "support": 95}},
        {"stop_loss_price": 96},
    )
    assert result["primary_take_profit"] > 100
    assert result["risk_reward_ratio"] > 0
