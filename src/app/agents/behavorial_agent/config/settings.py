import logging
from ..models.types import BiasType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = ""
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0

BEHAVIORAL_PATTERNS = {
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

RISK_THRESHOLDS = {
    "max_single_position": 0.20,
    "daily_loss_limit": 0.02,
    "max_risk_per_trade": 0.02,
    "overall_portfolio_risk": 0.08,
    "kelly_max": 0.25,
    "kelly_min": 0.02
}

BASE_RISK_RULES = [
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
]