from typing import TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

class SignalType(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    NEUTRAL = "NEUTRAL"

@dataclass
class StockRecommendation:
    ticker: str
    company_name: str
    sector: str
    price: float
    allocation_percentage: float
    market_cap: Optional[float] = None
    reasoning: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StockRecommendation':
        return cls(
            ticker=data.get('ticker', ''),
            company_name=data.get('company_name', ''),
            sector=data.get('sector', ''),
            price=data.get('price', 0.0),
            allocation_percentage=data.get('allocation_percentage', 0.0),
            market_cap=data.get('market_cap'),
            reasoning=data.get('reasoning')
        )

@dataclass
class TechnicalIndicators:
    sma_20: Any = None
    sma_50: Any = None
    sma_200: Any = None
    ema_12: Any = None
    ema_26: Any = None
    ema_50: Any = None
    macd: Any = None
    macd_signal: Any = None
    macd_histogram: Any = None
    rsi: Any = None
    stoch_k: Any = None
    stoch_d: Any = None
    bb_upper: Any = None
    bb_middle: Any = None
    bb_lower: Any = None
    obv: Any = None
    ad: Any = None
    williams_r: Any = None
    cci: Any = None
    adx: Any = None
    atr: Any = None
    pivot_point: float = 0.0
    resistance_1: float = 0.0
    support_1: float = 0.0
    resistance_2: float = 0.0
    support_2: float = 0.0
    price_vs_sma20: float = 0.0
    price_vs_sma50: float = 0.0
    price_vs_sma200: float = 0.0
    volume_ratio: float = 1.0

@dataclass
class ChartPatterns:
    doji: Any = None
    hammer: Any = None
    shooting_star: Any = None
    engulfing_bullish: Any = None
    morning_star: Any = None
    evening_star: Any = None
    three_white_soldiers: Any = None
    three_black_crows: Any = None
    hanging_man: Any = None
    inverted_hammer: Any = None
    higher_highs: bool = False
    lower_lows: bool = False
    above_sma20: bool = False
    below_sma20: bool = False
    high_volume_breakout: bool = False
    low_volume_decline: bool = False

@dataclass
class TradingSignals:
    overall_signal: SignalType = SignalType.NEUTRAL
    strength: int = 0
    bullish_signals: List[str] = field(default_factory=list)
    bearish_signals: List[str] = field(default_factory=list)
    key_levels: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class StockTechnicalData:
    ticker: str
    recommendation: StockRecommendation
    price_data: Optional[pd.DataFrame] = None
    indicators: Optional[TechnicalIndicators] = None
    patterns: Optional[ChartPatterns] = None
    signals: Optional[TradingSignals] = None
    current_price: float = 0.0
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0
    price_change_1m: float = 0.0
    error: Optional[str] = None

@dataclass
class TechnicalAnalysis:
    technical_indicators_summary: str
    pattern_detection_summary: str
    technical_trading_signals: str
    timestamp: Optional[str] = None

class AgentState(TypedDict):
    messages: List
    error: Optional[str]

class TechnicalAgentState(AgentState):
    recommendations: List[Dict]
    technical_data: Optional[Dict[str, StockTechnicalData]]
    indicators_analysis: Optional[str]
    patterns_analysis: Optional[str]
    signals_analysis: Optional[str]
    technical_analysis: Optional[TechnicalAnalysis]

class DataFetcherState(AgentState):
    recommendations: List[Dict]
    technical_data: Dict[str, StockTechnicalData]

class IndicatorAnalysisState(AgentState):
    technical_data: Dict[str, StockTechnicalData]
    indicators_analysis: str

class PatternAnalysisState(AgentState):
    technical_data: Dict[str, StockTechnicalData]
    patterns_analysis: str

class SignalGenerationState(AgentState):
    technical_data: Dict[str, StockTechnicalData]
    indicators_analysis: str
    patterns_analysis: str
    signals_analysis: str

@dataclass
class AgentResponse:
    success: bool
    data: Any = None
    error: Optional[str] = None
    message: Optional[str] = None
