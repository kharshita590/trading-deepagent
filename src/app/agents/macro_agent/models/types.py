from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MacroAnalysis:
    economic_conditions_summary: str
    interest_rate_impact_summary: str
    global_events_summary: str
    market_sentiment: Optional[str] = None
    macro_data: Optional[Dict] = None


@dataclass
class StockRecommendation:
    ticker: str
    company_name: str
    sector: str
    price: float
    allocation_percentage: float
    allocation_amount: float
    market_cap: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class MacroData:
    interest_rates: Dict
    economic_indicators: Dict
    sector_performance: Dict
    fetch_timestamp: str


class MacroAgentState(TypedDict):
    recommendations: List[Dict]
    macro_data: Optional[Dict]
    economic_analysis: Optional[str]
    interest_rate_analysis: Optional[str]
    global_events_analysis: Optional[str]
    macro_analysis: Optional[MacroAnalysis]
    market_sentiment: Optional[str]
    messages: List
    portfolio_amount: Optional[float]


class DataFetchResult(TypedDict):
    success: bool
    data: Optional[Dict]
    error: Optional[str]
    timestamp: str
