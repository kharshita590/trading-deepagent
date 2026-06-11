from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass

@dataclass
class VolatilityLiquidityAnalysis:
    volatility_assessment_summary: str
    liquidity_analysis_summary: str
    risk_management_recommendations: str
    stock_metrics: Optional[Dict[str, Dict]] = None

class VolatilityLiquidityAgentState(TypedDict):
    recommendations: List[Dict]  
    volatility_liquidity_analysis: Optional[VolatilityLiquidityAnalysis]
    messages: List
    vol_liq_data: Optional[Dict]
    volatility_analysis: Optional[str]
    liquidity_analysis: Optional[str]
    risk_recommendations: Optional[str]
    vol_liq_data: Optional[Dict]
