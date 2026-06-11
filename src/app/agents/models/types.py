from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel,Field
class QueryState(BaseModel):
    user_query: str = ""
    extracted_parameters: Dict = Field(default_factory=dict)
    missing_parameters: List[str] = Field(default_factory=list)
    is_complete: bool = False
    conversation_history: List[Dict] = Field(default_factory=list)

# i want to invest in long horizon and my preferred sectors are healthcare pharma and excluded sectors tobacco and i can tlerate low risk
class PortfolioState(BaseModel):
    investment_amount: float
    risk_tolerance: str = "moderate"
    investment_horizon: str = "medium"
    user_preferences: Dict = Field(default_factory=dict)
    
    allocation_decision: Any = None
    recommendations: List[Dict] = Field(default_factory=list)
    
    fundamental_analysis: Any = None
    macro_analysis: Any = None
    technical_analysis: Any = None
    volatility_liquidity_analysis: Any = None
    behavioral_psychology_analysis: Any = None
    fundamental_data: Any = None
    macro_data: Any = None
    technical_data: Any = None
    volatility_liquidity_data: Any = None
    behavioral_data: Any = None
    analysis_timings: Dict = Field(default_factory=dict)
    
    risk_management_result: Any = None
    
    messages: List = Field(default_factory=list)
    errors: Dict = Field(default_factory=dict)
