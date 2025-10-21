from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FundamentalAnalysis:
    company_financials_summary: str
    sector_strength_summary: str
    fundamental_investment_thesis: str

class FundamentalAgentState(TypedDict):
    recommendations: List[Dict]  
    fundamental_analysis: Optional[FundamentalAnalysis]
    messages: List
    financial_data: Optional[Dict]
    company_financials_analysis: Optional[str]
    sector_strength_analysis: Optional[str]
    investment_thesis_analysis: Optional[str]