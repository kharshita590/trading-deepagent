from typing import TypedDict, List, Dict
from dataclasses import dataclass
from enum import Enum

class AllocationStrategy(Enum):
    SINGLE_STOCK = "single_stock"
    MULTI_STOCK = "multi_stock"
    HYBRID = "hybrid"

@dataclass
class StockRecommendation:
    ticker: str
    company_name: str
    price: float
    allocation_percentage: float
    allocation_amount: float
    sector: str
    market_cap: str
    reasoning: str

@dataclass
class StockData:
    ticker: str
    name: str
    sector: str
    price: float
    market_cap: str

class ResearchAgentState(TypedDict):
    allocation_strategy: AllocationStrategy
    investment_amount: float
    risk_score: float
    diversification_score: float
    reasoning: str
    preferred_sectors: List[str]
    messages: List
    stock_lists: Dict[str, List[Dict]] 
    stock_prices: Dict[str, float]      
    filtered_stocks: Dict[str, List[StockData]]  
    
    recommendations: List[StockRecommendation]
