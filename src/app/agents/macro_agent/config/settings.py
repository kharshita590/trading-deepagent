import os
from typing import Dict
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    max_retries: int = 3
    timeout: int = 30


@dataclass
class DataProviderConfig:
    interest_rate_period: str = "5d"
    economic_indicator_period: str = "30d"
    sector_performance_period: str = "30d"
    cache_ttl: int = 300 


@dataclass
class AnalysisConfig:
    min_analysis_length: int = 200
    max_analysis_length: int = 800
    include_sector_breakdown: bool = True
    include_risk_assessment: bool = True


class AppConfig:    
    llm = LLMConfig()    
    data_provider = DataProviderConfig()    
    analysis = AnalysisConfig()    
    SECTOR_ETFS: Dict[str, str] = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer Goods": "XLY",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication Services": "XLC"
    }    
    MARKET_INDICES: Dict[str, str] = {
        "sp500": "SPY",
        "volatility": "^VIX",
        "dollar_index": "DX-Y.NYB",
        "treasury_10y": "^TNX",
        "fed_rate": "^IRX"
    }    
    VOLATILITY_THRESHOLDS = {
        "low": 15,
        "moderate": 20,
        "high": 30,
        "extreme": 40
    }    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.llm.api_key:
            raise ValueError("Google API key not configured")
        return True