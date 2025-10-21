import os
from dataclasses import dataclass
from typing import Optional
import logging

@dataclass
class LLMConfig:
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    api_key: str = ""
    max_retries: int = 3
    timeout: int = 60

@dataclass
class DataConfig:
    default_period: str = "1y"
    cache_enabled: bool = True
    cache_ttl: int = 300  
    
@dataclass
class TechnicalIndicatorConfig:
    sma_periods: list = None
    ema_periods: list = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: int = 2
    stoch_fastk: int = 14
    stoch_slowk: int = 3
    stoch_slowd: int = 3
    atr_period: int = 14
    adx_period: int = 14
    cci_period: int = 14
    williams_period: int = 14
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]

@dataclass
class SignalConfig:
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    strong_buy_threshold: int = 3
    buy_threshold: int = 1
    strong_sell_threshold: int = -3
    sell_threshold: int = -1
    volume_breakout_multiplier: float = 1.5
    volume_decline_multiplier: float = 0.7
    signal_strength_multiplier: int = 10

@dataclass
class LoggingConfig:
    level: int = logging.INFO
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'

class Settings:
    def __init__(self):
        self.llm = LLMConfig()
        self.data = DataConfig()
        self.indicators = TechnicalIndicatorConfig()
        self.signals = SignalConfig()
        self.logging = LoggingConfig()
        
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=self.logging.level,
            format=self.logging.format,
            datefmt=self.logging.date_format
        )
    
    @classmethod
    def from_env(cls):
        settings = cls()        
        if api_key := os.getenv('GOOGLE_API_KEY'):
            settings.llm.api_key = api_key
        if model := os.getenv('LLM_MODEL'):
            settings.llm.model_name = model
        if temp := os.getenv('LLM_TEMPERATURE'):
            settings.llm.temperature = float(temp)
        
        return settings

settings = Settings()