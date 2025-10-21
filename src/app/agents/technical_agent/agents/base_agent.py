import logging
from typing import Dict, List
from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..models.types import (
    TechnicalAgentState, StockRecommendation, StockTechnicalData,
    TechnicalAnalysis, AgentResponse
)
from ..services.technical_data import TechnicalDataProvider
from ..config.settings import settings

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm.model_name,
            temperature=settings.llm.temperature,
            google_api_key=settings.llm.api_key
        )
        logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    async def execute(self, state: TechnicalAgentState) -> TechnicalAgentState:
        """Execute agent logic"""
        pass
    
    def log_info(self, message: str):
        """Log info message"""
        logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        logger.error(f"[{self.name}] {message}")


class DataFetcherAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataFetcherAgent")
        self.data_provider = TechnicalDataProvider()
    async def execute(self, state: TechnicalAgentState) -> TechnicalAgentState:
        self.log_info("Fetching technical data for all recommendations")
        
        recommendations = state.get("recommendations", [])
        technical_data = {}
        
        for rec in recommendations:
            ticker = rec.get("ticker", "")
            self.log_info(f"Processing {ticker}")            
            price_data = self.data_provider.get_price_data(ticker)
            
            if not price_data.empty:
                indicators = self.data_provider.calculate_technical_indicators(price_data)                
                patterns = self.data_provider.detect_chart_patterns(price_data)                
                current_price = price_data['Close'].iloc[-1]
                signals = self.data_provider.generate_trading_signals(indicators, patterns, current_price)                
                technical_data[ticker] = {
                    "recommendation": rec,
                    "price_data": price_data,
                    "indicators": indicators,
                    "patterns": patterns,
                    "signals": signals,
                    "current_price": current_price,
                    "price_change_1d": ((current_price / price_data['Close'].iloc[-2]) - 1) * 100 if len(price_data) > 1 else 0,
                    "price_change_5d": ((current_price / price_data['Close'].iloc[-6]) - 1) * 100 if len(price_data) > 5 else 0,
                    "price_change_1m": ((current_price / price_data['Close'].iloc[-22]) - 1) * 100 if len(price_data) > 22 else 0
                }
            else:
                self.log_error(f"No price data available for {ticker}")
                technical_data[ticker] = {"error": f"No price data for {ticker}"}
        
        state["technical_data"] = technical_data
        state["messages"].append(AIMessage(content=f"Fetched technical data for {len(recommendations)} stocks"))
        
        self.log_info(f"Successfully processed {len(technical_data)} stocks")
        return state

class IndicatorAnalysisAgent(BaseAgent):    
    def __init__(self):
        super().__init__("IndicatorAnalysisAgent")
    
    async def execute(self, state: TechnicalAgentState) -> TechnicalAgentState:
        self.log_info("Analyzing technical indicators")
        technical_data = state.get("technical_data", {})
        
        indicators_prompt = f"""
Analyze the technical indicators for this portfolio of stocks:

{self._format_indicators_data(technical_data)}

Provide a comprehensive technical indicators summary (4-5 sentences) covering:
1. Overall technical health of the portfolio
2. Key technical indicator signals (RSI, MACD, Moving Averages)
3. Momentum and trend analysis across stocks
4. Support and resistance level analysis
5. Volume analysis and confirmation signals
6. Technical divergences or confirmations

Focus on:
- RSI levels (overbought/oversold conditions)
- MACD crossovers and momentum
- Moving average alignments and trends
- Bollinger Band positions
- Volume patterns and breakouts
- ADX trend strength indicators
"""
        try:
            response = await self.llm.ainvoke([HumanMessage(content=indicators_prompt)])
            state["indicators_analysis"] = response.content
            state["messages"].append(AIMessage(content="Completed technical indicators analysis"))
            self.log_info("Indicators analysis completed")
        except Exception as e:
            self.log_error(f"Error in indicators analysis: {e}")
            state["indicators_analysis"] = f"Technical indicators analysis unavailable: {str(e)}"
        
        return state
    
    def _format_indicators_data(self, technical_data: Dict) -> str:
        formatted = []
        for ticker, data in technical_data.items():
            if "error" in data:
                continue
            indicators = data.get("indicators", {})
            rec = data.get("recommendation", {})
            current_price = data.get("current_price", 0)
            
            formatted.append(f"""
STOCK: {rec.get('company_name', 'N/A')} ({ticker})
- Current Price: ₹{current_price:.2f} | Allocation: {rec.get('allocation_percentage', 0):.1f}%
- Price vs MAs: SMA20: {indicators.get('price_vs_sma20', 0):+.1f}% | SMA50: {indicators.get('price_vs_sma50', 0):+.1f}% | SMA200: {indicators.get('price_vs_sma200', 0):+.1f}%
- RSI: {indicators.get('rsi', [0])[-1] if hasattr(indicators.get('rsi', [0]), '__len__') else 0:.1f}
- MACD: {indicators.get('macd', [0])[-1] if hasattr(indicators.get('macd', [0]), '__len__') else 0:.4f}
- Volume Ratio: {indicators.get('volume_ratio', 1):.2f}x
- ATR: {indicators.get('atr', [0])[-1] if hasattr(indicators.get('atr', [0]), '__len__') else 0:.2f}
- Support/Resistance: S1: ₹{indicators.get('support_1', 0):.1f} | R1: ₹{indicators.get('resistance_1', 0):.1f}
""")      
        return '\n'.join(formatted)


class PatternAnalysisAgent(BaseAgent):    
    def __init__(self):
        super().__init__("PatternAnalysisAgent")
    
    async def execute(self, state: TechnicalAgentState) -> TechnicalAgentState:
        self.log_info("Detecting chart patterns")
        
        technical_data = state.get("technical_data", {})
        
        patterns_prompt = f"""
Analyze chart patterns and formations for this portfolio:

{self._format_patterns_data(technical_data)}

Provide a comprehensive pattern detection summary (4-5 sentences) covering:
1. Major chart patterns identified across the portfolio
2. Candlestick pattern analysis and significance
3. Trend patterns (higher highs, lower lows, consolidation)
4. Breakout and breakdown patterns
5. Reversal vs continuation pattern analysis
6. Pattern reliability and confirmation signals

Focus on:
- Bullish patterns (hammer, morning star, engulfing)
- Bearish patterns (shooting star, evening star, hanging man)
- Trend continuation patterns
- Support/resistance breakouts
- Volume confirmation of patterns
- Pattern completion and target levels
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=patterns_prompt)])
            state["patterns_analysis"] = response.content
            state["messages"].append(AIMessage(content="Completed chart pattern analysis"))
            self.log_info("Pattern analysis completed")
        except Exception as e:
            self.log_error(f"Error in pattern analysis: {e}")
            state["patterns_analysis"] = f"Chart pattern analysis unavailable: {str(e)}"
        
        return state
    
    def _format_patterns_data(self, technical_data: Dict) -> str:
        formatted = []
        for ticker, data in technical_data.items():
            if "error" in data:
                continue
            patterns = data.get("patterns", {})
            rec = data.get("recommendation", {})
            price_changes = f"1D: {data.get('price_change_1d', 0):+.1f}% | 5D: {data.get('price_change_5d', 0):+.1f}% | 1M: {data.get('price_change_1m', 0):+.1f}%"
            
            recent_patterns = []
            pattern_names = ['doji', 'hammer', 'shooting_star', 'engulfing_bullish', 'morning_star', 'evening_star']
            for pattern in pattern_names:
                if pattern in patterns and hasattr(patterns[pattern], '__len__') and len(patterns[pattern]) > 0:
                    if patterns[pattern][-1] != 0:
                        recent_patterns.append(f"{pattern.replace('_', ' ').title()}: {patterns[pattern][-1]}")
            
            formatted.append(f"""
{ticker}: {rec.get('company_name', 'N/A')}
- Price Performance: {price_changes}
- Trend Patterns: Higher Highs: {patterns.get('higher_highs', False)} | Lower Lows: {patterns.get('lower_lows', False)}
- Position vs SMA20: {'Above' if patterns.get('above_sma20', False) else 'Below'}
- Volume Patterns: High Vol Breakout: {patterns.get('high_volume_breakout', False)} | Low Vol Decline: {patterns.get('low_volume_decline', False)}
- Recent Candlestick Patterns: {', '.join(recent_patterns) if recent_patterns else 'None significant'}
""")
        
        return '\n'.join(formatted)


class SignalGenerationAgent(BaseAgent):    
    def __init__(self):
        super().__init__("SignalGenerationAgent")
    
    async def execute(self, state: TechnicalAgentState) -> TechnicalAgentState:
        self.log_info("Generating technical trading signals")
        
        technical_data = state.get("technical_data", {})
        
        signals_prompt = f"""
Generate technical trading signals and recommendations for this portfolio:

{self._format_signals_data(technical_data)}

TECHNICAL CONTEXT:
Indicators Analysis: {state.get("indicators_analysis", "Not available")}
Patterns Analysis: {state.get("patterns_analysis", "Not available")}

Provide comprehensive technical trading signals summary (4-5 sentences) covering:
1. Overall technical signal for the portfolio (bullish/bearish/neutral)
2. Individual stock technical recommendations
3. Entry and exit level suggestions
4. Risk management based on technical levels
5. Time horizon for technical signals
6. Confidence levels and signal strength

Consider:
- Signal convergence/divergence across indicators
- Risk-reward ratios at current levels
- Stop-loss and take-profit recommendations
- Technical momentum sustainability
- Market timing and entry strategies
- Portfolio technical diversification
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=signals_prompt)])
            state["signals_analysis"] = response.content
            state["messages"].append(AIMessage(content="Completed trading signals generation"))
            self.log_info("Signals generation completed")
        except Exception as e:
            self.log_error(f"Error in signals generation: {e}")
            state["signals_analysis"] = f"Technical signals analysis unavailable: {str(e)}"
        
        return state
    
    def _format_signals_data(self, technical_data: Dict) -> str:
        formatted = []
        total_signals = {"BUY": 0, "SELL": 0, "HOLD": 0, "STRONG BUY": 0, "STRONG SELL": 0}
        
        for ticker, data in technical_data.items():
            if "error" in data:
                continue
            
            signals = data.get("signals", {})
            rec = data.get("recommendation", {})
            
            overall_signal = signals.get("overall_signal", "HOLD")
            strength = signals.get("strength", 0)
            total_signals[overall_signal] = total_signals.get(overall_signal, 0) + 1
            
            bullish_signals = signals.get("bullish_signals", [])
            bearish_signals = signals.get("bearish_signals", [])
            key_levels = signals.get("key_levels", {})
            
            formatted.append(f"""
{ticker}: {rec.get('company_name', 'N/A')} ({rec.get('allocation_percentage', 0):.1f}% allocation)
- Technical Signal: {overall_signal} (Strength: {strength:+d}/100)
- Bullish Factors: {', '.join(bullish_signals[:3])} {'...' if len(bullish_signals) > 3 else ''}
- Bearish Factors: {', '.join(bearish_signals[:3])} {'...' if len(bearish_signals) > 3 else ''}
- Key Levels: Support ₹{key_levels.get('support_1', 0):.1f} | Resistance ₹{key_levels.get('resistance_1', 0):.1f} | Pivot ₹{key_levels.get('pivot_point', 0):.1f}
""")
        
        portfolio_signal_summary = f"\nPORTFOLIO TECHNICAL SIGNALS DISTRIBUTION: {dict(total_signals)}"
        formatted.append(portfolio_signal_summary)
        
        return '\n'.join(formatted)


class CompilationAgent(BaseAgent):    
    def __init__(self):
        super().__init__("CompilationAgent")
    
    async def execute(self, state: TechnicalAgentState) -> TechnicalAgentState:
        self.log_info("Compiling technical analysis")
        
        technical_analysis = TechnicalAnalysis(
            technical_indicators_summary=state.get("indicators_analysis", "Analysis unavailable"),
            pattern_detection_summary=state.get("patterns_analysis", "Analysis unavailable"),
            technical_trading_signals=state.get("signals_analysis", "Analysis unavailable")
        )
        state["technical_analysis"] = technical_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive technical analysis"))
        self.log_info("Technical analysis compilation completed")
        return state