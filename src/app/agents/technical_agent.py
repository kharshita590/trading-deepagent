from typing import TypedDict, List, Dict, Optional
import json
from dataclasses import dataclass
import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from langgraph import StateGraph, END
from langgraph.graph.message import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import talib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TechnicalAnalysis:
    technical_indicators_summary: str
    pattern_detection_summary: str
    technical_trading_signals: str

class TechnicalAgentState(TypedDict):
    recommendations: List[Dict]  
    technical_analysis: Optional[TechnicalAnalysis]
    messages: List

class TechnicalDataProvider:
    @staticmethod
    def get_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
        if df.empty:
            return {"error": "No price data available"}
        
        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            volume = df['Volume'].values
            
            indicators = {}
            
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['sma_200'] = talib.SMA(close, timeperiod=200)
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)
            
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd
            indicators['macd_signal'] = macdsignal
            indicators['macd_histogram'] = macdhist
            
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd
            
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            indicators['obv'] = talib.OBV(close, volume)
            indicators['ad'] = talib.AD(high, low, close, volume)
            
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
            
            indicators['pivot_point'] = (high[-1] + low[-1] + close[-1]) / 3
            indicators['resistance_1'] = 2 * indicators['pivot_point'] - low[-1]
            indicators['support_1'] = 2 * indicators['pivot_point'] - high[-1]
            indicators['resistance_2'] = indicators['pivot_point'] + (high[-1] - low[-1])
            indicators['support_2'] = indicators['pivot_point'] - (high[-1] - low[-1])
            
            indicators['price_vs_sma20'] = (close[-1] / indicators['sma_20'][-1] - 1) * 100 if not np.isnan(indicators['sma_20'][-1]) else 0
            indicators['price_vs_sma50'] = (close[-1] / indicators['sma_50'][-1] - 1) * 100 if not np.isnan(indicators['sma_50'][-1]) else 0
            indicators['price_vs_sma200'] = (close[-1] / indicators['sma_200'][-1] - 1) * 100 if not np.isnan(indicators['sma_200'][-1]) else 0
            
            avg_volume_20 = np.mean(volume[-20:])
            indicators['volume_ratio'] = volume[-1] / avg_volume_20 if avg_volume_20 > 0 else 1
            
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14) 
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> Dict:
        if df.empty:
            return {"error": "No price data for pattern detection"}
        
        try:
            patterns = {}
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            open_price = df['Open'].values
            
            patterns['doji'] = talib.CDLDOJI(open_price, high, low, close)
            patterns['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
            patterns['engulfing_bullish'] = talib.CDLENGULFING(open_price, high, low, close)
            patterns['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
            patterns['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
            patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)
            patterns['three_black_crows'] = talib.CDL3BLACKCROWS(open_price, high, low, close)
            patterns['hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)
            patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_price, high, low, close)
            
            recent_highs = high[-20:]
            recent_lows = low[-20:]
            patterns['higher_highs'] = len([i for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1]]) > len(recent_highs) * 0.6
            patterns['lower_lows'] = len([i for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1]]) > len(recent_lows) * 0.6
            
            sma_20 = talib.SMA(close, timeperiod=20)
            patterns['above_sma20'] = close[-1] > sma_20[-1] if not np.isnan(sma_20[-1]) else False
            patterns['below_sma20'] = close[-1] < sma_20[-1] if not np.isnan(sma_20[-1]) else False
            
            volume = df['Volume'].values
            avg_volume = np.mean(volume[-20:])
            patterns['high_volume_breakout'] = volume[-1] > avg_volume * 1.5 and close[-1] > open_price[-1]
            patterns['low_volume_decline'] = volume[-1] < avg_volume * 0.7 and close[-1] < open_price[-1]
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def generate_trading_signals(indicators: Dict, patterns: Dict, current_price: float) -> Dict:
        try:
            signals = {
                "overall_signal": "NEUTRAL",
                "strength": 0, 
                "bullish_signals": [],
                "bearish_signals": [],
                "key_levels": {}
            }
            
            bullish_count = 0
            bearish_count = 0
            
            rsi_current = indicators.get('rsi', [0])[-1] if hasattr(indicators.get('rsi', [0]), '__len__') else indicators.get('rsi', 0)
            if rsi_current < 30:
                signals["bullish_signals"].append("RSI Oversold (Bullish)")
                bullish_count += 2
            elif rsi_current > 70:
                signals["bearish_signals"].append("RSI Overbought (Bearish)")
                bearish_count += 2
            
            macd_current = indicators.get('macd', [0])[-1] if hasattr(indicators.get('macd', [0]), '__len__') else 0
            macd_signal_current = indicators.get('macd_signal', [0])[-1] if hasattr(indicators.get('macd_signal', [0]), '__len__') else 0
            if macd_current > macd_signal_current:
                signals["bullish_signals"].append("MACD Bullish Crossover")
                bullish_count += 1
            else:
                signals["bearish_signals"].append("MACD Bearish")
                bearish_count += 1
            
            if indicators.get('price_vs_sma20', 0) > 0:
                signals["bullish_signals"].append("Price Above SMA20")
                bullish_count += 1
            else:
                signals["bearish_signals"].append("Price Below SMA20")
                bearish_count += 1
            
            if indicators.get('price_vs_sma50', 0) > 2:
                signals["bullish_signals"].append("Strong Uptrend (SMA50)")
                bullish_count += 1
            elif indicators.get('price_vs_sma50', 0) < -2:
                signals["bearish_signals"].append("Strong Downtrend (SMA50)")
                bearish_count += 1
            
            if patterns.get('higher_highs', False):
                signals["bullish_signals"].append("Higher Highs Pattern")
                bullish_count += 1
            if patterns.get('lower_lows', False):
                signals["bearish_signals"].append("Lower Lows Pattern")
                bearish_count += 1
            
            if hasattr(patterns.get('hammer', []), '__len__') and len(patterns.get('hammer', [])) > 0 and patterns['hammer'][-1] > 0:
                signals["bullish_signals"].append("Hammer Pattern")
                bullish_count += 1
            if hasattr(patterns.get('shooting_star', []), '__len__') and len(patterns.get('shooting_star', [])) > 0 and patterns['shooting_star'][-1] > 0:
                signals["bearish_signals"].append("Shooting Star Pattern")
                bearish_count += 1
            
            if patterns.get('high_volume_breakout', False):
                signals["bullish_signals"].append("High Volume Breakout")
                bullish_count += 2
            
            net_signal = bullish_count - bearish_count
            signals["strength"] = max(-100, min(100, net_signal * 10))
            
            if net_signal >= 3:
                signals["overall_signal"] = "STRONG BUY"
            elif net_signal >= 1:
                signals["overall_signal"] = "BUY"
            elif net_signal <= -3:
                signals["overall_signal"] = "STRONG SELL"
            elif net_signal <= -1:
                signals["overall_signal"] = "SELL"
            else:
                signals["overall_signal"] = "HOLD"
            
            signals["key_levels"] = {
                "support_1": indicators.get('support_1', current_price * 0.95),
                "support_2": indicators.get('support_2', current_price * 0.90),
                "resistance_1": indicators.get('resistance_1', current_price * 1.05),
                "resistance_2": indicators.get('resistance_2', current_price * 1.10),
                "pivot_point": indicators.get('pivot_point', current_price)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {"error": str(e)}

class TechnicalAgent:
    def __init__(self, llm_model="gpt-4"):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)
        self.data_provider = TechnicalDataProvider()

    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(TechnicalAgentState)
        
        workflow.add_node("fetch_technical_data", self.fetch_technical_data)
        workflow.add_node("analyze_technical_indicators", self.analyze_technical_indicators)
        workflow.add_node("detect_chart_patterns", self.detect_chart_patterns)
        workflow.add_node("generate_trading_signals", self.generate_trading_signals)
        workflow.add_node("compile_technical_analysis", self.compile_technical_analysis)
        
        workflow.add_edge("fetch_technical_data", "analyze_technical_indicators")
        workflow.add_edge("analyze_technical_indicators", "detect_chart_patterns")
        workflow.add_edge("detect_chart_patterns", "generate_trading_signals")
        workflow.add_edge("generate_trading_signals", "compile_technical_analysis")
        workflow.add_edge("compile_technical_analysis", END)
        
        workflow.set_entry_point("fetch_technical_data")
        
        return workflow

    async def fetch_technical_data(self, state: TechnicalAgentState) -> TechnicalAgentState:
        logger.info("Fetching technical data for all recommendations")
        
        recommendations = state.get("recommendations", [])
        technical_data = {}
        
        for rec in recommendations:
            ticker = rec.get("ticker", "")
            
            price_data = self.data_provider.get_price_data(ticker, period="1y")
            
            if not price_data.empty:
                indicators = self.data_provider.calculate_technical_indicators(price_data)
                
                patterns = self.data_provider.detect_chart_patterns(price_data)
                
                current_price = price_data['Close'].iloc[-1] if not price_data.empty else 0
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
                logger.warning(f"No price data available for {ticker}")
                technical_data[ticker] = {"error": f"No price data for {ticker}"}
        
        state["technical_data"] = technical_data
        state["messages"].append(AIMessage(content=f"Fetched technical data for {len(recommendations)} stocks"))
        
        return state

    async def analyze_technical_indicators(self, state: TechnicalAgentState) -> TechnicalAgentState:
        logger.info("Analyzing technical indicators")
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
        except Exception as e:
            logger.error(f"Error in indicators analysis: {e}")
            state["indicators_analysis"] = f"Technical indicators analysis unavailable: {str(e)}"
        
        return state

    async def detect_chart_patterns(self, state: TechnicalAgentState) -> TechnicalAgentState:
        logger.info("Detecting chart patterns")
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
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            state["patterns_analysis"] = f"Chart pattern analysis unavailable: {str(e)}"
        
        return state

    async def generate_trading_signals(self, state: TechnicalAgentState) -> TechnicalAgentState:
        logger.info("Generating technical trading signals")
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
        except Exception as e:
            logger.error(f"Error in signals generation: {e}")
            state["signals_analysis"] = f"Technical signals analysis unavailable: {str(e)}"
        
        return state

    async def compile_technical_analysis(self, state: TechnicalAgentState) -> TechnicalAgentState:
        logger.info("Compiling technical analysis")
        technical_analysis = TechnicalAnalysis(
            technical_indicators_summary=state.get("indicators_analysis", "Analysis unavailable"),
            pattern_detection_summary=state.get("patterns_analysis", "Analysis unavailable"),
            technical_trading_signals=state.get("signals_analysis", "Analysis unavailable")
        )
        
        state["technical_analysis"] = technical_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive technical analysis"))
        
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
