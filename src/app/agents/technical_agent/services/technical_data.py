"""
Technical data provider for fetching and calculating technical indicators
"""
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import talib

from ..models.types import TechnicalIndicators, ChartPatterns, TradingSignals, SignalType
from ..config.settings import settings

logger = logging.getLogger(__name__)

class TechnicalDataProvider:
    """Provider for technical analysis data"""
    
    def __init__(self):
        self.config = settings
    
    def get_price_data(self, ticker: str, period: str = None) -> pd.DataFrame:
        """Fetch price data for a ticker"""
        if period is None:
            period = self.config.data.default_period
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No price data available for {ticker}")
            else:
                logger.info(f"Fetched {len(data)} rows of price data for {ticker}")
            
            return data
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        if df.empty:
            return {"error": "No price data available"}
        
        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            volume = df['Volume'].values
            
            indicators = {}
            
            # Moving Averages
            for period in self.config.indicators.sma_periods:
                indicators[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            
            for period in self.config.indicators.ema_periods:
                indicators[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(
                close,
                fastperiod=self.config.indicators.macd_fast,
                slowperiod=self.config.indicators.macd_slow,
                signalperiod=self.config.indicators.macd_signal
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macdsignal
            indicators['macd_histogram'] = macdhist
            
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=self.config.indicators.rsi_period)
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=self.config.indicators.stoch_fastk,
                slowk_period=self.config.indicators.stoch_slowk,
                slowd_period=self.config.indicators.stoch_slowd
            )
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close,
                timeperiod=self.config.indicators.bollinger_period,
                nbdevup=self.config.indicators.bollinger_std,
                nbdevdn=self.config.indicators.bollinger_std
            )
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # Volume Indicators
            indicators['obv'] = talib.OBV(close, volume)
            indicators['ad'] = talib.AD(high, low, close, volume)
            
            # Other Indicators
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=self.config.indicators.williams_period)
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=self.config.indicators.cci_period)
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.config.indicators.adx_period)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.config.indicators.atr_period)
            
            # Pivot Points
            indicators['pivot_point'] = (high[-1] + low[-1] + close[-1]) / 3
            indicators['resistance_1'] = 2 * indicators['pivot_point'] - low[-1]
            indicators['support_1'] = 2 * indicators['pivot_point'] - high[-1]
            indicators['resistance_2'] = indicators['pivot_point'] + (high[-1] - low[-1])
            indicators['support_2'] = indicators['pivot_point'] - (high[-1] - low[-1])
            
            # Price vs Moving Averages
            for period in self.config.indicators.sma_periods:
                sma_key = f'sma_{period}'
                if sma_key in indicators and not np.isnan(indicators[sma_key][-1]):
                    indicators[f'price_vs_sma{period}'] = (close[-1] / indicators[sma_key][-1] - 1) * 100
                else:
                    indicators[f'price_vs_sma{period}'] = 0
            
            # Volume Ratio
            avg_volume_20 = np.mean(volume[-20:])
            indicators['volume_ratio'] = volume[-1] / avg_volume_20 if avg_volume_20 > 0 else 1
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {"error": str(e)}
    
    def detect_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect chart patterns"""
        if df.empty:
            return {"error": "No price data for pattern detection"}
        
        try:
            patterns = {}
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            open_price = df['Open'].values
            
            # Candlestick Patterns
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
            
            # Trend Patterns
            recent_highs = high[-20:]
            recent_lows = low[-20:]
            patterns['higher_highs'] = len([i for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1]]) > len(recent_highs) * 0.6
            patterns['lower_lows'] = len([i for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1]]) > len(recent_lows) * 0.6
            
            # Position vs SMA
            sma_20 = talib.SMA(close, timeperiod=20)
            patterns['above_sma20'] = close[-1] > sma_20[-1] if not np.isnan(sma_20[-1]) else False
            patterns['below_sma20'] = close[-1] < sma_20[-1] if not np.isnan(sma_20[-1]) else False
            
            # Volume Patterns
            volume = df['Volume'].values
            avg_volume = np.mean(volume[-20:])
            patterns['high_volume_breakout'] = volume[-1] > avg_volume * self.config.signals.volume_breakout_multiplier and close[-1] > open_price[-1]
            patterns['low_volume_decline'] = volume[-1] < avg_volume * self.config.signals.volume_decline_multiplier and close[-1] < open_price[-1]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return {"error": str(e)}
    
    def generate_trading_signals(self, indicators: Dict, patterns: Dict, current_price: float) -> Dict:
        """Generate trading signals from indicators and patterns"""
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
            
            # RSI Analysis
            rsi_current = indicators.get('rsi', [0])[-1] if hasattr(indicators.get('rsi', [0]), '__len__') else indicators.get('rsi', 0)
            if rsi_current < self.config.signals.rsi_oversold:
                signals["bullish_signals"].append("RSI Oversold (Bullish)")
                bullish_count += 2
            elif rsi_current > self.config.signals.rsi_overbought:
                signals["bearish_signals"].append("RSI Overbought (Bearish)")
                bearish_count += 2
            
            # MACD Analysis
            macd_current = indicators.get('macd', [0])[-1] if hasattr(indicators.get('macd', [0]), '__len__') else 0
            macd_signal_current = indicators.get('macd_signal', [0])[-1] if hasattr(indicators.get('macd_signal', [0]), '__len__') else 0
            if macd_current > macd_signal_current:
                signals["bullish_signals"].append("MACD Bullish Crossover")
                bullish_count += 1
            else:
                signals["bearish_signals"].append("MACD Bearish")
                bearish_count += 1
            
            # Moving Average Analysis
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
            
            # Pattern Analysis
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
            
            # Calculate net signal
            net_signal = bullish_count - bearish_count
            signals["strength"] = max(-100, min(100, net_signal * self.config.signals.signal_strength_multiplier))
            
            # Determine overall signal
            if net_signal >= self.config.signals.strong_buy_threshold:
                signals["overall_signal"] = "STRONG BUY"
            elif net_signal >= self.config.signals.buy_threshold:
                signals["overall_signal"] = "BUY"
            elif net_signal <= self.config.signals.strong_sell_threshold:
                signals["overall_signal"] = "STRONG SELL"
            elif net_signal <= self.config.signals.sell_threshold:
                signals["overall_signal"] = "SELL"
            else:
                signals["overall_signal"] = "HOLD"
            
            # Key levels
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