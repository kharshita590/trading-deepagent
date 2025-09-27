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
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VolatilityLiquidityAnalysis:
    volatility_assessment_summary: str
    liquidity_analysis_summary: str
    risk_management_recommendations: str

class VolatilityLiquidityAgentState(TypedDict):
    recommendations: List[Dict]  
    volatility_liquidity_analysis: Optional[VolatilityLiquidityAnalysis]
    messages: List

class VolatilityLiquidityDataProvider:    
    @staticmethod
    def get_extended_price_data(ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)            
            data_1y = stock.history(period="1y")  
            data_3m = stock.history(period="3mo")  
            data_1m = stock.history(period="1mo")  
            data_5d = stock.history(period="5d", interval="1h")              
            info = stock.info
            
            return {
                "data_1y": data_1y,
                "data_3m": data_3m,
                "data_1m": data_1m,
                "data_5d": data_5d,
                "info": info,
                "ticker": ticker
            }
            
        except Exception as e:
            logger.error(f"Error fetching extended data for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    @staticmethod
    def calculate_volatility_metrics(data_dict: Dict) -> Dict:
        try:
            metrics = {}            
            data_1y = data_dict.get("data_1y", pd.DataFrame())
            data_3m = data_dict.get("data_3m", pd.DataFrame())
            data_1m = data_dict.get("data_1m", pd.DataFrame())
            info = data_dict.get("info", {})
            
            if data_1y.empty:
                return {"error": "No price data available for volatility calculation"}            
            returns_1y = data_1y['Close'].pct_change().dropna()
            returns_3m = data_3m['Close'].pct_change().dropna() if not data_3m.empty else returns_1y[-63:]
            returns_1m = data_1m['Close'].pct_change().dropna() if not data_1m.empty else returns_1y[-22:]
            
            metrics['historical_vol_1y'] = returns_1y.std() * np.sqrt(252) * 100  
            metrics['historical_vol_3m'] = returns_3m.std() * np.sqrt(252) * 100
            metrics['historical_vol_1m'] = returns_1m.std() * np.sqrt(252) * 100
            
            rolling_vol_30d = returns_1y.rolling(window=30).std() * np.sqrt(252) * 100
            metrics['current_30d_vol'] = rolling_vol_30d.iloc[-1] if not rolling_vol_30d.empty else 0
            metrics['avg_30d_vol'] = rolling_vol_30d.mean() if not rolling_vol_30d.empty else 0
            metrics['max_30d_vol'] = rolling_vol_30d.max() if not rolling_vol_30d.empty else 0
            metrics['min_30d_vol'] = rolling_vol_30d.min() if not rolling_vol_30d.empty else 0
            
            if len(data_1y) > 14:
                atr = talib.ATR(data_1y['High'].values, data_1y['Low'].values, data_1y['Close'].values, timeperiod=14)
                metrics['atr_14'] = atr[-1] if not np.isnan(atr[-1]) else 0
                metrics['atr_percentage'] = (atr[-1] / data_1y['Close'].iloc[-1]) * 100 if atr[-1] > 0 else 0
            
            current_price = data_1y['Close'].iloc[-1]
            high_52w = data_1y['High'].max()
            low_52w = data_1y['Low'].min()
            
            metrics['price_range_52w'] = ((high_52w - low_52w) / low_52w) * 100
            metrics['current_vs_52w_high'] = ((current_price - high_52w) / high_52w) * 100
            metrics['current_vs_52w_low'] = ((current_price - low_52w) / low_52w) * 100
            
            if len(rolling_vol_30d) > 1:
                current_vol_percentile = stats.percentileofscore(rolling_vol_30d.dropna(), metrics['current_30d_vol'])
                metrics['volatility_percentile'] = current_vol_percentile
            else:
                metrics['volatility_percentile'] = 50
            
            metrics['beta'] = info.get('beta', 1.0)
            
            if metrics['historical_vol_1y'] < 15:
                metrics['volatility_class'] = 'Low'
            elif metrics['historical_vol_1y'] < 30:
                metrics['volatility_class'] = 'Moderate'
            elif metrics['historical_vol_1y'] < 50:
                metrics['volatility_class'] = 'High'
            else:
                metrics['volatility_class'] = 'Very High'
            
            recent_vol = metrics['current_30d_vol']
            avg_vol = metrics['avg_30d_vol']
            
            if recent_vol > avg_vol * 1.2:
                metrics['volatility_trend'] = 'Increasing'
            elif recent_vol < avg_vol * 0.8:
                metrics['volatility_trend'] = 'Decreasing'
            else:
                metrics['volatility_trend'] = 'Stable'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_liquidity_metrics(data_dict: Dict) -> Dict:
        try:
            metrics = {}            
            data_1y = data_dict.get("data_1y", pd.DataFrame())
            data_3m = data_dict.get("data_3m", pd.DataFrame())
            data_1m = data_dict.get("data_1m", pd.DataFrame())
            data_5d = data_dict.get("data_5d", pd.DataFrame())
            info = data_dict.get("info", {})
            
            if data_1y.empty:
                return {"error": "No data available for liquidity calculation"}
            
            volume_1y = data_1y['Volume']
            metrics['avg_volume_1y'] = volume_1y.mean()
            metrics['avg_volume_3m'] = data_3m['Volume'].mean() if not data_3m.empty else volume_1y[-63:].mean()
            metrics['avg_volume_1m'] = data_1m['Volume'].mean() if not data_1m.empty else volume_1y[-22:].mean()
            
            current_volume = volume_1y.iloc[-1]
            metrics['current_volume'] = current_volume
            metrics['volume_ratio_3m'] = current_volume / metrics['avg_volume_3m'] if metrics['avg_volume_3m'] > 0 else 1
            metrics['volume_ratio_1m'] = current_volume / metrics['avg_volume_1m'] if metrics['avg_volume_1m'] > 0 else 1
            
            recent_avg_volume = volume_1y[-10:].mean() 
            older_avg_volume = volume_1y[-30:-10].mean()  
            
            if recent_avg_volume > older_avg_volume * 1.15:
                metrics['volume_trend'] = 'Increasing'
            elif recent_avg_volume < older_avg_volume * 0.85:
                metrics['volume_trend'] = 'Decreasing'
            else:
                metrics['volume_trend'] = 'Stable'            
            market_cap = info.get('marketCap', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            float_shares = info.get('floatShares', shares_outstanding)
            
            metrics['market_cap'] = market_cap
            metrics['shares_outstanding'] = shares_outstanding
            metrics['float_shares'] = float_shares
            metrics['float_percentage'] = (float_shares / shares_outstanding * 100) if shares_outstanding > 0 else 100
            
            current_price = data_1y['Close'].iloc[-1]
            metrics['avg_dollar_volume_1y'] = (volume_1y * data_1y['Close']).mean()
            metrics['avg_dollar_volume_3m'] = (data_3m['Volume'] * data_3m['Close']).mean() if not data_3m.empty else (volume_1y[-63:] * data_1y['Close'][-63:]).mean()
            metrics['current_dollar_volume'] = current_volume * current_price
            
            daily_spreads = ((data_1y['High'] - data_1y['Low']) / data_1y['Close']) * 100
            metrics['avg_daily_spread'] = daily_spreads.mean()
            metrics['current_daily_spread'] = daily_spreads.iloc[-1]
            
            if float_shares > 0:
                metrics['daily_turnover'] = (current_volume / float_shares) * 100
                metrics['avg_daily_turnover'] = (metrics['avg_volume_1y'] / float_shares) * 100
            else:
                metrics['daily_turnover'] = 0
                metrics['avg_daily_turnover'] = 0
            
            if not data_1m.empty:
                vwap_1m = (data_1m['Close'] * data_1m['Volume']).sum() / data_1m['Volume'].sum()
                metrics['price_vs_vwap_1m'] = ((current_price - vwap_1m) / vwap_1m) * 100
            else:
                metrics['price_vs_vwap_1m'] = 0
            
            if not data_5d.empty and len(data_5d) > 20:
                hourly_volumes = data_5d['Volume']
                hourly_ranges = ((data_5d['High'] - data_5d['Low']) / data_5d['Close']) * 100
                
                metrics['intraday_vol_consistency'] = 1 - (hourly_volumes.std() / hourly_volumes.mean()) if hourly_volumes.mean() > 0 else 0
                metrics['intraday_spread_avg'] = hourly_ranges.mean()
            else:
                metrics['intraday_vol_consistency'] = 0.5 
                metrics['intraday_spread_avg'] = metrics['avg_daily_spread'] / 4 
            
            if metrics['avg_dollar_volume_1y'] > 100000000: 
                metrics['liquidity_class'] = 'Very High'
            elif metrics['avg_dollar_volume_1y'] > 50000000:  
                metrics['liquidity_class'] = 'High'
            elif metrics['avg_dollar_volume_1y'] > 10000000:  
                metrics['liquidity_class'] = 'Moderate'
            elif metrics['avg_dollar_volume_1y'] > 1000000:  
                metrics['liquidity_class'] = 'Low'
            else:
                metrics['liquidity_class'] = 'Very Low'
            
            if market_cap > 0 and metrics['avg_volume_1y'] > 0:
                impact_factor = 1000000 / (market_cap/1000000000 * metrics['avg_volume_1y']/1000000)
                metrics['estimated_impact_cost'] = min(impact_factor * 0.1, 5.0) 
            else:
                metrics['estimated_impact_cost'] = 2.0 
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_risk_metrics(volatility_metrics: Dict, liquidity_metrics: Dict, allocation_amount: float) -> Dict:
        try:
            risk_metrics = {}
            daily_vol = volatility_metrics.get('historical_vol_1y', 20) / np.sqrt(252) / 100
            var_95_1d = allocation_amount * 1.645 * daily_vol 
            var_99_1d = allocation_amount * 2.326 * daily_vol  
            
            risk_metrics['var_95_1day'] = var_95_1d
            risk_metrics['var_99_1day'] = var_99_1d
            risk_metrics['var_95_1day_pct'] = (var_95_1d / allocation_amount) * 100
            
            atr_pct = volatility_metrics.get('atr_percentage', 2)
            max_daily_loss = allocation_amount * (atr_pct / 100) * 2  
            risk_metrics['max_expected_daily_loss'] = max_daily_loss
            risk_metrics['max_expected_daily_loss_pct'] = (max_daily_loss / allocation_amount) * 100
            
            avg_dollar_volume = liquidity_metrics.get('avg_dollar_volume_1y', 1000000)
            position_vs_volume = (allocation_amount / avg_dollar_volume) * 100
            
            risk_metrics['position_vs_avg_volume_pct'] = position_vs_volume
            
            if position_vs_volume > 20:
                risk_metrics['liquidity_risk'] = 'Very High'
                risk_metrics['liquidity_risk_score'] = 90
            elif position_vs_volume > 10:
                risk_metrics['liquidity_risk'] = 'High'
                risk_metrics['liquidity_risk_score'] = 75
            elif position_vs_volume > 5:
                risk_metrics['liquidity_risk'] = 'Moderate'
                risk_metrics['liquidity_risk_score'] = 50
            elif position_vs_volume > 1:
                risk_metrics['liquidity_risk'] = 'Low'
                risk_metrics['liquidity_risk_score'] = 25
            else:
                risk_metrics['liquidity_risk'] = 'Very Low'
                risk_metrics['liquidity_risk_score'] = 10
            
            vol_score = min(volatility_metrics.get('historical_vol_1y', 20) * 2, 100) 
            liq_score = risk_metrics['liquidity_risk_score']
            
            risk_metrics['combined_risk_score'] = (vol_score * 0.6 + liq_score * 0.4)
            
            if risk_metrics['combined_risk_score'] > 75:
                risk_metrics['overall_risk_level'] = 'Very High'
            elif risk_metrics['combined_risk_score'] > 60:
                risk_metrics['overall_risk_level'] = 'High'
            elif risk_metrics['combined_risk_score'] > 40:
                risk_metrics['overall_risk_level'] = 'Moderate'
            elif risk_metrics['combined_risk_score'] > 25:
                risk_metrics['overall_risk_level'] = 'Low'
            else:
                risk_metrics['overall_risk_level'] = 'Very Low'
            
            if risk_metrics['combined_risk_score'] > 75:
                risk_metrics['position_size_adjustment'] = 0.5 
            elif risk_metrics['combined_risk_score'] > 60:
                risk_metrics['position_size_adjustment'] = 0.7 
            elif risk_metrics['combined_risk_score'] > 40:
                risk_metrics['position_size_adjustment'] = 0.85 
            else:
                risk_metrics['position_size_adjustment'] = 1.0  
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {"error": str(e)}

class VolatilityLiquidityAgent:
    def __init__(self, llm_model="gpt-4"):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)
        self.data_provider = VolatilityLiquidityDataProvider()

    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(VolatilityLiquidityAgentState)
        
        workflow.add_node("fetch_volatility_liquidity_data", self.fetch_volatility_liquidity_data)
        workflow.add_node("analyze_volatility_metrics", self.analyze_volatility_metrics)
        workflow.add_node("analyze_liquidity_metrics", self.analyze_liquidity_metrics)
        workflow.add_node("generate_risk_recommendations", self.generate_risk_recommendations)
        workflow.add_node("compile_volatility_liquidity_analysis", self.compile_volatility_liquidity_analysis)
        
        workflow.add_edge("fetch_volatility_liquidity_data", "analyze_volatility_metrics")
        workflow.add_edge("analyze_volatility_metrics", "analyze_liquidity_metrics")
        workflow.add_edge("analyze_liquidity_metrics", "generate_risk_recommendations")
        workflow.add_edge("generate_risk_recommendations", "compile_volatility_liquidity_analysis")
        workflow.add_edge("compile_volatility_liquidity_analysis", END)
        
        workflow.set_entry_point("fetch_volatility_liquidity_data")
        
        return workflow

    async def fetch_volatility_liquidity_data(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        """Fetch comprehensive volatility and liquidity data"""
        logger.info("Fetching volatility and liquidity data")
        
        recommendations = state.get("recommendations", [])
        vol_liq_data = {}
        
        for rec in recommendations:
            ticker = rec.get("ticker", "")
            allocation_amount = rec.get("allocation_amount", 0)            
            data_dict = self.data_provider.get_extended_price_data(ticker)
            
            if "error" not in data_dict:
                volatility_metrics = self.data_provider.calculate_volatility_metrics(data_dict)
                
                liquidity_metrics = self.data_provider.calculate_liquidity_metrics(data_dict)
                
                risk_metrics = self.data_provider.calculate_risk_metrics(
                    volatility_metrics, liquidity_metrics, allocation_amount
                )
                
                vol_liq_data[ticker] = {
                    "recommendation": rec,
                    "volatility_metrics": volatility_metrics,
                    "liquidity_metrics": liquidity_metrics,
                    "risk_metrics": risk_metrics
                }
            else:
                logger.warning(f"No data available for {ticker}")
                vol_liq_data[ticker] = {"error": data_dict.get("error", "Unknown error")}
        
        state["vol_liq_data"] = vol_liq_data
        state["messages"].append(AIMessage(content=f"Fetched volatility/liquidity data for {len(recommendations)} stocks"))
        
        return state

    async def analyze_volatility_metrics(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Analyzing volatility metrics")
        
        vol_liq_data = state.get("vol_liq_data", {})
        
        volatility_prompt = f"""
        Analyze the volatility characteristics of this portfolio:
        
        {self._format_volatility_data(vol_liq_data)}
        
        Provide a comprehensive volatility assessment summary (4-5 sentences) covering:
        1. Overall portfolio volatility profile and classification
        2. Individual stock volatility analysis and risk levels
        3. Historical vs current volatility trends
        4. Volatility clustering and correlation patterns
        5. Market regime analysis (low vol vs high vol environment)
        6. Volatility-adjusted position sizing recommendations
        
        Focus on:
        - Historical volatility levels (1Y, 3M, 1M comparison)
        - ATR analysis for intraday risk assessment
        - Beta analysis for market sensitivity
        - Volatility percentiles and extremes
        - Recent volatility trend changes
        - Portfolio volatility diversification benefits
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=volatility_prompt)])
            state["volatility_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            state["volatility_analysis"] = f"Volatility analysis unavailable: {str(e)}"
        
        return state

    async def analyze_liquidity_metrics(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Analyzing liquidity metrics")
        
        vol_liq_data = state.get("vol_liq_data", {})
        
        liquidity_prompt = f"""
        Analyze the liquidity characteristics of this portfolio:
        
        {self._format_liquidity_data(vol_liq_data)}
        
        Provide a comprehensive liquidity analysis summary (4-5 sentences) covering:
        1. Overall portfolio liquidity profile and accessibility
        2. Individual stock liquidity analysis and trading ease
        3. Volume trends and market depth assessment
        4. Impact cost estimation for position sizes
        5. Liquidity risk during market stress scenarios
        6. Optimal execution strategies for each position
        
        Focus on:
        - Average daily volume and dollar volume analysis
        - Volume consistency and seasonal patterns
        - Market cap and float analysis
        - Bid-ask spread proxies and trading costs
        - Turnover ratios and market maker presence
        - Position size vs daily volume constraints
        - Emergency exit liquidity assessment
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=liquidity_prompt)])
            state["liquidity_analysis"] = response.content
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            state["liquidity_analysis"] = f"Liquidity analysis unavailable: {str(e)}"
        
        return state

    async def generate_risk_recommendations(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Generating risk management recommendations")
        
        vol_liq_data = state.get("vol_liq_data", {})
        
        risk_prompt = f"""
        Generate risk management recommendations for this portfolio based on volatility and liquidity analysis:
        
        {self._format_risk_data(vol_liq_data)}
        
        CONTEXT:
        Volatility Analysis: {state.get("volatility_analysis", "Not available")}
        Liquidity Analysis: {state.get("liquidity_analysis", "Not available")}
        
        Provide comprehensive risk management recommendations (4-5 sentences) covering:
        1. Position sizing adjustments based on vol/liquidity profiles
        2. Stop-loss and risk management level recommendations
        3. Portfolio rebalancing frequency suggestions
        4. Market timing and entry/exit strategy guidance
        5. Stress testing and scenario planning recommendations
        6. Emergency liquidity and exit planning
        
        Consider:
        - VaR calculations and maximum expected losses
        - Liquidity-adjusted position limits
        - Volatility-based stop losses
        - Optimal order execution strategies
        - Portfolio correlation and concentration risks
        - Market regime change preparedness
        - Cost-benefit analysis of risk vs return
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=risk_prompt)])
            state["risk_recommendations"] = response.content
        except Exception as e:
            logger.error(f"Error in risk recommendations: {e}")
            state["risk_recommendations"] = f"Risk recommendations unavailable: {str(e)}"
        
        return state

    async def compile_volatility_liquidity_analysis(self, state: VolatilityLiquidityAgentState) -> VolatilityLiquidityAgentState:
        logger.info("Compiling volatility/liquidity analysis")
        
        vol_liq_analysis = VolatilityLiquidityAnalysis(
            volatility_assessment_summary=state.get("volatility_analysis", "Analysis unavailable"),
            liquidity_analysis_summary=state.get("liquidity_analysis", "Analysis unavailable"),
            risk_management_recommendations=state.get("risk_recommendations", "Analysis unavailable")
        )
        
        state["volatility_liquidity_analysis"] = vol_liq_analysis
        state["messages"].append(AIMessage(content="Completed comprehensive volatility/liquidity analysis"))
        
        return state

    def _format_volatility_data(self, vol_liq_data: Dict) -> str:
        formatted = []
        for ticker, data in vol_liq_data.items():
            if "error" in data:
                continue
                
            vol_metrics = data.get("volatility_metrics", {})
            rec = data.get("recommendation", {})
            
            formatted.append(f"""
STOCK: {rec.get('company_name', 'N/A')} ({ticker})
- Allocation: {rec.get('allocation_percentage', 0):.1f}% (₹{rec.get('allocation_amount', 0):,.0f})
- Historical Volatility: 1Y: {vol_metrics.get('historical_vol_1y', 0):.1f}% | 3M: {vol_metrics.get('historical_vol_3m', 0):.1f}% | 1M: {vol_metrics.get('historical_vol_1m', 0):.1f}%
- Current 30D Vol: {vol_metrics.get('current_30d_vol', 0):.1f}% (Percentile: {vol_metrics.get('volatility_percentile', 50):.0f}%)
- ATR: {vol_metrics.get('atr_14', 0):.2f} ({vol_metrics.get('atr_percentage', 0):.2f}% of price)
- Volatility Class: {vol_metrics.get('volatility_class', 'Unknown')} | Trend: {vol_metrics.get('volatility_trend', 'Unknown')}
- Beta: {vol_metrics.get('beta', 1.0):.2f} | 52W Range: {vol_metrics.get('price_range_52w', 0):.1f}%
- Current vs 52W High: {vol_metrics.get('current_vs_52w_high', 0):+.1f}% | vs Low: {vol_metrics.get('current_vs_52w_low', 0):+.1f}%
""")
        
        return '\n'.join(formatted)

    def _format_liquidity_data(self, vol_liq_data: Dict) -> str:
        """Format liquidity data for analysis"""
        formatted = []
        for ticker, data in vol_liq_data.items():
            if "error" in data:
                continue
                
            liq_metrics = data.get("liquidity_metrics", {})
            rec = data.get("recommendation", {})
            
            formatted.append(f"""
{ticker}: {rec.get('company_name', 'N/A')} ({rec.get('allocation_percentage', 0):.1f}% allocation)
- Average Volume: 1Y: {liq_metrics.get('avg_volume_1y', 0):,.0f} | 3M: {liq_metrics.get('avg_volume_3m', 0):,.0f} | Current: {liq_metrics.get('current_volume', 0):,.0f}
- Volume Ratios: 3M: {liq_metrics.get('volume_ratio_3m', 1):.2f}x | 1M: {liq_metrics.get('volume_ratio_1m', 1):.2f}x | Trend: {liq_metrics.get('volume_trend', 'Unknown')}
- Dollar Volume: 1Y Avg: ₹{liq_metrics.get('avg_dollar_volume_1y', 0):,.0f} | Current: ₹{liq_metrics.get('current_dollar_volume', 0):,.0f}
- Market Cap: ₹{liq_metrics.get('market_cap', 0):,.0f} | Float: {liq_metrics.get('float_percentage', 100):.1f}%
- Daily Turnover: Current: {liq_metrics.get('daily_turnover', 0):.3f}% | Average: {liq_metrics.get('avg_daily_turnover', 0):.3f}%
- Liquidity Class: {liq_metrics.get('liquidity_class', 'Unknown')} | Avg Spread: {liq_metrics.get('avg_daily_spread', 0):.2f}%
- Price vs VWAP (1M): {liq_metrics.get('price_vs_vwap_1m', 0):+.2f}% | Impact Cost Est: {liq_metrics.get('estimated_impact_cost', 0):.2f}%
""")
        
        return '\n'.join(formatted)

    def _format_risk_data(self, vol_liq_data: Dict) -> str:
        """Format risk data for analysis"""
        formatted = []
        total_portfolio_value = sum(data.get("recommendation", {}).get("allocation_amount", 0) for data in vol_liq_data.values() if "error" not in data)
        
        formatted.append(f"PORTFOLIO RISK SUMMARY (Total: ₹{total_portfolio_value:,.0f}):\n")
        
        for ticker, data in vol_liq_data.items():
            if "error" in data:
                continue
                
            risk_metrics = data.get("risk_metrics", {})
            rec = data.get("recommendation", {})
            
            formatted.append(f"""
{ticker}: {rec.get('company_name', 'N/A')}
- Position: ₹{rec.get('allocation_amount', 0):,.0f} ({rec.get('allocation_percentage', 0):.1f}% of portfolio)
- VaR (95%, 1-day): ₹{risk_metrics.get('var_95_1day', 0):,.0f} ({risk_metrics.get('var_95_1day_pct', 0):.2f}% of position)
- Max Expected Daily Loss: ₹{risk_metrics.get('max_expected_daily_loss', 0):,.0f} ({risk_metrics.get('max_expected_daily_loss_pct', 0):.2f}%)
- Position vs Avg Volume: {risk_metrics.get('position_vs_avg_volume_pct', 0):.1f}%
- Liquidity Risk: {risk_metrics.get('liquidity_risk', 'Unknown')} (Score: {risk_metrics.get('liquidity_risk_score', 0)}/100)
- Overall Risk Level: {risk_metrics.get('overall_risk_level', 'Unknown')} (Score: {risk_metrics.get('combined_risk_score', 0):.0f}/100)
- Recommended Position Adjustment: {risk_metrics.get('position_size_adjustment', 1.0):.0%} of current size
""")
        
        return '\n'.join(formatted)
