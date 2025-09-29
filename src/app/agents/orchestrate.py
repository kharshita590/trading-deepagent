from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import asdict

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


from app.agents.investment_allocation import InvestmentAllocationAgent, AllocationStrategy
from app.agents.research_agent import ResearchAgent, StockRecommendation
from app.agents.macro_agent import MacroAgent
from app.agents.fundamental_agent import FundamentalAgent
from app.agents.technical_agent import TechnicalAgent
from app.agents.volatility_agent import VolatilityLiquidityAgent
from app.agents.behavioral_agent import BehavioralPsychologyAgent
from app.agents.risk_management_agent import RiskManagementAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MacroAnalysis:
    economic_score: float
    interest_rate_impact: float
    global_events_score: float
    reasoning: str


@dataclass
class FundamentalAnalysis:
    financial_score: float
    sector_strength: float
    pe_ratio: float
    debt_ratio: float
    reasoning: str


@dataclass
class TechnicalAnalysis:
    trend_score: float
    momentum_indicators: Dict[str, float]
    support_resistance: Dict[str, float]
    pattern_detected: str
    reasoning: str


@dataclass
class VolatilityAnalysis:
    volatility_score: float
    liquidity_score: float
    risk_rating: str
    reasoning: str


@dataclass
class BehavioralAnalysis:
    sentiment_score: float
    market_psychology: str
    risk_factors: List[str]
    reasoning: str


@dataclass
class RiskManagement:
    stop_loss_price: float
    take_profit_price: float
    position_size: float
    risk_reward_ratio: float
    max_loss_amount: float
    reasoning: str


@dataclass
class FinalRecommendation:
    selected_stocks: List[StockRecommendation]
    total_score: float
    confidence_level: float
    macro_analysis: MacroAnalysis
    fundamental_analysis: FundamentalAnalysis
    technical_analysis: TechnicalAnalysis
    volatility_analysis: VolatilityAnalysis
    behavioral_analysis: BehavioralAnalysis
    risk_management: RiskManagement
    reasoning: str


class InvestmentWorkflowState(TypedDict):
    investment_amount: float
    user_preferences: Dict[str, Any]
    
    allocation_strategy: AllocationStrategy
    preferred_sectors: List[str]
    risk_tolerance: str
    
    stock_recommendations: List[StockRecommendation]
    
    macro_analysis: Optional[MacroAnalysis]
    fundamental_analysis: Optional[FundamentalAnalysis]
    technical_analysis: Optional[TechnicalAnalysis]
    volatility_analysis: Optional[VolatilityAnalysis]
    behavioral_analysis: Optional[BehavioralAnalysis]
    
    final_recommendation: Optional[FinalRecommendation]
    
    risk_management: Optional[RiskManagement]
    
    messages: List[Any]
    current_step: str
    analysis_complete: bool
    errors: List[str]

class InvestmentOrchestrator:
    def __init__(self, api_keys: Dict[str, str], llm_model: str = "gemini-1.5-pro"):
        self.api_keys = api_keys
        self.llm_model = llm_model
        
        self.allocation_agent = InvestmentAllocationAgent(llm_model)
        self.research_agent = ResearchAgent(llm_model)
        self.macro_agent = MacroAgent(llm_model)
        self.fundamental_agent = FundamentalAgent(llm_model)
        self.technical_agent = TechnicalAgent(llm_model)
        self.volatility_agent = VolatilityLiquidityAgent(llm_model)
        self.behavioral_agent = BehavioralPsychologyAgent()
        self.risk_management_agent = RiskManagementAgent()

    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(InvestmentWorkflowState)
        
        workflow.add_node("allocation", self.run_allocation_agent)
        workflow.add_node("research", self.run_research_agent)
        workflow.add_node("parallel_analysis", self.run_parallel_analysis)
        workflow.add_node("final_decision", self.make_final_decision)
        workflow.add_node("risk_management", self.run_risk_management)
        
        workflow.set_entry_point("allocation")        
        workflow.add_edge("allocation", "research")        
        workflow.add_edge("research", "parallel_analysis")        
        workflow.add_edge("parallel_analysis", "risk_management")        
        workflow.add_edge("final_decision", "risk_management")        
        workflow.add_edge("risk_management", END)
        return workflow.compile()

    async def run_allocation_agent(self, state: InvestmentWorkflowState) -> InvestmentWorkflowState:
        logger.info("Running Allocation Agent")
        state["current_step"] = "allocation"
        
        try:
            allocation_input = {
                "investment_amount": state["investment_amount"],
                "user_preferences": state.get("user_preferences", {}),
                "messages": []
            }
            
            workflow = self.allocation_agent.create_workflow()
            result = await workflow.ainvoke(allocation_input)            
            
            state["allocation_strategy"] = result["allocation_strategy"]
            market_cond = result.get("market_conditions", {}) or {}
            preferred = market_cond.get("recommended_sectors") \
            or result.get("preferred_sectors") \
            or result.get("sector_constraints") \
            or ["technology", "healthcare"]  
            state["preferred_sectors"] = preferred
            state["risk_tolerance"] = result.get("risk_tolerance", "moderate")
            
            state["messages"].append(AIMessage(
                content=f"Allocation completed: {state['allocation_strategy'].value}, "
                        f"Preferred sectors: {', '.join(state['preferred_sectors']) if state['preferred_sectors'] else 'Any'}"
            ))
            
        except Exception as e:
            logger.error(f"Allocation agent failed: {e}")
            state["errors"].append(f"Allocation error: {str(e)}")
            
        return state
        
    async def run_research_agent(self, state: InvestmentWorkflowState) -> InvestmentWorkflowState:
        logger.info("Running Research Agent")
        state["current_step"] = "research"
        
        try:
            research_input = {
                "allocation_strategy": state["allocation_strategy"],
                "investment_amount": state["investment_amount"],
                "preferred_sectors": state.get("preferred_sectors", []),
                "risk_tolerance": state.get("risk_tolerance", "moderate"),
                "user_preferences": state.get("user_preferences", {}),
                "messages": []
            }
            print(research_input)
            
            workflow = self.research_agent.create_workflow()
            result = await workflow.ainvoke(research_input)
            print(result)
            state["stock_recommendations"] = result.get("stock_recommendations", result.get("recommendations", []))
            
            recommendations_count = len(state["stock_recommendations"])
            total_allocation = sum(rec.allocation_percentage for rec in state["stock_recommendations"])
            
            state["messages"].append(AIMessage(
                content=f"Research completed: Found {recommendations_count} stock recommendations "
                    f"with {total_allocation:.1f}% total allocation"
            ))
            
        except Exception as e:
            logger.error(f"Research agent failed: {e}")
            state["errors"].append(f"Research error: {str(e)}")
            
        return state

    async def run_parallel_analysis(self, state: InvestmentWorkflowState) -> InvestmentWorkflowState:
        logger.info("Running Parallel Analysis Agents")
        state["current_step"] = "parallel_analysis"
        
        if not state.get("stock_recommendations"):
            state["errors"].append("No stock recommendations available for analysis")
            return state
        
        tasks = []
        
        for recommendation in state["stock_recommendations"]:
            ticker = recommendation.ticker            
            tasks.append(("macro", self._run_macro_analysis(ticker)))            
            tasks.append(("fundamental", self._run_fundamental_analysis(ticker)))            
            tasks.append(("technical", self._run_technical_analysis(ticker)))            
            tasks.append(("volatility", self._run_volatility_analysis(ticker)))            
            tasks.append(("behavioral", self._run_behavioral_analysis(ticker)))
        
        try:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            analysis_results = {}
            for i, (analysis_type, result) in enumerate(zip([task[0] for task in tasks], results)):
                if isinstance(result, Exception):
                    logger.error(f"{analysis_type} analysis failed: {result}")
                    state["errors"].append(f"{analysis_type} analysis error: {str(result)}")
                else:
                    if analysis_type not in analysis_results:
                        analysis_results[analysis_type] = []
                    analysis_results[analysis_type].append(result)
            
            state["macro_analysis"] = analysis_results.get("macro", [None])[0]
            state["fundamental_analysis"] = analysis_results.get("fundamental", [None])[0]
            state["technical_analysis"] = analysis_results.get("technical", [None])[0]
            state["volatility_analysis"] = analysis_results.get("volatility", [None])[0]
            state["behavioral_analysis"] = analysis_results.get("behavioral", [None])[0]
            
            state["analysis_complete"] = True
            state["messages"].append(AIMessage(
                content="Parallel analysis completed: Macro, Fundamental, Technical, Volatility, and Behavioral analysis done"
            ))
            
        except Exception as e:
            logger.error(f"Parallel analysis failed: {e}")
            state["errors"].append(f"Parallel analysis error: {str(e)}")
            
        return state

    async def make_final_decision(self, state: InvestmentWorkflowState) -> InvestmentWorkflowState:
        logger.info("Making Final Decision")
        state["current_step"] = "final_decision"
        
        try:
            recommendations = state.get("stock_recommendations", [])
            if not recommendations:
                state["errors"].append("No recommendations available for final decision")
                return state
            
            scored_recommendations = []
            for rec in recommendations:
                total_score = self._calculate_total_score(
                    state.get("macro_analysis"),
                    state.get("fundamental_analysis"),
                    state.get("technical_analysis"),
                    state.get("volatility_analysis"),
                    state.get("behavioral_analysis")
                )
                
                scored_recommendations.append((rec, total_score))
            
            scored_recommendations.sort(key=lambda x: x[1], reverse=True)
            
            best_stocks = [rec for rec, score in scored_recommendations[:state["target_stocks"]]]
            avg_score = sum(score for _, score in scored_recommendations[:state["target_stocks"]]) / len(best_stocks) if best_stocks else 0
            
            final_rec = FinalRecommendation(
                selected_stocks=best_stocks,
                total_score=avg_score,
                confidence_level=min(avg_score / 100, 1.0),
                macro_analysis=state.get("macro_analysis"),
                fundamental_analysis=state.get("fundamental_analysis"),
                technical_analysis=state.get("technical_analysis"),
                volatility_analysis=state.get("volatility_analysis"),
                behavioral_analysis=state.get("behavioral_analysis"),
                risk_management=None,  # Will be filled by risk management agent
                reasoning=f"Selected {len(best_stocks)} stocks based on comprehensive analysis"
            )
            
            state["final_recommendation"] = final_rec
            state["messages"].append(AIMessage(
                content=f"Final decision made: Selected {len(best_stocks)} stocks with confidence {final_rec.confidence_level:.2%}"
            ))
            
        except Exception as e:
            logger.error(f"Final decision failed: {e}")
            state["errors"].append(f"Final decision error: {str(e)}")
            
        return state

    async def run_risk_management(self, state: InvestmentWorkflowState) -> InvestmentWorkflowState:
        logger.info("Running Risk Management Agent")
        state["current_step"] = "risk_management"
        
        try:
            recommendations = state.get("stock_recommendations", [])
            if not recommendations:
                state["errors"].append("No stock recommendations available for risk management")
                return state            
            def dataclass_to_dict(obj):
                if obj is None:
                    return {}
                if hasattr(obj, '__dataclass_fields__'):
                    return asdict(obj)
                return obj if isinstance(obj, dict) else {}
            
            selected_stocks = []
            for rec in recommendations:
                stock_dict = {
                    'symbol': rec.ticker,
                    'current_price': getattr(rec, 'current_price', 100.0),
                    'volatility': getattr(rec, 'volatility', 0.2),
                    'sector': getattr(rec, 'sector', 'Unknown'),
                    'weight': rec.allocation_percentage / 100.0,
                    'technical_levels': {
                        'support': getattr(rec, 'support_level', 0),
                        'resistance': getattr(rec, 'resistance_level', 0)
                    }
                }
                selected_stocks.append(stock_dict)            
            risk_input = {
                "investment_amount": state["investment_amount"],
                "selected_stocks": selected_stocks,
                "macro_analysis": dataclass_to_dict(state.get("macro_analysis")),
                "fundamental_data": dataclass_to_dict(state.get("fundamental_analysis")),
                "technical_analysis": dataclass_to_dict(state.get("technical_analysis")),
                "volatility_data": dataclass_to_dict(state.get("volatility_analysis")),
                "behavioral_biases": [
                    dataclass_to_dict(state.get("behavioral_analysis")) if state.get("behavioral_analysis") else {}
                ],
                "position_sizing": {},  
                "risk_metrics": {},
                "stop_loss_levels": {},
                "take_profit_levels": {},
                "position_risks": {},
                "portfolio_risk_limits": {},
                "risk_monitoring_rules": [],
                "emergency_exit_conditions": [],
                "risk_adjusted_positions": {}
            }            
            workflow = self.risk_management_agent.create_workflow()
            result = await workflow.ainvoke(risk_input)            
            risk_mgmt = RiskManagement(
                stop_loss_price=result.get("stop_loss_levels", {}).get("average_stop", 0),
                take_profit_price=result.get("take_profit_levels", {}).get("average_target", 0),
                position_size=result.get("risk_adjusted_positions", {}).get("total_size", 0),
                risk_reward_ratio=result.get("risk_metrics", {}).get("avg_risk_reward", 0),
                max_loss_amount=result.get("risk_metrics", {}).get("max_drawdown_limit", 0),
                reasoning=result.get("risk_management_summary", {}).get("primary_stop_method", "Risk management applied")
            )
            
            state["risk_management"] = risk_mgmt            
            final_rec = FinalRecommendation(
                selected_stocks=recommendations,
                total_score=self._calculate_portfolio_score(state),
                confidence_level=self._calculate_confidence_level(state),
                macro_analysis=state.get("macro_analysis"),
                fundamental_analysis=state.get("fundamental_analysis"),
                technical_analysis=state.get("technical_analysis"),
                volatility_analysis=state.get("volatility_analysis"),
                behavioral_analysis=state.get("behavioral_analysis"),
                risk_management=risk_mgmt,
                reasoning=f"Portfolio recommendation with {len(recommendations)} stocks, "
                        f"total investment: ₹{sum(rec.allocation_amount for rec in recommendations):,.0f}"
            )
            
            state["final_recommendation"] = final_rec            
            total_amount = sum(rec.allocation_amount for rec in recommendations)
            stock_list = ", ".join([f"{rec.ticker}({rec.allocation_percentage:.1f}%)" for rec in recommendations])
            
            state["messages"].append(AIMessage(
                content=f"Risk management completed for portfolio: {stock_list}. "
                    f"Total allocated: ₹{total_amount:,.0f}"
            ))
            
        except Exception as e:
            logger.error(f"Risk management failed: {e}")
            state["errors"].append(f"Risk management error: {str(e)}")
            
        return state

    def _calculate_portfolio_score(self, state: InvestmentWorkflowState) -> float:
        scores = []
        weights = {"macro": 0.2, "fundamental": 0.3, "technical": 0.2, "volatility": 0.15, "behavioral": 0.15}
        
        if state.get("macro_analysis"):
            scores.append(state["macro_analysis"].economic_score * weights["macro"])
        if state.get("fundamental_analysis"):
            scores.append(state["fundamental_analysis"].financial_score * weights["fundamental"])
        if state.get("technical_analysis"):
            scores.append(state["technical_analysis"].trend_score * weights["technical"])
        if state.get("volatility_analysis"):
            scores.append(state["volatility_analysis"].volatility_score * weights["volatility"])
        if state.get("behavioral_analysis"):
            scores.append(state["behavioral_analysis"].sentiment_score * weights["behavioral"])
        
        return sum(scores) if scores else 50.0

    def _calculate_confidence_level(self, state: InvestmentWorkflowState) -> float:
        """Calculate confidence level based on analysis completeness and scores"""
        completed_analyses = 0
        total_analyses = 5
        
        if state.get("macro_analysis"):
            completed_analyses += 1
        if state.get("fundamental_analysis"):
            completed_analyses += 1
        if state.get("technical_analysis"):
            completed_analyses += 1
        if state.get("volatility_analysis"):
            completed_analyses += 1
        if state.get("behavioral_analysis"):
            completed_analyses += 1
        
        completeness_score = completed_analyses / total_analyses
        portfolio_score = self._calculate_portfolio_score(state)
        
        # Confidence is based on both completeness and quality of analysis
        confidence = (completeness_score * 0.4) + ((portfolio_score / 100) * 0.6)
        return min(max(confidence, 0.0), 1.0)
        
    async def _run_macro_analysis(self, ticker: str) -> MacroAnalysis:
        return MacroAnalysis(
            economic_score=75.0,
            interest_rate_impact=0.1,
            global_events_score=80.0,
            reasoning=f"Macro analysis for {ticker}"
        )

    async def _run_fundamental_analysis(self, ticker: str) -> FundamentalAnalysis:
        """Run fundamental analysis for a ticker"""
        return FundamentalAnalysis(
            financial_score=80.0,
            sector_strength=75.0,
            pe_ratio=15.5,
            debt_ratio=0.3,
            reasoning=f"Fundamental analysis for {ticker}"
        )

    async def _run_technical_analysis(self, ticker: str) -> TechnicalAnalysis:
        """Run technical analysis for a ticker"""
        return TechnicalAnalysis(
            trend_score=70.0,
            momentum_indicators={"RSI": 55.0, "MACD": 0.5},
            support_resistance={"support": 100.0, "resistance": 120.0},
            pattern_detected="bullish_flag",
            reasoning=f"Technical analysis for {ticker}"
        )

    async def _run_volatility_analysis(self, ticker: str) -> VolatilityAnalysis:
        """Run volatility analysis for a ticker"""
        return VolatilityAnalysis(
            volatility_score=65.0,
            liquidity_score=85.0,
            risk_rating="moderate",
            reasoning=f"Volatility analysis for {ticker}"
        )

    async def _run_behavioral_analysis(self, ticker: str) -> BehavioralAnalysis:
        return BehavioralAnalysis(
            sentiment_score=70.0,
            market_psychology="cautiously_optimistic",
            risk_factors=["market_volatility", "sector_rotation"],
            reasoning=f"Behavioral analysis for {ticker}"
        )

    def _calculate_total_score(self, macro, fundamental, technical, volatility, behavioral) -> float:
        """Calculate total score - kept for backward compatibility but not used"""
        scores = []
        weights = {"macro": 0.2, "fundamental": 0.3, "technical": 0.2, "volatility": 0.15, "behavioral": 0.15}
        
        if macro:
            scores.append(macro.economic_score * weights["macro"])
        if fundamental:
            scores.append(fundamental.financial_score * weights["fundamental"])
        if technical:
            scores.append(technical.trend_score * weights["technical"])
        if volatility:
            scores.append(volatility.volatility_score * weights["volatility"])
        if behavioral:
            scores.append(behavioral.sentiment_score * weights["behavioral"])
        
        return sum(scores) if scores else 0.0
    
    async def run_complete_workflow(self, investment_amount: float, user_preferences: Dict[str, Any] = None) -> InvestmentWorkflowState:
        logger.info(f"Starting complete investment workflow for amount: ₹{investment_amount}")
        
        initial_state = InvestmentWorkflowState(
            investment_amount=investment_amount,
            user_preferences=user_preferences or {},
            allocation_strategy=None,
            preferred_sectors=[],
            risk_tolerance="moderate",
            stock_recommendations=[],
            macro_analysis=None,
            fundamental_analysis=None,
            technical_analysis=None,
            volatility_analysis=None,
            behavioral_analysis=None,
            final_recommendation=None,
            risk_management=None,
            messages=[HumanMessage(content=f"Investment request for ₹{investment_amount}")],
            current_step="initialization",
            analysis_complete=False,
            errors=[]
        )
        
        try:
            workflow = self.create_workflow()
            final_state = await workflow.ainvoke(initial_state)
            
            logger.info("Complete workflow finished successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state["errors"].append(f"Workflow error: {str(e)}")
            return initial_state


async def main():
    api_keys = {
        "openai": "your-openai-key",
        "twelve_data": "your-twelve-data-key",
        "economic_api": "your-economic-api-key",
        "financial_api": "your-financial-api-key",
        "technical_api": "your-technical-api-key",
        "volatility_api": "your-volatility-api-key",
        "sentiment_api": "your-sentiment-api-key"
    }
    
    orchestrator = InvestmentOrchestrator(api_keys)
    
    user_preferences = {
        "risk_level": "moderate",
        "investment_horizon": "long_term",
        "sectors_to_avoid": ["tobacco", "weapons"]
    }
    
    result = await orchestrator.run_complete_workflow(
        investment_amount=1000.0,
        user_preferences=user_preferences
    )
    
    print(f"Workflow completed with {len(result['errors'])} errors")
    print(f"Final recommendation: {result['final_recommendation']}")
    print(f"Risk management: {result['risk_management']}")


if __name__ == "__main__":
    asyncio.run(main())