import asyncio
from langgraph.graph import StateGraph, END
from ..models.types import BehavioralPsychologyState
from ..agents.bias_analysis import BiasAnalysisAgent
from ..agents.psychological_profile import PsychologicalProfileAgent
from ..agents.risk_calculator import RiskCalculator,StopLossCalculationAgent,PositionSizingAgent
from ..agents.risk_management import RiskManagementRulesAgent
from ..agents.trade_plan import TradePlanGenerationAgent
from ..config.settings import logger

class BehavioralPsychologyOrchestrator:
    def __init__(self):
        self.bias_analysis_agent = BiasAnalysisAgent()
        self.stop_loss_agent = StopLossCalculationAgent()
        self.position_sizing_agent = PositionSizingAgent()
        self.risk_rules_agent = RiskManagementRulesAgent()
        self.trade_plan_agent = TradePlanGenerationAgent()
        self.profile_agent = PsychologicalProfileAgent()
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(BehavioralPsychologyState)
        
        workflow.add_node("analyze_biases", self.bias_analysis_agent.execute)
        workflow.add_node("calculate_stop_loss", self.stop_loss_agent.execute)
        workflow.add_node("calculate_position_sizing", self.position_sizing_agent.execute)
        workflow.add_node("generate_risk_rules", self.risk_rules_agent.execute)
        workflow.add_node("generate_trade_plan", self.trade_plan_agent.execute)
        workflow.add_node("compile_psychological_profile", self.profile_agent.execute)
        
        workflow.add_edge("analyze_biases", "calculate_stop_loss")
        workflow.add_edge("calculate_stop_loss", "calculate_position_sizing")
        workflow.add_edge("calculate_position_sizing", "generate_risk_rules")
        workflow.add_edge("generate_risk_rules", "generate_trade_plan")
        workflow.add_edge("generate_trade_plan", "compile_psychological_profile")
        workflow.add_edge("compile_psychological_profile", END)
        
        workflow.set_entry_point("analyze_biases")
        
        return workflow
    
    async def run(self, state: BehavioralPsychologyState):
        logger.info("Starting behavioral psychology analysis workflow")
        
        if 'messages' not in state:
            state['messages'] = []
        
        workflow = self.create_workflow()
        app = workflow.compile()
        
        final_state = await app.ainvoke(state)
        
        return final_state

# async def main():
#     initial_state = {
#         "investment_amount": 50000.0,
#         "selected_stocks": [
#             {
#                 "symbol": "ABBOTINDIA.NS",
#                 "current_price": 29805.0,
#                 "volatility": 0.25,
#                 "historical_win_rate": 0.60,
#                 "avg_win": 0.10,
#                 "avg_loss": 0.05
#             },
#             {
#                 "symbol": "3MINDIA.NS",
#                 "current_price": 29565.0,
#                 "volatility": 0.22,
#                 "historical_win_rate": 0.58,
#                 "avg_win": 0.09,
#                 "avg_loss": 0.04
#             },
#             {
#                 "symbol": "ABB.NS",
#                 "current_price": 5202.0,
#                 "volatility": 0.30,
#                 "historical_win_rate": 0.55,
#                 "avg_win": 0.08,
#                 "avg_loss": 0.06
#             }
#         ],
#         "technical_analysis": {
#             "confidence_score": 0.85,
#             "momentum_score": 0.78
#         },
#         "volatility_data": {},
#         "fundamental_data": {}
#     }
    
#     orchestrator = BehavioralPsychologyOrchestrator()
#     result = await orchestrator.run(initial_state)
    
#     print("\n" + "="*80)
#     print("BEHAVIORAL PSYCHOLOGY ANALYSIS RESULTS")
#     print("="*80)
    
#     print("\n--- BEHAVIORAL BIASES IDENTIFIED ---")
#     for bias in result.get('behavioral_biases', []):
#         print(f"- {bias['type'].value.upper()}: {bias['description']}")
#         print(f"  Severity: {bias['severity']:.2f}")
#         print(f"  Mitigation: {bias['mitigation_strategy']}\n")
    
#     print("\n--- STOP-LOSS LEVELS ---")
#     for symbol, levels in result.get('stop_loss_levels', {}).items():
#         print(f"{symbol}:")
#         print(f"  Stop Price: ₹{levels['stop_price']:.2f}")
#         print(f"  Stop %: {levels['stop_percent']*100:.2f}%")
#         print(f"  Psychological Factor: {levels['psychological_factor']:.2f}\n")
    
#     print("\n--- POSITION SIZING ---")
#     for symbol, sizing in result.get('position_sizing', {}).items():
#         print(f"{symbol}:")
#         print(f"  Amount: ₹{sizing['amount']:.2f}")
#         print(f"  % of Total: {sizing['percent_of_total']*100:.2f}%")
#         print(f"  Kelly Optimal: ₹{sizing['kelly_optimal']:.2f}")
#         print(f"  Psych Adjusted: ₹{sizing['psychological_adjusted']:.2f}\n")
    
#     print("\n--- RISK MANAGEMENT RULES ---")
#     for rule in result.get('risk_management_rules', []):
#         print(f"- [{rule['rule_id']}] {rule['description']}")
    
#     print("\n--- TRADE PLAN ---")
#     trade_plan = result.get('trade_plan', {})
#     print(f"Strategy: {trade_plan.get('strategy')}")
#     print(f"Total Positions: {trade_plan.get('total_positions')}")
#     print(f"Max Risk Per Trade: {trade_plan.get('max_risk_per_trade')*100}%")
#     print(f"Overall Portfolio Risk: {trade_plan.get('overall_portfolio_risk')*100}%")
#     print(f"Rebalancing: {trade_plan.get('rebalancing_frequency')}")
    
#     print("\n--- PSYCHOLOGICAL PROFILE ---")
#     profile = result.get('psychological_profile', {})
#     print(f"Bias Count: {profile.get('bias_count')}")
#     print(f"Primary Concerns: {', '.join(profile.get('primary_concerns', []))}")
#     print(f"Risk Adjustment Applied: {profile.get('risk_adjustment_applied')}")
    
#     print("\n" + "="*80)
    
#     return result

# if __name__ == "__main__":
#     asyncio.run(main())