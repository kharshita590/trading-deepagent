import asyncio
from langgraph.graph import StateGraph, END
from ..models.types import FundamentalAgentState
from ..agents.company_financial import CompanyFinancialsAgent
from ..agents.compilation import FundamentalCompilationAgent
from ..agents.financial_data import FinancialDataFetchAgent
from ..agents.investement_thesis import InvestmentThesisAgent
from ..agents.sector_strength import SectorStrengthAgent
from ..models.types import FundamentalAnalysis,FundamentalAgentState
import logging 
logger = logging.getLogger(__name__)

class FundamentalOrchestrator:
    def __init__(self):
        self.data_fetch_agent = FinancialDataFetchAgent()
        self.company_financials_agent = CompanyFinancialsAgent()
        self.sector_strength_agent = SectorStrengthAgent()
        self.investment_thesis_agent = InvestmentThesisAgent()
        self.compilation_agent = FundamentalCompilationAgent()
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(FundamentalAgentState)
        
        workflow.add_node("fetch_financial_data", self.data_fetch_agent.execute)
        workflow.add_node("analyze_company_financials", self.company_financials_agent.execute)
        workflow.add_node("analyze_sector_strength", self.sector_strength_agent.execute)
        workflow.add_node("generate_investment_thesis", self.investment_thesis_agent.execute)
        workflow.add_node("compile_fundamental_analysis", self.compilation_agent.execute)
        
        workflow.add_edge("fetch_financial_data", "analyze_company_financials")
        workflow.add_edge("analyze_company_financials", "analyze_sector_strength")
        workflow.add_edge("analyze_sector_strength", "generate_investment_thesis")
        workflow.add_edge("generate_investment_thesis", "compile_fundamental_analysis")
        workflow.add_edge("compile_fundamental_analysis", END)
        
        workflow.set_entry_point("fetch_financial_data")
        
        return workflow
    
    async def run(self, recommendations):
        logger.info("Starting fundamental analysis workflow")
        
        initial_state = {
            "recommendations": recommendations,
            "fundamental_analysis": None,
            "messages": []
        }
        
        workflow = self.create_workflow()
        app = workflow.compile()
        
        final_state = await app.ainvoke(initial_state)
        
        return final_state

# async def main():
#     recommendations = [
#         {
#             "ticker": "ABBOTINDIA.NS",
#             "company_name": "Abbott India Ltd.",
#             "sector": "Healthcare",
#             "price": 29805.00,
#             "allocation_percentage": 33.3,
#             "allocation_amount": 16666.67,
#             "reasoning": "Pick from Healthcare sector"
#         },
#         {
#             "ticker": "3MINDIA.NS",
#             "company_name": "3M India Ltd.",
#             "sector": "Healthcare",
#             "price": 29565.00,
#             "allocation_percentage": 33.3,
#             "allocation_amount": 16666.67,
#             "reasoning": "Pick from Healthcare sector"
#         },
#         {
#             "ticker": "ABB.NS",
#             "company_name": "ABB India Ltd.",
#             "sector": "Healthcare",
#             "price": 5202.00,
#             "allocation_percentage": 33.3,
#             "allocation_amount": 16666.67,
#             "reasoning": "Pick from Healthcare sector"
#         }
#     ]
    
#     orchestrator = FundamentalOrchestrator()
#     result = await orchestrator.run(recommendations)
    
#     print("\n" + "="*80)
#     print("FUNDAMENTAL ANALYSIS RESULTS")
#     print("="*80)
    
#     analysis = result.get("fundamental_analysis")
#     if analysis:
#         print("\n--- COMPANY FINANCIALS SUMMARY ---")
#         print(analysis.company_financials_summary)
        
#         print("\n--- SECTOR STRENGTH SUMMARY ---")
#         print(analysis.sector_strength_summary)
        
#         print("\n--- FUNDAMENTAL INVESTMENT THESIS ---")
#         print(analysis.fundamental_investment_thesis)
    
#     print("\n" + "="*80)
    
#     return result

# if __name__ == "__main__":
#     asyncio.run(main())