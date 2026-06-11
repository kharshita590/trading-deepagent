import logging
from langgraph.graph import StateGraph, END
from ..models.types import PortfolioState
from .workflow_nodes import WorkflowNodes

logger = logging.getLogger(__name__)


class WorkflowBuilder:    
    def __init__(self, nodes: WorkflowNodes):
        self.nodes = nodes
    
    def build(self) -> StateGraph:
        workflow = StateGraph(PortfolioState)        
        self._add_nodes(workflow)        
        self._configure_edges(workflow)
        
        return workflow.compile()
    
    def _add_nodes(self, workflow: StateGraph):
        workflow.add_node("run_investment_allocation", self.nodes.run_investment_allocation)
        workflow.add_node("run_research", self.nodes.run_research)
        workflow.add_node("run_fundamental_analysis", self.nodes.run_fundamental_analysis)
        workflow.add_node("run_macro_analysis", self.nodes.run_macro_analysis)
        workflow.add_node("run_technical_analysis", self.nodes.run_technical_analysis)
        workflow.add_node("run_volatility_liquidity", self.nodes.run_volatility_liquidity)
        workflow.add_node("run_behavioral_psychology", self.nodes.run_behavioral_psychology)
        workflow.add_node("parallel_join", self.nodes.join_parallel_analyses)
        workflow.add_node("run_risk_management", self.nodes.run_risk_management)
    
    def _configure_edges(self, workflow: StateGraph):
        workflow.set_entry_point("run_investment_allocation")
        
        workflow.add_edge("run_investment_allocation", "run_research")
        
        workflow.add_edge("run_research", "run_fundamental_analysis")
        workflow.add_edge("run_research", "run_macro_analysis")
        workflow.add_edge("run_research", "run_technical_analysis")
        workflow.add_edge("run_research", "run_volatility_liquidity")
        workflow.add_edge("run_research", "run_behavioral_psychology")
        
        workflow.add_edge("run_fundamental_analysis", "parallel_join")
        workflow.add_edge("run_macro_analysis", "parallel_join")
        workflow.add_edge("run_technical_analysis", "parallel_join")
        workflow.add_edge("run_volatility_liquidity", "parallel_join")
        workflow.add_edge("run_behavioral_psychology", "parallel_join")
        
        workflow.add_edge("parallel_join", "run_risk_management")
        workflow.add_edge("run_risk_management", END)
