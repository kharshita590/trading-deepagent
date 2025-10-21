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
        workflow.add_node("investment_allocation", self.nodes.run_investment_allocation)
        workflow.add_node("research", self.nodes.run_research)
        workflow.add_node("fundamental_analysis", self.nodes.run_fundamental_analysis)
        workflow.add_node("macro_analysis", self.nodes.run_macro_analysis)
        workflow.add_node("technical_analysis", self.nodes.run_technical_analysis)
        workflow.add_node("volatility_liquidity_analysis", self.nodes.run_volatility_liquidity)
        workflow.add_node("behavioral_psychology_analysis", self.nodes.run_behavioral_psychology)
        workflow.add_node("parallel_join", self.nodes.join_parallel_analyses)
        workflow.add_node("risk_management", self.nodes.run_risk_management)
    
    def _configure_edges(self, workflow: StateGraph):
        workflow.set_entry_point("investment_allocation")
        
        workflow.add_edge("investment_allocation", "research")
        
        workflow.add_edge("research", "fundamental_analysis")
        workflow.add_edge("research", "macro_analysis")
        workflow.add_edge("research", "technical_analysis")
        workflow.add_edge("research", "volatility_liquidity_analysis")
        workflow.add_edge("research", "behavioral_psychology_analysis")
        
        workflow.add_edge("fundamental_analysis", "parallel_join")
        workflow.add_edge("macro_analysis", "parallel_join")
        workflow.add_edge("technical_analysis", "parallel_join")
        workflow.add_edge("volatility_liquidity_analysis", "parallel_join")
        workflow.add_edge("behavioral_psychology_analysis", "parallel_join")
        
        workflow.add_edge("parallel_join", "risk_management")
        workflow.add_edge("risk_management", END)
