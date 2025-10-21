import asyncio
from langgraph.graph import StateGraph, END
from ..models.types import RiskManagementState, RiskLevel
from ..agents.risk_management import RiskManagementAgent

class RiskManagementOrchestrator:
    def __init__(self, base_risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_agent = RiskManagementAgent(base_risk_level)
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(RiskManagementState)
        
        workflow.add_node("risk_management", self.risk_agent)
        
        workflow.add_edge("risk_management", END)
        
        workflow.set_entry_point("risk_management")
        
        return workflow
    
    async def run(self, state: RiskManagementState) -> RiskManagementState:
        if 'messages' not in state:
            state['messages'] = []
        
        workflow = self.create_workflow()
        app = workflow.compile()
        
        final_state = await app.ainvoke(state)
        return final_state
