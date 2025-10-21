import logging
from ..models.types import ResearchAgentState

logger = logging.getLogger(__name__)

class BaseResearchAgent:    
    def __init__(self, llm=None):
        self.llm = llm
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")
    
    async def execute(self, state: ResearchAgentState) -> ResearchAgentState:
        raise NotImplementedError(f"{self.name} must implement execute method")
    
    def log_message(self, state: ResearchAgentState, message: str):
        from langchain_core.messages import AIMessage
        logger.info(f"[{self.name}] {message}")
        state["messages"].append(AIMessage(content=f"[{self.name}] {message}"))
