from .base_agent import BaseResearchAgent
from ..models.types import ResearchAgentState
from ..services.api_service import TwelveDataService
from ..config.settings import TWELVEDATA_API_KEY

class StockFetcherAgent(BaseResearchAgent):    
    def __init__(self, llm=None):
        super().__init__(llm)
        self.api_service = TwelveDataService(TWELVEDATA_API_KEY)
    
    async def execute(self, state: ResearchAgentState) -> ResearchAgentState:
        preferred_sectors = state["preferred_sectors"]
        self.log_message(state, f"Fetching stocks for sectors: {preferred_sectors}")
        
        stock_lists = {}
        
        for sector in preferred_sectors:
            stocks = await self.api_service.fetch_stocks_by_sector(sector)
            if stocks:
                stock_lists[sector] = stocks
                self.log_message(state, f"Found {len(stocks)} stocks in {sector}")
            else:
                self.log_message(state, f"No stocks found for {sector}")
        
        state["stock_lists"] = stock_lists
        return state