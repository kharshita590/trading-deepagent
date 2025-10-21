
from .base_agent import BaseResearchAgent
from ..models.types import ResearchAgentState, StockData
from ..services.data_service import StockDataService
from ..config.settings import STOCK_FILTERING

class StockFilterAgent(BaseResearchAgent):
    async def execute(self, state: ResearchAgentState) -> ResearchAgentState:
        self.log_message(state, "Filtering and sorting stocks")
        
        stock_lists = state.get("stock_lists", {})
        max_stocks = STOCK_FILTERING["max_stocks_to_process"]
        filtered_stocks = {}
        
        for sector, stocks in stock_lists.items():
            valid_stocks = []
            for stock in stocks:
                if stock.get("symbol") and stock.get("name"):
                    valid_stocks.append({
                        "symbol": stock.get("symbol"),
                        "name": stock.get("name"),
                        "market_cap": stock.get("market_cap", ""),
                        "sector": sector
                    })
            
            if valid_stocks:
                try:
                    valid_stocks.sort(
                        key=lambda x: StockDataService.parse_market_cap(x["market_cap"]),
                        reverse=True
                    )
                    self.log_message(state, f"Sorted {len(valid_stocks)} stocks in {sector} by market cap")
                except Exception as e:
                    self.log_message(state, f"Could not sort {sector}, using original order")
                
                selected = valid_stocks[:max_stocks]
                filtered_stocks[sector] = selected
                self.log_message(state, f"Selected top {len(selected)} stocks from {sector}")
        
        state["filtered_stocks"] = filtered_stocks
        return state

