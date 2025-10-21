
from .base_agent import BaseResearchAgent
from ..models.types import ResearchAgentState, StockData
from ..services.data_service import StockDataService
from ..config.settings import STOCK_FILTERING

class PriceFetcherAgent(BaseResearchAgent):
    async def execute(self, state: ResearchAgentState) -> ResearchAgentState:
        self.log_message(state, "Fetching stock prices")
        
        filtered_stocks = state.get("filtered_stocks", {})
        investment_amount = state["investment_amount"]
        
        all_tickers = []
        ticker_to_sector = {}
        ticker_to_data = {}
        
        for sector, stocks in filtered_stocks.items():
            for stock in stocks:
                ticker = stock["symbol"]
                all_tickers.append(ticker)
                ticker_to_sector[ticker] = sector
                ticker_to_data[ticker] = stock
        
        self.log_message(state, f"Fetching prices for {len(all_tickers)} stocks")
        
        prices = await StockDataService.fetch_prices_batch(
            all_tickers, 
            batch_size=STOCK_FILTERING["batch_size"]
        )
        
        valid_stocks_by_sector = {}
        
        for ticker, price in prices.items():
            if price > 0 and price <= investment_amount:
                sector = ticker_to_sector[ticker]
                stock_info = ticker_to_data[ticker]
                
                stock_data = StockData(
                    ticker=ticker,
                    name=stock_info["name"],
                    sector=sector,
                    price=price,
                    market_cap=stock_info["market_cap"]
                )
                
                if sector not in valid_stocks_by_sector:
                    valid_stocks_by_sector[sector] = []
                valid_stocks_by_sector[sector].append(stock_data)
        
        state["stock_prices"] = prices
        state["filtered_stocks"] = valid_stocks_by_sector
        
        total_valid = sum(len(stocks) for stocks in valid_stocks_by_sector.values())
        self.log_message(state, f"Found {total_valid} valid stocks with prices")
        
        return state
