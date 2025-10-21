from .base_agent import BaseResearchAgent
from ..models.types import ResearchAgentState, StockRecommendation, AllocationStrategy
from ..config.settings import ALLOCATION_RULES

class RecommendationAgent(BaseResearchAgent):
    async def execute(self, state: ResearchAgentState) -> ResearchAgentState:
        self.log_message(state, "Generating stock recommendations")
        
        strategy = state["allocation_strategy"]
        amount = state["investment_amount"]
        filtered_stocks = state.get("filtered_stocks", {})
        
        recommendations = []
        
        if strategy == AllocationStrategy.SINGLE_STOCK:
            recommendations = self._generate_single_stock(filtered_stocks, amount)
        else:
            recommendations = self._generate_multi_stock(filtered_stocks, amount, strategy)
        
        state["recommendations"] = recommendations
        
        total_allocated = sum(rec.allocation_amount for rec in recommendations)
        summary = f"Generated {len(recommendations)} recommendations. Total: ₹{total_allocated:,.0f}"
        self.log_message(state, summary)
        
        return state
    
    def _generate_single_stock(self, filtered_stocks, amount):
        recommendations = []
        
        all_stocks = []
        for sector, stocks in filtered_stocks.items():
            all_stocks.extend(stocks)
        
        if all_stocks:
            best_stock = max(all_stocks, key=lambda x: x.price)
            recommendations.append(StockRecommendation(
                ticker=best_stock.ticker,
                company_name=best_stock.name,
                price=best_stock.price,
                allocation_percentage=100.0,
                allocation_amount=amount,
                sector=best_stock.sector,
                market_cap=best_stock.market_cap,
                reasoning=f"Single stock pick from {best_stock.sector}"
            ))
        
        return recommendations
    
    def _generate_multi_stock(self, filtered_stocks, amount, strategy):
        recommendations = []
        all_picks = []
        
        for sector, stocks in filtered_stocks.items():
            if not stocks:
                continue
            
            sorted_stocks = sorted(stocks, key=lambda x: x.price, reverse=True)
            num_picks = ALLOCATION_RULES.get(strategy.value, {}).get("stocks_per_sector", 3)
            picks = sorted_stocks[:num_picks]
            
            for stock in picks:
                all_picks.append(stock)
        
        if all_picks:
            equal_alloc = amount / len(all_picks)
            allocation_pct = 100.0 / len(all_picks)
            
            for stock in all_picks:
                recommendations.append(StockRecommendation(
                    ticker=stock.ticker,
                    company_name=stock.name,
                    price=stock.price,
                    allocation_percentage=allocation_pct,
                    allocation_amount=equal_alloc,
                    sector=stock.sector,
                    market_cap=stock.market_cap,
                    reasoning=f"Pick from {stock.sector} sector"
                ))
        
        return recommendations