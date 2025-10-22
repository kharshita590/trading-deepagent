ALLOCATION_FACTORS = {
    "diversification_benefits": {
        "single_stock_risk": 0.8, 
        "multi_stock_protection": 0.6,  
        "optimal_diversification": 0.3   
    },
    "amount_thresholds": {
        "small_investment": 10000,  
        "medium_investment": 50000, 
        "large_investment": 100000 
    },
    "volatility_considerations": {
        "high_vol_sectors": ["tech", "crypto", "small_cap"],
        "stable_sectors": ["utilities", "consumer_staples", "healthcare"],
        "defensive_sectors": ["government_bonds", "dividend_stocks"]
    }
}

LLM_CONFIG = {
    "model": "gemini-2.5-flash",
    "temperature": 0,
    "api_key": "AIzaSyDl0-DuUoAmjs4hjM8E7TnRL7qazQ2Bq8w"
}
