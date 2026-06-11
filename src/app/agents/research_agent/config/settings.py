from app.config.env import getenv

TWELVEDATA_API_KEY = getenv("TWELVEDATA_API_KEY", "") or ""

LLM_CONFIG = {
    "model": "gemini-2.5-flash",
    "temperature": 0,
    "api_key": getenv("GOOGLE_API_KEY", "") or ""
}

STOCK_FILTERING = {
    "max_stocks_to_process": 200,
    "batch_size": 50,
    "batch_delay": 0.1  
}

ALLOCATION_RULES = {
    "single_stock": {
        "stocks_per_sector": 1,
        "selection_criteria": "highest_price"
    },
    "multi_stock": {
        "stocks_per_sector": 3,
        "selection_criteria": "top_by_price"
    },
    "hybrid": {
        "stocks_per_sector": 2,
        "selection_criteria": "balanced"
    }
}
