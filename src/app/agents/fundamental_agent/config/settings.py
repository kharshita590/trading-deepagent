import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = ""
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financial Services": "XLF", 
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}