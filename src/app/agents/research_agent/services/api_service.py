
import aiohttp
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TwelveDataService:    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
    
    async def fetch_stocks_by_sector(self, sector: str) -> List[Dict]:
        """Fetch stocks for a given sector"""
        url = f"{self.base_url}/stocks"
        params = {
            "sector": sector,
            "country": "India",
            "apikey": self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        stocks = data.get("data", [])
                        logger.info(f"Fetched {len(stocks)} stocks for sector {sector}")
                        return stocks
                    else:
                        logger.error(f"Failed to fetch stocks for sector {sector}: {resp.status}")
                        return []
        except Exception as e:
            logger.exception(f"Error fetching stocks for sector {sector}: {e}")
            return []
