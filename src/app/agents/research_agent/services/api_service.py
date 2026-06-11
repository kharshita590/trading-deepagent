from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class NSEStockRecord:
    symbol: str
    company_name: str
    sector: str


class NSEStockUniverseService:
    def __init__(
        self,
        universe_csv: Optional[str] = None,
        twelve_data_api_key: str = "",
        use_twelvedata_fallback: Optional[bool] = None,
    ):
        self.universe_csv = Path(universe_csv or Path(__file__).resolve().parents[3] / "data" / "nse_universe.csv")
        self.twelve_data_api_key = twelve_data_api_key
        self.use_twelvedata_fallback = use_twelvedata_fallback if use_twelvedata_fallback is not None else os.getenv("USE_TWELVEDATA_FALLBACK", "false").lower() == "true"
        self.base_url = "https://api.twelvedata.com"

    def _load_csv(self) -> List[NSEStockRecord]:
        if not self.universe_csv.exists():
            logger.warning("NSE universe CSV not found at %s", self.universe_csv)
            return []

        records: List[NSEStockRecord] = []
        with self.universe_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = (row.get("symbol") or "").strip().upper()
                sector = (row.get("sector") or "").strip()
                company_name = (row.get("company_name") or "").strip()
                if symbol:
                    records.append(NSEStockRecord(symbol=symbol, company_name=company_name or symbol, sector=sector or "Unknown"))
        return records

    async def _fallback_twelvedata(self, sector: str) -> List[Dict]:
        if not self.use_twelvedata_fallback or not self.twelve_data_api_key:
            return []

        url = f"{self.base_url}/stocks"
        params = {
            "sector": sector,
            "country": "India",
            "apikey": self.twelve_data_api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("data", [])
        except Exception as exc:
            logger.exception("TwelveData fallback failed for sector %s: %s", sector, exc)
        return []

    async def fetch_stocks_by_sector(self, sector: str) -> List[Dict]:
        sector_normalized = sector.strip().lower()
        records = self._load_csv()
        matches = [
            {
                "symbol": record.symbol,
                "name": record.company_name,
                "sector": record.sector,
                "exchange": "NSE",
            }
            for record in records
            if record.sector.strip().lower() == sector_normalized or sector_normalized in record.sector.strip().lower()
        ]

        if matches:
            logger.info("Loaded %s NSE stocks for sector %s from local universe", len(matches), sector)
            return matches

        fallback = await self._fallback_twelvedata(sector)
        if fallback:
            logger.info("Loaded %s fallback stocks for sector %s from TwelveData", len(fallback), sector)
        return fallback


TwelveDataService = NSEStockUniverseService
