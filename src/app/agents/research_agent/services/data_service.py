import yfinance as yf
import asyncio
import logging
import pandas as pd
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class StockDataService:    
    @staticmethod
    async def fetch_prices_batch(tickers: List[str], batch_size: int = 50) -> Dict[str, float]:
        """Fetch stock prices in batches using yfinance"""
        all_prices = {}
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            nse_tickers = [f"{ticker}.NS" for ticker in batch_tickers]
            
            try:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    data = await loop.run_in_executor(
                        executor,
                        lambda: yf.download(
                            nse_tickers, 
                            period="1d", 
                            interval="1d", 
                            progress=False, 
                            threads=True
                        )
                    )
                    
                    if not data.empty:
                        if len(batch_tickers) == 1:
                            ticker = batch_tickers[0]
                            if 'Close' in data.columns and not data['Close'].empty:
                                all_prices[ticker] = float(data['Close'].iloc[-1])
                            else:
                                all_prices[ticker] = 0.0
                        else:
                            # Multiple tickers
                            for ticker in batch_tickers:
                                nse_ticker = f"{ticker}.NS"
                                try:
                                    if ('Close', nse_ticker) in data.columns:
                                        close_series = data[('Close', nse_ticker)]
                                        if not close_series.empty and not pd.isna(close_series.iloc[-1]):
                                            all_prices[ticker] = float(close_series.iloc[-1])
                                        else:
                                            all_prices[ticker] = 0.0
                                    else:
                                        all_prices[ticker] = 0.0
                                except Exception as e:
                                    logger.warning(f"Error processing {ticker}: {e}")
                                    all_prices[ticker] = 0.0
                    else:
                        for ticker in batch_tickers:
                            all_prices[ticker] = 0.0
                            
            except Exception as e:
                logger.error(f"Batch fetch failed for batch starting at {i}: {e}")
                for ticker in batch_tickers:
                    all_prices[ticker] = 0.0
            
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.1)
        
        logger.info(f"Fetched prices for {len(all_prices)} stocks in batches")
        return all_prices
    
    @staticmethod
    def parse_market_cap(mc_str: str) -> float:
        """Parse market cap string to numeric value"""
        if not mc_str or mc_str == "":
            return 0
        
        mc_str = str(mc_str).upper().replace(",", "")
        
        try:
            if "B" in mc_str:
                return float(mc_str.replace("B", "")) * 1_000_000_000
            elif "M" in mc_str:
                return float(mc_str.replace("M", "")) * 1_000_000
            elif "K" in mc_str:
                return float(mc_str.replace("K", "")) * 1_000
            else:
                return float(mc_str) if mc_str.isdigit() else 0
        except:
            return 0
