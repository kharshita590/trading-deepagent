from __future__ import annotations

import pandas as pd

from app.agents.technical_agent.services.technical_data import TechnicalDataProvider


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100, 102, 101, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            "High": [101, 103, 102, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            "Low": [99, 101, 100, 102, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            "Close": [100, 102, 101, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            "Volume": [1000, 1100, 1200, 900, 1000, 1300, 1400, 1500, 1200, 1300, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500],
        }
    )


def test_indicator_and_signal_calculation():
    provider = TechnicalDataProvider()
    df = _sample_df()
    indicators = provider.calculate_technical_indicators(df)
    patterns = provider.detect_chart_patterns(df)
    signals = provider.generate_trading_signals(indicators, patterns, float(df["Close"].iloc[-1]))

    assert "rsi" in indicators
    assert "key_levels" in signals
    assert signals["overall_signal"] in {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}
