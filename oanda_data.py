import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import ta
from oanda_config import OandaConfig

class OandaDataFetcher:
    """Fetches market data from Oanda API"""

    def __init__(self, config: OandaConfig):
        self.config = config

    def get_candles(
        self,
        instrument: str,
        granularity: str = "H1",
        count: int = 100,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLC candles from Oanda

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Time frame (M1, M5, M15, M30, H1, H4, D, W, M)
            count: Number of candles to fetch
            from_time: RFC3339 format timestamp
            to_time: RFC3339 format timestamp
        """
        endpoint = f"{self.config.base_url}/v3/instruments/{instrument}/candles"

        params = {
            "granularity": granularity,
            "count": count,
            "price": "MBA"  # Mid, Bid, Ask
        }

        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time

        response = requests.get(
            endpoint,
            headers=self.config.headers,
            params=params
        )

        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.text}")

        data = response.json()
        candles = data.get("candles", [])

        if not candles:
            return pd.DataFrame()

        df = self._process_candles(candles)
        return df

    def _process_candles(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw candle data into DataFrame"""
        processed = []

        for candle in candles:
            if candle["complete"]:
                mid = candle["mid"]
                processed.append({
                    "time": pd.to_datetime(candle["time"]),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(candle["volume"])
                })

        df = pd.DataFrame(processed)
        if not df.empty:
            df.set_index("time", inplace=True)
            df = df.sort_index()

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators matching the notebook implementation"""
        if df.empty:
            return df

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=df['close'],
            window=20,
            window_dev=2
        )
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()

        # Moving Average and Slope
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_20_slope'] = df['ma_20'].diff()

        # Fill NaNs
        df = df.bfill()
        df = df.ffill()

        return df

    def get_latest_price(self, instrument: str) -> Dict[str, float]:
        """Get latest bid/ask prices"""
        endpoint = f"{self.config.base_url}/v3/instruments/{instrument}/pricing"

        params = {
            "instruments": instrument
        }

        response = requests.get(
            endpoint,
            headers=self.config.headers,
            params=params
        )

        if response.status_code != 200:
            raise Exception(f"Failed to fetch price: {response.text}")

        data = response.json()
        prices = data.get("prices", [])

        if not prices:
            return {}

        price_data = prices[0]
        return {
            "bid": float(price_data["bids"][0]["price"]),
            "ask": float(price_data["asks"][0]["price"]),
            "spread": float(price_data["asks"][0]["price"]) - float(price_data["bids"][0]["price"])
        }

    def prepare_model_input(
        self,
        df: pd.DataFrame,
        seq_length: int = 30,
        feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """Prepare data for model input"""
        if feature_cols is None:
            feature_cols = [
                'open', 'high', 'low', 'close',
                'rsi', 'bb_high', 'bb_low', 'ma_20', 'ma_20_slope'
            ]

        # Ensure we have enough data
        if len(df) < seq_length:
            raise ValueError(f"Insufficient data: {len(df)} < {seq_length}")

        # Get the last seq_length rows
        recent_data = df[feature_cols].tail(seq_length).values

        return recent_data