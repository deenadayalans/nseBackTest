"""
real_options.py — Fast lookup of real option OHLC from downloaded parquet files.

Usage:
    loader = RealOptionsLoader()
    price = loader.get_price("NIFTY50", timestamp, strike=24000)
    # Returns actual 5-min close price, or None if not available
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from functools import lru_cache

DATA_DIR = Path("./data/options")


class RealOptionsLoader:
    """
    Caches option 5-min data in memory per (symbol, expiry_date, strike).
    Falls back gracefully to None when data is missing.
    """

    def __init__(self):
        self._cache: dict[tuple, pd.DataFrame] = {}
        self._missing: set[tuple] = set()   # avoid repeated disk misses

    def _load(self, symbol: str, expiry_date: date, strike: int) -> pd.DataFrame | None:
        key = (symbol, expiry_date, strike)
        if key in self._missing:
            return None
        if key in self._cache:
            return self._cache[key]

        path = DATA_DIR / symbol / str(expiry_date) / f"{strike}_CE.parquet"
        if not path.exists():
            self._missing.add(key)
            return None

        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        self._cache[key] = df
        return df

    def get_price(self, symbol: str, ts: pd.Timestamp,
                  strike: int, expiry_date: date,
                  col: str = "close") -> float | None:
        """
        Return the 5-min bar close (or open/high/low) at timestamp ts
        for the given symbol/strike/expiry.

        Returns None if the data file doesn't exist or ts isn't in it.
        """
        df = self._load(symbol, expiry_date, strike)
        if df is None or df.empty:
            return None

        # Find the bar whose timestamp matches ts (floor to 5-min)
        ts_floor = ts.floor("5min")
        if ts_floor in df.index:
            val = df.at[ts_floor, col]
            return float(val) if not np.isnan(val) else None

        # Try exact match (sometimes timestamps are off by seconds)
        mask = (df.index >= ts_floor) & (df.index < ts_floor + pd.Timedelta(minutes=5))
        if mask.any():
            return float(df[mask].iloc[0][col])

        return None

    def nearest_strike(self, symbol: str, expiry_date: date,
                       spot: float, step: int) -> int:
        """Round spot to nearest available strike for this expiry."""
        atm = int(round(spot / step) * step)
        # Check if we have data; if not, try adjacent strikes
        for offset in range(0, 6):
            for delta in ([0] if offset == 0 else [offset, -offset]):
                s = atm + delta * step
                key = (symbol, expiry_date, s)
                path = DATA_DIR / symbol / str(expiry_date) / f"{s}_CE.parquet"
                if path.exists():
                    return s
        return atm   # return ATM even if no file (caller handles None gracefully)

    def has_data(self, symbol: str, expiry_date: date) -> bool:
        """True if any options data exists for this expiry."""
        folder = DATA_DIR / symbol / str(expiry_date)
        return folder.exists() and any(folder.glob("*.parquet"))

    def coverage_report(self) -> pd.DataFrame:
        """Show which expiry dates have real data vs are missing."""
        if not DATA_DIR.exists():
            return pd.DataFrame(columns=["symbol", "expiry", "strikes"])
        rows = []
        for symbol_dir in DATA_DIR.iterdir():
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            for exp_dir in sorted(symbol_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                files = list(exp_dir.glob("*.parquet"))
                rows.append({
                    "symbol":  symbol,
                    "expiry":  exp_dir.name,
                    "strikes": len(files),
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "expiry", "strikes"]
        )
