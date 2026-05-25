# data/kite_data.py
# Layer 1: Historical data download from Zerodha Kite API
# Stores locally as Parquet for fast backtesting reads
# Also computes technical features used by strategies

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from kiteconnect import KiteConnect

# ── Kite session helpers ──────────────────────────────────────────────────────

def get_kite_session(api_key: str, api_secret: str, request_token: str) -> KiteConnect:
    """
    Full login flow. Run once to get access_token, then save it.
    request_token comes from the redirect URL after browser login.
    
    Usage:
        1. Open: https://kite.trade/connect/login?api_key=YOUR_KEY&v=3
        2. After login, copy the request_token from the redirect URL
        3. Call this function with that token
        4. Save the returned access_token for reuse (valid until next day)
    """
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    kite.set_access_token(data["access_token"])
    print(f"Logged in as: {data['user_id']} | Token expires at midnight IST")
    return kite


def load_kite_with_token(api_key: str, access_token: str) -> KiteConnect:
    """Reconnect using a saved access_token (no browser needed)."""
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# ── Instrument lookup ─────────────────────────────────────────────────────────

def get_nifty_instruments(kite: KiteConnect) -> pd.DataFrame:
    """
    Return all Nifty-related instruments (spot + futures + options).
    Instrument tokens are needed for historical data calls.
    """
    instruments = kite.instruments("NSE") + kite.instruments("NFO")
    df = pd.DataFrame(instruments)
    
    # Filter to Nifty 50 instruments
    nifty = df[df["name"].str.contains("NIFTY", na=False) & 
               ~df["name"].str.contains("MIDCAP|BANK|FIN|IT|AUTO|METAL|PHARMA|REALTY", na=False)]
    
    return nifty[["instrument_token", "tradingsymbol", "name", "expiry", 
                   "instrument_type", "exchange"]].sort_values("expiry")


# ── Historical data download ─────────────────────────────────────────────────

INTERVAL_MAP = {
    "1min":  "minute",
    "3min":  "3minute",
    "5min":  "5minute",
    "15min": "15minute",
    "30min": "30minute",
    "1hr":   "60minute",
    "1day":  "day",
}

def download_historical(
    kite: KiteConnect,
    instrument_token: int,
    symbol: str,
    from_date: datetime,
    to_date: datetime,
    interval: str = "1day",
    save_dir: str = "./data/historical",
) -> pd.DataFrame:
    """
    Download OHLCV data from Kite. Handles chunking (Kite has per-request limits).
    
    Limits per request:
        minute data  → max 60 days
        hourly data  → max 400 days
        daily data   → max 2000 candles (no limit in practice)
    """
    kite_interval = INTERVAL_MAP.get(interval, interval)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    chunk_days = 58 if "min" in interval else 380  # stay within limits

    current = from_date
    while current < to_date:
        chunk_end = min(current + timedelta(days=chunk_days), to_date)
        print(f"  Fetching {symbol} {interval}: {current.date()} → {chunk_end.date()}")
        
        try:
            records = kite.historical_data(
                instrument_token,
                from_date=current,
                to_date=chunk_end,
                interval=kite_interval,
                continuous=False,
            )
            all_chunks.extend(records)
            time.sleep(0.35)  # be polite to the API (rate limit: ~3 req/sec)
        except Exception as e:
            print(f"  Warning: chunk failed ({e}), skipping...")
        
        current = chunk_end + timedelta(days=1)
    
    if not all_chunks:
        print(f"  No data returned for {symbol}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_chunks)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.columns = [c.lower() for c in df.columns]
    # Strip timezone — backtrader requires naive datetimes
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Save as Parquet (faster + smaller than CSV)
    fname = f"{save_dir}/{symbol}_{interval}.parquet"
    df.to_parquet(fname)
    print(f"  Saved {len(df)} rows → {fname}")
    
    return df


def load_historical(symbol: str, interval: str = "1day",
                    data_dir: str = "./data/historical") -> pd.DataFrame:
    """Load previously downloaded data from local Parquet."""
    fname = f"{data_dir}/{symbol}_{interval}.parquet"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No data file found: {fname}. Run download first.")
    df = pd.read_parquet(fname)
    # Backtrader requires timezone-naive datetime index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators used by strategies and LLM context.
    All features are added as new columns — original OHLCV is preserved.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend indicators ──────────────────────────────────────────
    df["ema9"]  = c.ewm(span=9,  adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"]= c.ewm(span=200,adjust=False).mean()
    
    # Trend direction: +1 bullish, -1 bearish
    df["trend"] = np.where(df["ema21"] > df["ema50"], 1, -1)

    # ── Momentum ──────────────────────────────────────────────────
    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Volatility ────────────────────────────────────────────────
    # ATR (14) — used for stop loss sizing
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / c * 100  # ATR as % of price

    # Bollinger Bands (20, 2σ)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── Volume ────────────────────────────────────────────────────
    df["vol_sma20"] = v.rolling(20).mean()
    # Nifty spot has zero volume from Kite; guard against 0/0 → NaN
    df["vol_ratio"] = np.where(df["vol_sma20"] > 0, v / df["vol_sma20"], 1.0)

    # ── Price structure ───────────────────────────────────────────
    df["daily_return"] = c.pct_change() * 100
    df["high_low_pct"] = (h - l) / c * 100   # intraday range %

    # ── Market regime (simple: based on 200 EMA) ─────────────────
    df["regime"] = np.where(c > df["ema200"], "bull", "bear")

    return df.dropna()


# ── Quick-start download helper ───────────────────────────────────────────────

def download_nifty_dataset(kite: KiteConnect, years_back: int = 3) -> dict:
    """
    Download a full Nifty 50 dataset (daily + hourly) for backtesting.
    Returns dict of DataFrames.
    
    This takes a few minutes the first time. After that, load from local files.
    """
    to_date   = datetime.now()
    from_date = to_date - timedelta(days=365 * years_back)
    
    # Nifty 50 spot instrument token = 256265 (fixed by NSE/Kite)
    NIFTY_TOKEN = 256265
    SYMBOL = "NIFTY50"
    
    print(f"Downloading Nifty 50 data: {from_date.date()} → {to_date.date()}")
    
    datasets = {}
    for interval in ["1day", "1hr", "15min"]:
        print(f"\n[{interval}]")
        df = download_historical(kite, NIFTY_TOKEN, SYMBOL, from_date, to_date, interval)
        if not df.empty:
            df = add_features(df)
            datasets[interval] = df
    
    return datasets


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick feature-engineering test on synthetic data
    print("Testing feature pipeline with synthetic data...")
    
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    price = 12000 + np.cumsum(np.random.randn(500) * 50)
    
    test_df = pd.DataFrame({
        "open":   price + np.random.randn(500) * 10,
        "high":   price + np.abs(np.random.randn(500) * 30),
        "low":    price - np.abs(np.random.randn(500) * 30),
        "close":  price,
        "volume": np.random.randint(50_000_000, 200_000_000, 500).astype(float),
    }, index=dates)
    
    result = add_features(test_df)
    
    print(f"\nFeatures added: {list(result.columns)}")
    print(f"\nLast 3 rows:")
    print(result[["close","ema21","rsi14","atr14","bb_pct","regime"]].tail(3).to_string())
    print("\nData pipeline OK.")
