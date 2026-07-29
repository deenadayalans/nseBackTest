#!/usr/bin/env python3
"""
download_data.py — Download Nifty 50 and Sensex historical data from Kite Connect.

Run this after login.py has written a fresh access token to .env.
Data is saved as Parquet files in ./data/historical/ for fast backtesting reads.

Usage:
    python download_data.py              # Nifty + Sensex, all intervals incl. 3-min & 5-min
    python download_data.py --years 3   # last 3 years
    python download_data.py --daily      # daily bars only (fastest)
    python download_data.py --intraday   # 3-min + 5-min only (for OEH strategy)

Kite Connect intraday data limits (per single API call):
    minute / 3minute / 5minute / 10minute  → max 60 days per request
    15minute / 30minute                    → max 200 days per request
    60minute                               → max 400 days per request
    day                                    → max 2000 days per request
We auto-chunk requests to respect these limits.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from kite_data import load_kite_with_token, download_historical, add_features
from settings import settings

DATA_DIR = Path("./data/historical")

# ── Instrument tokens (fixed by NSE/Kite, never change) ──────────────────────
INSTRUMENTS = {
    "NIFTY50": 256265,    # Nifty 50 spot
    "SENSEX":  265,       # BSE Sensex spot
}


def _latest_date_in_parquet(symbol: str, interval: str) -> datetime | None:
    """Return the latest timestamp already stored, or None if file doesn't exist."""
    fname = DATA_DIR / f"{symbol}_{interval}.parquet"
    if not fname.exists():
        return None
    try:
        df = pd.read_parquet(fname)
        if df.empty:
            return None
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        return pd.Timestamp(idx.max()).to_pydatetime()
    except Exception:
        return None


def download_one(kite, token: int, symbol: str, interval: str,
                 from_date: datetime, to_date: datetime,
                 incremental: bool = True) -> None:
    """
    Download one symbol/interval.
    In incremental mode, skip dates already in the local parquet and
    only fetch what is missing.
    """
    if incremental:
        latest = _latest_date_in_parquet(symbol, interval)
        if latest is not None:
            # Start from the day after the last saved bar
            new_from = latest + timedelta(days=1)
            if new_from.date() >= to_date.date():
                print(f"  ✓ {symbol} [{interval}]: already up to date ({latest.date()})")
                return
            print(f"  [{interval}] incremental: {new_from.date()} → {to_date.date()}")
            from_date = new_from
        else:
            print(f"  [{interval}] first-time download: {from_date.date()} → {to_date.date()}")

    df_new = download_historical(
        kite,
        instrument_token=token,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
    )
    if df_new.empty:
        print(f"  ✗ No new data for {symbol} [{interval}]")
        return

    # Merge with existing data so we keep full history
    if incremental:
        fname = DATA_DIR / f"{symbol}_{interval}.parquet"
        if fname.exists():
            try:
                df_old = pd.read_parquet(fname)
                if df_old.index.tz is not None:
                    df_old.index = df_old.index.tz_localize(None)
                df_new = pd.concat([df_old, df_new])
                df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()
            except Exception as e:
                print(f"  ⚠ Could not merge with existing data: {e} — overwriting")

    df_new = add_features(df_new)
    print(f"  ✓ {symbol} [{interval}]: {len(df_new):,} total bars saved "
          f"(+{len(df_new):,} after merge)" if not incremental
          else f"  ✓ {symbol} [{interval}]: {len(df_new):,} total bars")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Nifty 50 + Sensex historical data (incremental by default)"
    )
    parser.add_argument("--years",    type=int, default=5,
                        help="Years of history for first-time download (default: 5)")
    parser.add_argument("--full",     action="store_true",
                        help="Force full re-download (ignore existing data)")
    parser.add_argument("--daily",    action="store_true",
                        help="Download daily bars only (fastest)")
    parser.add_argument("--intraday", action="store_true",
                        help="Download 3-min + 5-min only (for OEH strategy)")
    args = parser.parse_args()

    api_key      = settings.KITE_API_KEY
    access_token = settings.KITE_ACCESS_TOKEN

    if not api_key or api_key == "your_api_key_here":
        print("ERROR: KITE_API_KEY not set. Run 'cp .env.example .env' and fill it in.")
        sys.exit(1)
    if not access_token:
        print("ERROR: KITE_ACCESS_TOKEN not set. Run 'python login.py' first.")
        sys.exit(1)

    kite      = load_kite_with_token(api_key, access_token)
    to_date   = datetime.now()
    from_date = to_date - timedelta(days=365 * args.years)

    if args.daily:
        intervals = ["1day"]
    elif args.intraday:
        intervals = ["3min", "5min"]
    else:
        intervals = ["1day", "1hr", "15min", "5min", "3min"]

    incremental = not args.full
    mode_str    = "INCREMENTAL (new bars only)" if incremental else "FULL RE-DOWNLOAD"
    print(f"\nMode:        {mode_str}")
    print(f"Intervals:   {', '.join(intervals)}")
    print(f"Instruments: {', '.join(INSTRUMENTS.keys())}\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, token in INSTRUMENTS.items():
        print(f"── {symbol} (token {token}) ───────────────────────────────")
        for interval in intervals:
            download_one(kite, token, symbol, interval, from_date, to_date,
                         incremental=incremental)
        print()

    print("Done.")
    print("Run 'python oeh_reversal.py' to backtest with fresh data.")


if __name__ == "__main__":
    main()
