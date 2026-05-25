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
import sys
from datetime import datetime, timedelta

from kite_data import load_kite_with_token, download_historical, add_features
from settings import settings


# ── Instrument tokens (fixed by NSE/Kite, never change) ──────────────────────
INSTRUMENTS = {
    "NIFTY50": 256265,    # Nifty 50 spot
    "SENSEX":  265,       # BSE Sensex spot
}

def download_one(kite, token: int, symbol: str, interval: str,
                 from_date: datetime, to_date: datetime) -> None:
    """
    Download one symbol/interval. kite_data.download_historical already
    handles 58-day chunking internally, so we just pass the full date range.
    """
    df = download_historical(
        kite,
        instrument_token=token,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
    )
    if df.empty:
        print(f"  ✗ No data returned for {symbol} [{interval}]")
        return
    df = add_features(df)
    print(f"  ✓ {symbol} [{interval}]: {len(df):,} bars saved")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Nifty 50 + Sensex historical data (all intervals)"
    )
    parser.add_argument("--years",    type=int, default=5,
                        help="Years of history to download (default: 5)")
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
        # Full suite: daily + all intraday
        intervals = ["1day", "1hr", "15min", "5min", "3min"]

    print(f"\nDownloading: {from_date.date()} → {to_date.date()}")
    print(f"Intervals:   {', '.join(intervals)}")
    print(f"Instruments: {', '.join(INSTRUMENTS.keys())}\n")

    for symbol, token in INSTRUMENTS.items():
        print(f"── {symbol} (token {token}) ───────────────────────────────")
        for interval in intervals:
            print(f"  [{interval}]")
            download_one(kite, token, symbol, interval, from_date, to_date)
        print()

    print("Done.")
    print("Run 'python open_high_strategy.py' to backtest OEH/OEL on 3-min + 5-min data.")
    print("Run 'python backtest.py --real' for daily/hourly strategies.")


if __name__ == "__main__":
    main()
