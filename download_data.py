#!/usr/bin/env python3
"""
download_data.py — Download real Nifty 50 historical data from Kite Connect.

Run this after login.py has written a fresh access token to .env.
Data is saved as Parquet files in ./data/historical/ for fast backtesting reads.

Usage:
    python download_data.py              # last 5 years, daily + hourly
    python download_data.py --years 3   # last 3 years
    python download_data.py --daily      # daily only (faster)
"""

import argparse
import sys
from datetime import datetime, timedelta

from kite_data import load_kite_with_token, download_historical, add_features
from settings import settings


NIFTY_TOKEN = settings.NIFTY_SPOT_TOKEN   # 256265 — fixed forever by NSE/Kite


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Nifty historical data")
    parser.add_argument("--years",  type=int, default=5, help="Years of history (default: 5)")
    parser.add_argument("--daily",  action="store_true",  help="Download daily bars only")
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

    intervals = ["1day"] if args.daily else ["1day", "1hr", "15min"]

    print(f"\nNifty 50 download: {from_date.date()} → {to_date.date()}")
    print(f"Intervals: {', '.join(intervals)}")
    print(f"Token: {NIFTY_TOKEN}\n")

    for interval in intervals:
        print(f"[{interval}] fetching...")
        df = download_historical(
            kite,
            instrument_token=NIFTY_TOKEN,
            symbol="NIFTY50",
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )
        if df.empty:
            print(f"  ✗ No data returned for {interval}")
            continue

        df = add_features(df)
        print(f"  ✓ {len(df)} bars  |  cols: {list(df.columns[:6])} ...")

    print("\nDone. Run 'python backtest.py --real' to backtest on this data.")


if __name__ == "__main__":
    main()
