#!/usr/bin/env python3
"""
download_options.py — Download real 5-min OHLC for Nifty & Sensex options

Data is stored as:
    data/options/NIFTY50/<YYYY-MM-DD>/<strike>_CE.parquet
    data/options/NIFTY50/<YYYY-MM-DD>/<strike>_PE.parquet
    data/options/SENSEX/<YYYY-MM-DD>/<strike>_CE.parquet
    data/options/SENSEX/<YYYY-MM-DD>/<strike>_PE.parquet

Usage:
    python download_options.py          # full historical (slow, hits Kite limits)
    python download_options.py --today  # only active/current expiries (fast, run daily)
"""

import sys
import time
import os
import pandas as pd
from datetime import datetime, timedelta, date
from pathlib import Path

from kite_data import load_kite_with_token, load_historical
from settings import settings

API_KEY      = settings.KITE_API_KEY
ACCESS_TOKEN = settings.KITE_ACCESS_TOKEN

# ── Config ────────────────────────────────────────────────────────────────────

STRIKES_EACH_SIDE = 5   # ATM ± 5 → 11 CE + 11 PE per expiry

STRIKE_STEP = {
    "NIFTY50": 50,
    "SENSEX":  100,
}

EXPIRY_DOW = {
    "NIFTY50": 1,   # Tuesday
    "SENSEX":  3,   # Thursday
}

FROM_DATE = datetime(2021, 6, 1)
TO_DATE   = datetime.today()

DATA_DIR  = "./data/options"

# ── Helpers ───────────────────────────────────────────────────────────────────

def all_expiry_dates(expiry_dow: int) -> list[datetime.date]:
    """Return every date between FROM_DATE and TO_DATE on the given weekday."""
    dates = []
    d = FROM_DATE.date()
    while d <= TO_DATE.date():
        if d.weekday() == expiry_dow:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def get_instruments_df(kite, exchange: str) -> pd.DataFrame:
    """Download full instruments list and cache it as CSV."""
    cache = f"./data/{exchange}_instruments.csv"
    if os.path.exists(cache):
        age_hours = (time.time() - os.path.getmtime(cache)) / 3600
        if age_hours < 20:   # reuse if fetched today
            return pd.read_csv(cache, parse_dates=["expiry"])
    print(f"  Downloading {exchange} instruments list…")
    insts = kite.instruments(exchange)
    df = pd.DataFrame(insts)
    df["expiry"] = pd.to_datetime(df["expiry"])
    Path("./data").mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def snap_to_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def strikes_around_atm(atm: int, step: int, n: int) -> list[int]:
    return [atm + i * step for i in range(-n, n + 1)]


def option_parquet_path(symbol: str, expiry_date, strike: int, opt_type: str) -> Path:
    return Path(DATA_DIR) / symbol / str(expiry_date) / f"{strike}_{opt_type}.parquet"


def already_downloaded(symbol: str, expiry_date, strikes: list[int]) -> bool:
    """True if ALL strikes (CE + PE) for this expiry are already on disk."""
    for s in strikes:
        for t in ("CE", "PE"):
            if not option_parquet_path(symbol, expiry_date, s, t).exists():
                return False
    return True


def real_atm(symbol: str, step: int) -> int:
    """Get actual ATM from today's downloaded spot close. Falls back to hardcoded estimate."""
    fallback = {"NIFTY50": 24000, "SENSEX": 80000}
    try:
        df = load_historical(symbol, "1day")
        if df is None or df.empty:
            raise ValueError("empty")
        last_close = float(df["close"].iloc[-1])
        return snap_to_strike(last_close, step)
    except Exception:
        return snap_to_strike(fallback[symbol], step)


# ── Download ──────────────────────────────────────────────────────────────────

def download_expiry(kite, symbol: str, all_opts: pd.DataFrame,
                    expiry_date: date, atm: int, step: int,
                    force: bool = False) -> int:
    """
    Download 5-min OHLC for CE + PE, ATM ± N strikes expiring on expiry_date.
    force=True re-downloads even if files exist (use for active/today's expiry).
    Returns number of new files written, or -1 if contract not in instruments list.
    """
    strikes    = strikes_around_atm(atm, step, STRIKES_EACH_SIDE)
    index_name = {"NIFTY50": "NIFTY", "SENSEX": "SENSEX"}[symbol]

    if not force and already_downloaded(symbol, expiry_date, strikes):
        return 0

    # Fetch from start of the expiry week (covers pre-expiry day too)
    fetch_from = datetime.combine(expiry_date - timedelta(days=7), datetime.min.time())
    fetch_to   = datetime.combine(expiry_date, datetime.min.time().replace(hour=15, minute=30))
    # For active contracts don't go beyond now
    fetch_to   = min(fetch_to, datetime.now())

    downloaded = 0
    for opt_type in ("CE", "PE"):
        mask = (
            (all_opts["name"] == index_name) &
            (all_opts["instrument_type"] == opt_type) &
            (all_opts["expiry"].dt.date == expiry_date)
        )
        expiry_opts = all_opts[mask]

        if expiry_opts.empty:
            return -1   # contract expired → no longer in Kite instruments list

        for strike in strikes:
            out_path = option_parquet_path(symbol, expiry_date, strike, opt_type)
            if out_path.exists() and not force:
                continue

            row = expiry_opts[expiry_opts["strike"] == strike]
            if row.empty:
                continue

            token = int(row.iloc[0]["instrument_token"])
            tsym  = row.iloc[0]["tradingsymbol"]

            try:
                records = kite.historical_data(
                    token,
                    from_date=fetch_from,
                    to_date=fetch_to,
                    interval="5minute",
                    continuous=False,
                )
                time.sleep(0.35)
            except Exception as e:
                print(f"    [{tsym}] fetch error: {e}")
                continue

            if not records:
                continue

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]

            # Upsert: merge with existing data so we never lose earlier bars
            if out_path.exists() and force:
                existing = pd.read_parquet(out_path)
                df = pd.concat([existing, df]).sort_index()
                df = df[~df.index.duplicated(keep="last")]

            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path)
            downloaded += 1
            print(f"    {tsym}  {len(df)} bars  → {out_path}")

    return downloaded


def active_expiries(expiry_dow: int) -> list[date]:
    """Return the next 2 expiry dates from today (currently active contracts)."""
    today   = date.today()
    results = []
    d = today
    while len(results) < 2:
        if d.weekday() == expiry_dow:
            results.append(d)
        d += timedelta(days=1)
    return results


def main():
    today_mode = "--today" in sys.argv   # fast daily refresh vs full historical

    if not API_KEY or not ACCESS_TOKEN:
        print("ERROR: KITE_API_KEY or KITE_ACCESS_TOKEN missing in .env")
        print("Run  python login.py  first to get a fresh access token.")
        return

    print("Connecting to Kite…")
    kite = load_kite_with_token(API_KEY, ACCESS_TOKEN)

    print("Fetching NFO/BFO instruments list…")
    nfo_df = get_instruments_df(kite, "NFO")
    bse_df = get_instruments_df(kite, "BFO")
    all_opts = pd.concat([nfo_df, bse_df], ignore_index=True)
    all_opts["expiry"] = pd.to_datetime(all_opts["expiry"])

    for symbol in ["NIFTY50", "SENSEX"]:
        step       = STRIKE_STEP[symbol]
        expiry_dow = EXPIRY_DOW[symbol]
        atm        = real_atm(symbol, step)

        if today_mode:
            # Only download active (unexpired) contracts — very fast
            expiries = active_expiries(expiry_dow)
            print(f"\n{'='*60}")
            print(f"  {symbol} — today mode: {len(expiries)} active expiries")
            print(f"  Real ATM from spot data: {atm}  (± {STRIKES_EACH_SIDE} strikes)")
            print(f"{'='*60}")
            for exp_date in expiries:
                print(f"  Downloading {exp_date} (force-refresh)…")
                n = download_expiry(kite, symbol, all_opts, exp_date, atm, step, force=True)
                if n == -1:
                    print(f"  {exp_date}: contract not found in instruments list")
                else:
                    print(f"  {exp_date}: {n} files written")
        else:
            # Full historical sweep — skips cached, logs expired
            expiries  = all_expiry_dates(expiry_dow)
            total_dl  = 0
            skipped   = 0
            print(f"\n{'='*60}")
            print(f"  {symbol}: {len(expiries)} expiry dates to check")
            print(f"  ATM estimate: {atm} ± {STRIKES_EACH_SIDE} × {step}")
            print(f"{'='*60}")
            for i, exp_date in enumerate(expiries):
                strikes = strikes_around_atm(atm, step, STRIKES_EACH_SIDE)
                is_active = exp_date >= date.today()
                if already_downloaded(symbol, exp_date, strikes) and not is_active:
                    continue
                n = download_expiry(kite, symbol, all_opts, exp_date, atm, step,
                                    force=is_active)
                if n == -1:
                    skipped += 1
                else:
                    total_dl += n
                    if n > 0:
                        print(f"  {exp_date}  [{i+1}/{len(expiries)}]  +{n} files")

            print(f"\n  {symbol}: {total_dl} new files | {skipped} expired (unavailable)")
            if skipped:
                print(f"  NOTE: Expired contract history requires a paid provider (True Data).")

    print("\nDone. Run  python oeh_reversal.py  to backtest with real option prices.")


if __name__ == "__main__":
    main()
