#!/usr/bin/env python3
"""
start.py — One command to run your entire trading day.

Flow:
  1. Login (browser OAuth — the only manual step)
  2. Live alerter starts automatically (signals until 12:30, EOD at 3:30)
  3. After market closes → spot data download (Nifty + Sensex)
  4. Options data download (today's active expiries)
  5. Backtest re-run with fresh real data
  6. Summary printed

Usage:
    python start.py          # full day (login + alerts + EOD tasks)
    python start.py --eod    # skip login + alerter, just run EOD tasks
    python start.py --login  # just login (refresh token), nothing else
"""

import sys
import os
import re
import subprocess
import webbrowser
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ── Colour helpers (terminal output) ──────────────────────────────────────────
G = "\033[92m"   # green
Y = "\033[93m"   # yellow
R = "\033[91m"   # red
B = "\033[94m"   # blue
W = "\033[0m"    # reset

def banner(msg: str, colour=B):
    width = 62
    print(f"\n{colour}{'='*width}{W}")
    print(f"{colour}  {msg}{W}")
    print(f"{colour}{'='*width}{W}\n")

def ok(msg):   print(f"{G}  ✓  {msg}{W}")
def warn(msg): print(f"{Y}  ⚠  {msg}{W}")
def err(msg):  print(f"{R}  ✗  {msg}{W}")
def info(msg): print(f"     {msg}")

# ── Step helpers ───────────────────────────────────────────────────────────────

def run_step(label: str, cmd: list[str]) -> bool:
    """Run a subprocess and return True on success."""
    info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        ok(f"{label} done")
        return True
    else:
        err(f"{label} failed (exit {result.returncode})")
        return False


# ── Login (browser OAuth) ─────────────────────────────────────────────────────

ENV_FILE = Path(".env")

def _update_env_token(token: str) -> None:
    if not ENV_FILE.exists():
        ENV_FILE.write_text(f"KITE_ACCESS_TOKEN={token}\n")
        return
    text = ENV_FILE.read_text()
    if "KITE_ACCESS_TOKEN" in text:
        text = re.sub(r"KITE_ACCESS_TOKEN=.*", f"KITE_ACCESS_TOKEN={token}", text)
    else:
        text = text.rstrip("\n") + f"\nKITE_ACCESS_TOKEN={token}\n"
    ENV_FILE.write_text(text)


def do_login() -> bool:
    banner("STEP 1 — Kite Login")
    try:
        from kiteconnect import KiteConnect
        from settings import settings
    except ImportError as e:
        err(f"Import error: {e}. Run  pip install -r requirements.txt")
        return False

    api_key    = settings.KITE_API_KEY
    api_secret = settings.KITE_API_SECRET

    if not api_key:
        err("KITE_API_KEY not set in .env")
        return False

    kite      = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print(f"  Opening browser to Zerodha login…")
    print(f"  URL: {login_url}\n")
    try:
        webbrowser.open(login_url)
    except Exception:
        pass

    print("  After logging in, Zerodha redirects to a URL like:")
    print("    https://127.0.0.1/?request_token=XXXXXXXX&…\n")
    raw = input("  Paste the redirect URL (or just the request_token): ").strip()

    match = re.search(r"request_token=([A-Za-z0-9]+)", raw)
    token_input = match.group(1) if match else raw

    try:
        data         = kite.generate_session(token_input, api_secret=api_secret)
        access_token = data["access_token"]
        _update_env_token(access_token)
        ok(f"Logged in as {data['user_id']} ({data.get('user_name', '')})")
        ok("Access token saved to .env")
        return True
    except Exception as e:
        err(f"Login failed: {e}")
        return False


# ── Live alerter (blocks until market close) ──────────────────────────────────

def do_live_alerter() -> None:
    banner("STEP 2 — Live Alerter (runs until 3:30 PM IST)")
    now = datetime.now(IST).time()
    market_close = dtime(15, 30)

    if now >= market_close:
        warn("Market already closed — skipping live alerter")
        return

    print("  Live alerter is running. It will exit automatically at 3:30 PM.")
    print("  You will receive Telegram/WhatsApp alerts for OEH setups.")
    print("  Press Ctrl+C here to stop early.\n")

    # Run in the same process so Ctrl+C works naturally
    try:
        # Reload settings with fresh token from .env
        import importlib
        import settings as _s
        importlib.reload(_s)
        from live_alerter import OEHAlerter
        alerter = OEHAlerter()
        alerter.run()
    except KeyboardInterrupt:
        warn("Alerter stopped manually")
    except Exception as e:
        err(f"Alerter error: {e}")


# ── EOD tasks (after market close) ────────────────────────────────────────────

def do_eod() -> None:
    banner("STEP 3 — End-of-Day Data Tasks", colour=Y)

    print("  [3a] Downloading today's spot candles (Nifty + Sensex)…")
    run_step("Spot data", [sys.executable, "download_data.py"])

    print("\n  [3b] Downloading today's options data (active expiries)…")
    run_step("Options data", [sys.executable, "download_options.py", "--today"])

    print("\n  [3c] Re-running backtest with fresh data…")
    run_step("Backtest", [sys.executable, "oeh_reversal.py"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = set(sys.argv[1:])

    banner("NSE OEH Trading System", colour=G)
    print(f"  {datetime.now(IST).strftime('%A, %d %b %Y  %H:%M IST')}\n")

    # ── Mode: just login ──────────────────────────────────────────────────────
    if "--login" in args:
        do_login()
        return

    # ── Mode: just EOD tasks ──────────────────────────────────────────────────
    if "--eod" in args:
        do_eod()
        return

    # ── Full day mode ─────────────────────────────────────────────────────────
    if not do_login():
        err("Login failed — cannot continue.")
        sys.exit(1)

    do_live_alerter()   # blocks until 3:30 PM

    do_eod()            # runs immediately after alerter exits

    banner("Day Complete!", colour=G)
    print("  Your journal is in the  journal/  folder.")
    print("  Tomorrow, just run:  python start.py\n")


if __name__ == "__main__":
    main()
