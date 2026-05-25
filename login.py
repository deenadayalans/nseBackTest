#!/usr/bin/env python3
"""
login.py — Zerodha Kite Connect daily login.

Run this once each morning before trading/downloading data.
It writes the fresh access_token into your .env file so all
other scripts can pick it up automatically via settings.py.

Usage:
    python login.py
"""

import re
import sys
import webbrowser
from pathlib import Path

from kiteconnect import KiteConnect
from settings import settings

ENV_FILE = Path(".env")


def _update_env_token(token: str) -> None:
    """Write/overwrite KITE_ACCESS_TOKEN in .env."""
    if not ENV_FILE.exists():
        ENV_FILE.write_text(f"KITE_ACCESS_TOKEN={token}\n")
        return

    text = ENV_FILE.read_text()
    if "KITE_ACCESS_TOKEN" in text:
        text = re.sub(r"KITE_ACCESS_TOKEN=.*", f"KITE_ACCESS_TOKEN={token}", text)
    else:
        text = text.rstrip("\n") + f"\nKITE_ACCESS_TOKEN={token}\n"
    ENV_FILE.write_text(text)


def main() -> None:
    api_key    = settings.KITE_API_KEY
    api_secret = settings.KITE_API_SECRET

    if not api_key or api_key == "your_api_key_here":
        print("ERROR: KITE_API_KEY not set in .env")
        print("  1. Copy .env.example → .env")
        print("  2. Fill in KITE_API_KEY and KITE_API_SECRET")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print("\n" + "="*60)
    print("Kite Connect — Daily Login")
    print("="*60)
    print(f"\nOpening browser to:\n  {login_url}\n")

    try:
        webbrowser.open(login_url)
    except Exception:
        print("(Could not auto-open browser — paste the URL above manually.)")

    print("After logging in, Zerodha redirects you to a URL like:")
    print("  https://127.0.0.1/?request_token=XXXXXXXX&action=login&status=success")
    print()
    request_token = input("Paste the full redirect URL (or just the request_token): ").strip()

    # Accept full URL or bare token
    match = re.search(r"request_token=([A-Za-z0-9]+)", request_token)
    if match:
        request_token = match.group(1)

    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]

    _update_env_token(access_token)

    print(f"\n✓  Logged in as: {data['user_id']} ({data.get('user_name', '')})")
    print(f"✓  Access token saved to .env  (valid until midnight IST)")
    print(f"\nYou can now run:")
    print(f"  python download_data.py   ← fetch real Nifty data")
    print(f"  python backtest.py        ← run strategies on real data")


if __name__ == "__main__":
    main()
