#!/usr/bin/env python3
"""
price_watch.py — One-shot price alerts for any Kite option/index.

Polls LTP every 30 seconds and sends Telegram/WhatsApp alert when
the price crosses (or drops below) the specified level.

Usage:
    python price_watch.py NIFTY26MAY24100CE above 34
    python price_watch.py NIFTY26MAY23900PE below 50
    python price_watch.py NIFTY26MAY24100CE above 34  NIFTY26MAY23900PE below 50

Multiple alerts can be specified in one command.
"""

import sys
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from settings import settings
from kiteconnect import KiteConnect
from live_alerter import send_alert   # reuse same Telegram/WhatsApp sender

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("price_watch")

IST          = ZoneInfo("Asia/Kolkata")
POLL_SECS    = 30
MARKET_CLOSE = datetime.now(IST).replace(hour=15, minute=30, second=0, microsecond=0)


def parse_alerts(args: list[str]) -> list[dict]:
    """
    Parse groups of 3 args: TSYMBOL  above|below  PRICE
    e.g. ['NIFTY26MAY24100CE', 'above', '34', 'NIFTY26MAY23900PE', 'below', '50']
    """
    if len(args) % 3 != 0:
        print("Usage: python price_watch.py TSYMBOL above|below PRICE [TSYMBOL above|below PRICE ...]")
        sys.exit(1)
    alerts = []
    for i in range(0, len(args), 3):
        tsymbol   = args[i].upper()
        direction = args[i + 1].lower()
        level     = float(args[i + 2])
        if direction not in ("above", "below"):
            print(f"Direction must be 'above' or 'below', got: {direction}")
            sys.exit(1)
        alerts.append({
            "tsymbol":   tsymbol,
            "exchange":  "BFO" if tsymbol.startswith("SENSEX") else "NFO",
            "direction": direction,
            "level":     level,
            "fired":     False,
            "last_ltp":  None,
        })
    return alerts


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(0)

    alerts = parse_alerts(sys.argv[1:])

    kite = KiteConnect(api_key=settings.KITE_API_KEY)
    kite.set_access_token(settings.KITE_ACCESS_TOKEN)

    print("\n" + "─" * 50)
    print("  Price Watch — monitoring:")
    for a in alerts:
        arrow = "📈 above" if a["direction"] == "above" else "📉 below"
        print(f"    {a['tsymbol']:30s}  {arrow}  ₹{a['level']:.2f}")
    print(f"  Polling every {POLL_SECS}s until 3:30 PM IST")
    print("─" * 50 + "\n")

    while True:
        now = datetime.now(IST)
        if now >= MARKET_CLOSE:
            log.info("Market closed. Exiting price watch.")
            break

        pending = [a for a in alerts if not a["fired"]]
        if not pending:
            log.info("All alerts fired. Exiting.")
            break

        # Fetch LTPs for all pending symbols in one call
        keys = [f"{a['exchange']}:{a['tsymbol']}" for a in pending]
        try:
            quotes = kite.ltp(keys)
        except Exception as e:
            log.warning(f"LTP fetch failed: {e} — retrying in {POLL_SECS}s")
            time.sleep(POLL_SECS)
            continue

        for a in pending:
            key = f"{a['exchange']}:{a['tsymbol']}"
            q   = quotes.get(key)
            if not q:
                continue
            ltp = q["last_price"]
            a["last_ltp"] = ltp

            hit = (
                (a["direction"] == "above" and ltp >= a["level"]) or
                (a["direction"] == "below" and ltp <= a["level"])
            )

            arrow = "📈" if a["direction"] == "above" else "📉"
            log.info(f"{a['tsymbol']:30s}  CMP ₹{ltp:.2f}  "
                     f"(alert: {a['direction']} ₹{a['level']:.2f})  "
                     f"{'🔔 FIRED' if hit else ''}")

            if hit:
                a["fired"] = True
                msg = (
                    f"{arrow} *PRICE ALERT — {a['tsymbol']}*\n"
                    f"CMP ₹{ltp:.2f} crossed {a['direction']} ₹{a['level']:.2f}\n"
                    f"⏰ {now.strftime('%H:%M:%S')} IST"
                )
                send_alert(msg)

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
