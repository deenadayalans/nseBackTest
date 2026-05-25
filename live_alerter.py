#!/usr/bin/env python3
"""
live_alerter.py — Real-time OEH Signal Alerter with WhatsApp notifications

Flow:
  09:20 AM → Morning briefing on WhatsApp:
               • Daily trend (SuperTrend, EMA-50), previous close
               • Expected ATM, strikes being watched, sentiment
  09:15–12:30 → Watch for OEH on Nifty spot (5-min candles)
               • When OEH forms → alert with option chain OEH status
               • When reversal confirmed → ENTRY signal with exact levels

Setup (one-time):
  1. Add to .env:
       WA_PHONE=+91XXXXXXXXXX       (your WhatsApp number with country code)
       WA_APIKEY=XXXXXXX            (from CallMeBot, see below)
  2. Register CallMeBot (free):
       a. Save +34 644 59 77 31 in your contacts as "CallMeBot"
       b. Send:  I allow callmebot to send me messages
       c. You'll receive your API key via WhatsApp
  3. Run:  python live_alerter.py

Run:
    python live_alerter.py
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime, time as dtime, timedelta, date
from zoneinfo import ZoneInfo
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from kiteconnect import KiteTicker, KiteConnect

from settings import settings
from kite_data import load_historical, load_kite_with_token
from trade_journal import TradeJournal
from oeh_reversal import (
    compute_indicators, _supertrend,
    DAILY_ST_ATR, DAILY_ST_MULT,
    OEH_TOLERANCE, MIN_DROP, REVERSAL_BARS,
    MIN_SPOT_DROP, EMA_FAST, EMA_TREND,
    SCHEDULE,
)

# ── Timezone ──────────────────────────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")

# ── Timing ────────────────────────────────────────────────────────────────────
MORNING_BRIEF_TIME = dtime(9, 20)
OEH_WINDOW_START   = dtime(9, 20)
OEH_WINDOW_END     = dtime(12, 30)    # 9:15–12:30 works well per user's experience
MARKET_CLOSE       = dtime(15, 30)
CANDLE_MINS        = 5

# ── Instruments ───────────────────────────────────────────────────────────────
NIFTY_SPOT_TOKEN  = 256265
SENSEX_SPOT_TOKEN = 265

NIFTY_STRIKE_STEP  = 50
SENSEX_STRIKE_STEP = 100
STRIKES_EACH_SIDE  = 5    # ATM ± 5 → 11 CE + 11 PE strikes

# ── OEH detection parameters (must match oeh_reversal.py) ────────────────────
OEH_TOL_PCT  = OEH_TOLERANCE   # 0.05%
MIN_DROP_PCT = MIN_DROP         # 0.15%

# ── WhatsApp (CallMeBot) ──────────────────────────────────────────────────────
WA_PHONE  = settings.WA_PHONE  or os.environ.get("WA_PHONE",  "")
WA_APIKEY = settings.WA_APIKEY or os.environ.get("WA_APIKEY", "")
CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"

# ── Telegram (recommended — instant, free, no registration wait) ──────────────
# Setup (2 min):
#   1. Open Telegram → search @BotFather → send /newbot → follow prompts
#   2. Copy the token it gives you  →  TG_BOT_TOKEN=123456:ABCdef...
#   3. Start a chat with your new bot (search its name, press Start)
#   4. Open: https://api.telegram.org/bot<TOKEN>/getUpdates
#      Find "chat":{"id": XXXXXXXXX}  →  TG_CHAT_ID=XXXXXXXXX
TG_BOT_TOKEN = settings.TG_BOT_TOKEN or os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID   = settings.TG_CHAT_ID   or os.environ.get("TG_CHAT_ID",   "")
TELEGRAM_URL  = "https://api.telegram.org/bot{token}/sendMessage"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oeh_alerter")


# ══════════════════════════════════════════════════════════════════════════════
# WhatsApp
# ══════════════════════════════════════════════════════════════════════════════

def _md_to_html(text: str) -> str:
    """
    Convert simple *bold* Markdown to <b>bold</b> HTML for Telegram.
    Also preserves newlines. Telegram HTML is more forgiving than its Markdown parser.
    """
    import re
    # *bold* → <b>bold</b>  (non-greedy, single line)
    text = re.sub(r"\*(.+?)\*", r"<b>\1</b>", text)
    return text


def send_telegram(message: str) -> bool:
    """Send via Telegram Bot API using HTML formatting (avoids Markdown parse errors)."""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return False
    try:
        url  = TELEGRAM_URL.format(token=TG_BOT_TOKEN)
        resp = requests.post(url, json={
            "chat_id":    TG_CHAT_ID,
            "text":       _md_to_html(message),
            "parse_mode": "HTML",
        }, timeout=10)
        ok = resp.status_code == 200
        if not ok:
            log.warning(f"Telegram send failed: {resp.status_code} {resp.text[:120]}")
        return ok
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False


def send_whatsapp(message: str) -> bool:
    """Send via CallMeBot WhatsApp API."""
    if not WA_PHONE or not WA_APIKEY:
        return False
    try:
        resp = requests.get(CALLMEBOT_URL, params={
            "phone":  WA_PHONE,
            "text":   message,
            "apikey": WA_APIKEY,
        }, timeout=10)
        ok = resp.status_code == 200
        if not ok:
            log.warning(f"WhatsApp send failed: {resp.status_code} {resp.text[:80]}")
        return ok
    except Exception as e:
        log.error(f"WhatsApp error: {e}")
        return False


def send_alert(message: str) -> bool:
    """
    Send to ALL configured channels simultaneously.
    Both Telegram AND WhatsApp will receive every alert if both are set up.
    Falls back to console only if neither is configured.
    """
    sent = False

    if TG_BOT_TOKEN and TG_CHAT_ID:
        sent = send_telegram(message) or sent

    if WA_PHONE and WA_APIKEY and WA_APIKEY not in ("XXXXXXX", ""):
        sent = send_whatsapp(message) or sent

    if not sent:
        # Console fallback — never miss a signal
        print("\n" + "═"*60)
        print(message)
        print("═"*60 + "\n")

    return sent


# ══════════════════════════════════════════════════════════════════════════════
# Instruments — find live option tokens for ATM ± N strikes
# ══════════════════════════════════════════════════════════════════════════════

def get_option_tokens(kite: KiteConnect, symbol: str, atm_spot: float) -> dict:
    """
    Return {token: {"strike": int, "type": "CE"/"PE", "symbol": str}}
    for ATM ± STRIKES_EACH_SIDE × step, nearest weekly expiry.
    """
    exchange   = "NFO"   # Nifty options on NFO; Sensex on BFO
    name_map   = {"NIFTY50": "NIFTY", "SENSEX": "SENSEX"}
    step       = NIFTY_STRIKE_STEP if symbol == "NIFTY50" else SENSEX_STRIKE_STEP
    index_name = name_map[symbol]

    insts = kite.instruments(exchange)
    df    = pd.DataFrame(insts)
    df["expiry"] = pd.to_datetime(df["expiry"])

    # Filter to calls and puts for this index
    opts = df[(df["name"] == index_name) & (df["instrument_type"].isin(["CE", "PE"]))].copy()
    if opts.empty and symbol == "SENSEX":
        # Sensex options may be on BFO
        insts2 = kite.instruments("BFO")
        df2    = pd.DataFrame(insts2)
        df2["expiry"] = pd.to_datetime(df2["expiry"])
        opts = df2[(df2["name"] == "SENSEX") & (df2["instrument_type"].isin(["CE", "PE"]))].copy()

    if opts.empty:
        log.warning(f"No options found for {symbol}")
        return {}

    # Nearest expiry (this week's or next)
    today     = datetime.now(IST).date()
    exp_dow   = SCHEDULE[symbol][0]   # expiry weekday
    future    = opts[opts["expiry"].dt.date >= today].copy()
    if future.empty:
        return {}
    nearest_expiry = future["expiry"].min().date()
    this_expiry    = future[future["expiry"].dt.date == nearest_expiry]

    atm = int(round(atm_spot / step) * step)
    strikes = [atm + i * step for i in range(-STRIKES_EACH_SIDE, STRIKES_EACH_SIDE + 1)]

    result = {}
    for _, row in this_expiry.iterrows():
        s = int(row["strike"])
        if s in strikes:
            result[int(row["instrument_token"])] = {
                "strike":  s,
                "type":    row["instrument_type"],
                "tsymbol": row["tradingsymbol"],
                "expiry":  nearest_expiry,
            }

    log.info(f"{symbol}: {len(result)} option tokens for expiry {nearest_expiry} "
             f"(ATM {atm} ± {STRIKES_EACH_SIDE}×{step})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Candle builder — accumulates ticks into 5-min OHLCV bars
# ══════════════════════════════════════════════════════════════════════════════

class CandleBuilder:
    """Builds 5-min candles from individual ticks."""

    def __init__(self, candle_minutes: int = 5):
        self.minutes = candle_minutes
        self._bars: dict[int, dict] = {}    # token → current incomplete bar
        self._history: dict[int, list] = defaultdict(list)   # token → completed bars

    def _period_start(self, ts: datetime) -> datetime:
        """Floor timestamp to the nearest candle period."""
        m = (ts.minute // self.minutes) * self.minutes
        return ts.replace(minute=m, second=0, microsecond=0)

    def on_tick(self, token: int, ltp: float, ts: datetime) -> dict | None:
        """
        Feed a tick. Returns the completed candle dict if a new period started,
        else None.
        """
        period = self._period_start(ts)
        completed = None

        if token not in self._bars:
            self._bars[token] = {"period": period, "open": ltp, "high": ltp,
                                  "low": ltp, "close": ltp, "volume": 0}
        else:
            bar = self._bars[token]
            if period > bar["period"]:
                # New candle period — emit the completed bar
                completed = bar.copy()
                self._history[token].append(completed)
                self._bars[token] = {"period": period, "open": ltp, "high": ltp,
                                      "low": ltp, "close": ltp, "volume": 0}
            else:
                bar["high"]  = max(bar["high"], ltp)
                bar["low"]   = min(bar["low"],  ltp)
                bar["close"] = ltp

        return completed

    def history_df(self, token: int, limit: int = 200) -> pd.DataFrame:
        """Return last N completed candles as a DataFrame."""
        bars = self._history[token][-limit:]
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df = df.rename(columns={"period": "date"})
        df = df.set_index("date")
        return df

    def last_bar(self, token: int) -> dict | None:
        """Return the most recently COMPLETED bar."""
        hist = self._history.get(token, [])
        return hist[-1] if hist else None

    def current_bar(self, token: int) -> dict | None:
        return self._bars.get(token)


# ══════════════════════════════════════════════════════════════════════════════
# Market sentiment — morning briefing
# ══════════════════════════════════════════════════════════════════════════════

def build_morning_brief(symbol: str, today_date: date) -> str:
    """Build a morning briefing message using locally stored daily data."""
    try:
        df1d = load_historical(symbol, "1day")
        if df1d.index.tz:
            df1d.index = df1d.index.tz_localize(None)

        past = df1d[df1d.index.date < today_date].copy()
        if past.empty:
            return f"⚠️ No historical data for {symbol}"

        prev_close  = past["close"].iloc[-1]
        prev_open   = past["open"].iloc[-1]
        prev_change = (prev_close - past["close"].iloc[-2]) / past["close"].iloc[-2] * 100 \
                      if len(past) > 1 else 0

        c_series = pd.Series(past["close"].values)
        h = past["high"].values
        l = past["low"].values
        c = past["close"].values

        # Daily SuperTrend (macro trend gate)
        trend, _ = _supertrend(h, l, c, DAILY_ST_ATR, DAILY_ST_MULT)
        daily_st  = trend[-1]   # 1=BUY, -1=SELL

        # 20 DMA (daily SMA-20) — classic multi-week trend direction
        dma20 = float(c_series.rolling(20).mean().iloc[-1])

        # 50 DMA (daily SMA-50) — medium-term trend
        dma50 = float(c_series.rolling(50).mean().iloc[-1])

        # 5-min EMA-50 (intraday trend, seeded from history)
        ema50_5m = float(c_series.ewm(span=50, adjust=False).mean().iloc[-1])

        # ATM estimate
        step = NIFTY_STRIKE_STEP if symbol == "NIFTY50" else SENSEX_STRIKE_STEP
        atm  = int(round(prev_close / step) * step)

        # Direction signals
        def pos(val, label):
            return f"above {label} ✅" if prev_close > val else f"below {label} ⚠️"

        above_20dma = prev_close > dma20
        above_50dma = prev_close > dma50

        # Overall bias: count bullish signals
        bull_signals = sum([daily_st == 1, above_20dma, above_50dma])
        if bull_signals == 3:
            bias = "🟢 Strong BUY — all 3 indicators bullish"
        elif bull_signals == 2:
            bias = "🟡 Cautious BUY — 2/3 indicators bullish"
        elif bull_signals == 1:
            bias = "🟠 Weak — only 1/3 indicators bullish"
        else:
            bias = "🔴 BEARISH — all indicators bearish, skip OEH today"

        # Expiry type today
        exp_dow, pre_dow = SCHEDULE[symbol]
        dow = today_date.weekday()
        if dow == exp_dow:
            day_type = "⚡ EXPIRY DAY"
        elif pre_dow and dow == pre_dow:
            day_type = "📅 Pre-Expiry"
        else:
            day_type = "🗓 Non-active day"

        prev_emoji = "▲" if prev_change >= 0 else "▼"

        # Strikes to watch
        strikes_ce = [atm + i * step for i in range(-2, STRIKES_EACH_SIDE + 1)]
        strikes_pe = [atm + i * step for i in range(-STRIKES_EACH_SIDE, 3)]
        ce_str = "  ".join(str(s) for s in strikes_ce)
        pe_str = "  ".join(str(s) for s in strikes_pe)

        name = "NIFTY" if symbol == "NIFTY50" else "SENSEX"

        msg = (
            f"📊 *{name} OEH WATCHLIST — {today_date.strftime('%d %b %Y')}*\n"
            f"{day_type}\n\n"
            f"Yesterday: {prev_close:,.0f}  {prev_emoji}{abs(prev_change):.2f}%\n\n"
            f"*Direction indicators (daily chart):*\n"
            f"  SuperTrend: {'🟢 BUY' if daily_st == 1 else '🔴 SELL'}\n"
            f"  20 DMA: {dma20:,.0f}  ({pos(dma20, '20DMA')})\n"
            f"  50 DMA: {dma50:,.0f}  ({pos(dma50, '50DMA')})\n\n"
            f"Overall bias: {bias}\n\n"
            f"📌 Expected ATM: *{atm}*\n"
            f"CE strikes: {ce_str}\n"
            f"PE strikes: {pe_str}\n\n"
            f"🕘 OEH window: 09:15 – 12:30\n"
            f"Entry: spot drops ≥{MIN_SPOT_DROP}% from OEH → 2 green bars\n"
            f"Target: +55% | Stop: -50%"
        )
        return msg
    except Exception as e:
        return f"⚠️ Morning brief error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# OEH signal state machine
# ══════════════════════════════════════════════════════════════════════════════

class OEHStateMachine:
    """Tracks OEH detection and reversal for a single instrument."""

    def __init__(self, symbol: str):
        self.symbol      = symbol
        self.state       = "watch"   # watch → oeh_seen → reversing → signalled
        self.oeh_spot    = None
        self.oeh_time    = None
        self.consec_up   = 0
        self.last_close  = None
        self.wait_bars   = 0
        self.alert_sent  = False
        self.journal_sid = None   # signal ID for journal tracking

    def reset(self):
        self.state       = "watch"
        self.oeh_spot    = None
        self.oeh_time    = None
        self.consec_up   = 0
        self.last_close  = None
        self.wait_bars   = 0
        self.alert_sent  = False
        self.journal_sid = None

    def on_candle(self, bar: dict, prev_bar: dict | None,
                  st_dir: int, ema50: float, ema9: float,
                  t: dtime) -> str | None:
        """
        Feed a completed 5-min candle. Returns an alert type string or None.
        alert types: 'oeh_formed', 'reversal_bar_N', 'entry_confirmed'
        """
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

        if t < OEH_WINDOW_START or t > OEH_WINDOW_END:
            return None

        if self.state == "watch":
            # OEH: open ≈ high AND candle drops ≥ MIN_DROP%
            gap  = abs(o - h) / h * 100 if h > 0 else 99
            drop = (o - c) / o * 100    if o > 0 else 0
            if gap <= OEH_TOL_PCT and drop >= MIN_DROP_PCT:
                # Pre-OEH bar must have ST=BUY and price above EMA-50
                prev_st_buy   = (prev_bar is not None and prev_bar.get("st_dir", -1) == 1)
                above_ema50   = o > ema50
                if prev_st_buy and above_ema50:
                    self.state      = "oeh_seen"
                    self.oeh_spot   = o
                    self.oeh_time   = t.strftime("%H:%M")
                    self.consec_up  = 0
                    self.last_close = c
                    self.wait_bars  = 0
                    return "oeh_formed"

        elif self.state == "oeh_seen":
            self.wait_bars += 1
            if self.wait_bars > 30:
                self.reset()
                return None

            # Check min spot drop from OEH
            drop_from_oeh = (self.oeh_spot - c) / self.oeh_spot * 100 if self.oeh_spot else 0
            if drop_from_oeh < MIN_SPOT_DROP:
                self.last_close = c
                self.consec_up  = 0
                return None

            # Reversal: green bar AND higher close
            is_green   = c > o
            is_higher  = self.last_close is not None and c > self.last_close
            if is_green and is_higher:
                self.consec_up += 1
            else:
                self.consec_up = 0
            self.last_close = c

            if self.consec_up >= REVERSAL_BARS:
                # Final check: close > EMA-9 (momentum returning)
                if c > ema9:
                    self.state = "signalled"
                    return "entry_confirmed"
                else:
                    # Not quite above EMA-9 yet — keep watching
                    return f"reversal_bar_{self.consec_up}"
            elif self.consec_up > 0:
                return f"reversal_bar_{self.consec_up}"

        return None


# ══════════════════════════════════════════════════════════════════════════════
# Option chain snapshot — get live prices for all watched strikes
# ══════════════════════════════════════════════════════════════════════════════

def get_option_chain_snapshot(kite: KiteConnect, option_tokens: dict,
                               oeh_level: float | None = None) -> list[dict]:
    """
    Query live prices for all subscribed option tokens.
    Returns ALL strikes — no premium filter. User picks what they want.
    OEH flag marks strikes where the option itself also had open=high today.
    """
    if not option_tokens:
        return []
    tokens = list(option_tokens.keys())
    try:
        quotes = kite.quote(tokens)
    except Exception as e:
        log.error(f"Option chain quote error: {e}")
        return []

    rows = []
    for token, meta in option_tokens.items():
        q = quotes.get(str(token)) or quotes.get(token)
        if q is None:
            continue
        o   = q["ohlc"]["open"]
        h   = q["ohlc"]["high"]
        ltp = q["last_price"]
        # OEH on option: open ≈ high (within tolerance) AND has since dropped
        gap    = abs(o - h) / h * 100 if h > 0 else 99
        drop   = (o - ltp)  / o * 100 if o > 0 else 0
        is_oeh = gap <= OEH_TOL_PCT and drop >= MIN_DROP_PCT
        rows.append({
            "strike":  meta["strike"],
            "type":    meta["type"],
            "ltp":     max(ltp, 0.05),   # avoid zero display
            "open":    o,
            "high":    h,
            "oeh":     is_oeh,
            "tsymbol": meta["tsymbol"],
        })

    # Sort: CE ascending by strike, PE descending (shows ITM→ATM→OTM naturally)
    ce = sorted([r for r in rows if r["type"] == "CE"], key=lambda x: x["strike"])
    pe = sorted([r for r in rows if r["type"] == "PE"], key=lambda x: -x["strike"])
    return ce + pe


# ══════════════════════════════════════════════════════════════════════════════
# Alert message formatters
# ══════════════════════════════════════════════════════════════════════════════

def _chain_lines(rows: list[dict], opt_type: str) -> str:
    """Format one side of the option chain (CE or PE). OEH strikes marked ✅."""
    filtered = [r for r in rows if r["type"] == opt_type]
    if not filtered:
        return f"  {opt_type}: no data\n"
    lines = []
    for r in filtered:
        oeh_mark = " ✅ OEH" if r["oeh"] else ""
        lines.append(f"  {r['strike']} {opt_type}: ₹{r['ltp']:.0f}{oeh_mark}")
    return "\n".join(lines)


def fmt_oeh_alert(symbol: str, bar: dict, oeh_time: str,
                  opt_rows: list[dict], st_dir: int, ema50_5m: float,
                  dma20: float = 0, dma50: float = 0) -> str:
    """
    STEP 1 — Quick heads-up the moment OEH candle closes.
    Short and immediate. Full option chain follows in the entry alert.
    """
    name     = "NIFTY" if symbol == "NIFTY50" else "SENSEX"
    oeh_lvl  = bar["open"]
    close    = bar["close"]
    drop_pts = oeh_lvl - close
    drop_pct = drop_pts / oeh_lvl * 100

    # Direction confidence — how many indicators are bullish
    bull = sum([
        st_dir == 1,
        oeh_lvl > ema50_5m,
        oeh_lvl > dma20 if dma20 else True,
        oeh_lvl > dma50 if dma50 else True,
    ])
    confidence = {4: "Strong ✅✅", 3: "Good ✅", 2: "Weak ⚠️", 1: "Poor ❌"}.get(bull, "❌")

    return (
        f"👀 *OEH SPOTTED — {name}*\n"
        f"⏰ {oeh_time}\n\n"
        f"Open = High = *{oeh_lvl:,.0f}*\n"
        f"Dropped to: {close:,.0f}  ({drop_pts:.0f}pts ↓{drop_pct:.2f}%)\n\n"
        f"Direction confidence: {confidence}\n\n"
        f"⏳ Watching for reversal...\n"
        f"Will alert on each green bar → entry when confirmed 🔔"
    )


def fmt_reversal_update(symbol: str, bar: dict, oeh_spot: float,
                         bar_num: int, needed: int) -> str:
    """
    STEP 2 — Progress update as each reversal bar forms.
    Keeps the user informed without overwhelming.
    """
    name     = "NIFTY" if symbol == "NIFTY50" else "SENSEX"
    close    = bar["close"]
    drop_pts = oeh_spot - close
    drop_pct = drop_pts / oeh_spot * 100
    remain   = needed - bar_num
    prog     = "🟢" * bar_num + "⬛" * remain

    if remain == 0:
        status = "Entry alert coming next! 🚀"
    elif remain == 1:
        status = "One more green bar and we enter! 🔜"
    else:
        status = f"{remain} more green bars needed"

    return (
        f"{prog} *{name} — Reversal {bar_num}/{needed}*\n"
        f"OEH: {oeh_spot:,.0f} | Now: {close:,.0f}\n"
        f"Still {drop_pts:.0f}pts ({drop_pct:.2f}%) below OEH\n"
        f"{status}"
    )


def fmt_entry_alert(symbol: str, bar: dict, oeh_spot: float,
                     opt_rows: list[dict], step: int,
                     dma20: float = 0, dma50: float = 0) -> str:
    """
    STEP 3 — Entry signal with full option chain.
    Reversal confirmed — shows all strikes, target, stop for each.
    """
    name      = "NIFTY" if symbol == "NIFTY50" else "SENSEX"
    spot_now  = bar["close"]
    drop_pts  = oeh_spot - spot_now
    drop_pct  = drop_pts / oeh_spot * 100
    target_spot = oeh_spot                          # ride back to OEH level
    atm       = int(round(spot_now / step) * step)

    lots = 4 if symbol == "NIFTY50" else 5
    qty  = lots * (65 if symbol == "NIFTY50" else 20)

    def build_chain(opt_type: str) -> str:
        rows = [r for r in opt_rows if r["type"] == opt_type]
        if not rows:
            return "  no data"
        lines = []
        for r in rows:
            ltp     = r["ltp"]
            target  = round(ltp * 1.55)
            stop    = round(ltp * 0.50)
            atm_tag = " ◀ ATM" if r["strike"] == atm else ""
            oeh_tag = " ✅" if r["oeh"] else ""
            lines.append(
                f"  {r['strike']}: ₹{ltp:.0f}  T:₹{target}  S:₹{stop}{oeh_tag}{atm_tag}"
            )
        return "\n".join(lines)

    ce_chain = build_chain("CE")
    pe_chain = build_chain("PE")

    dma_conf = ""
    if dma20:
        dma_conf += f"  20DMA ({'✅' if spot_now > dma20 else '⚠️'})  "
    if dma50:
        dma_conf += f"50DMA ({'✅' if spot_now > dma50 else '⚠️'})"

    return (
        f"🚀 *ENTER NOW — {name} REVERSAL CONFIRMED*\n"
        f"{'🟢' * REVERSAL_BARS} {REVERSAL_BARS} green bars done!\n\n"
        f"OEH level: {oeh_spot:,.0f}\n"
        f"Spot now:  {spot_now:,.0f}  ({drop_pts:.0f}pts below OEH)\n"
        f"Ride back to: ~{target_spot:,.0f}\n\n"
        f"*CE strikes* (buy for upside):*\n{ce_chain}\n\n"
        f"*PE strikes* (buy if OEL reversal):*\n{pe_chain}\n\n"
        f"Qty: {lots} lots ({qty} units)\n"
        f"ST ✅ | EMA ✅ | Drop ✅"
        + (f" | {dma_conf}" if dma_conf else "")
        + f"\n\n⚡ Enter at market — target +55%, stop -50%"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main alerter class
# ══════════════════════════════════════════════════════════════════════════════

class OEHAlerter:
    def __init__(self):
        self.kite          = load_kite_with_token(settings.KITE_API_KEY,
                                                   settings.KITE_ACCESS_TOKEN)
        self.builder       = CandleBuilder(CANDLE_MINS)
        self.journal       = TradeJournal()   # daily trade log

        # One state machine per instrument
        self.state_machines = {
            "NIFTY50": OEHStateMachine("NIFTY50"),
            "SENSEX":  OEHStateMachine("SENSEX"),
        }

        # Spot tokens per instrument
        self.spot_tokens = {
            NIFTY_SPOT_TOKEN:  "NIFTY50",
            SENSEX_SPOT_TOKEN: "SENSEX",
        }

        self.opt_tokens    = {}       # token → meta (populated after open)
        self.indicators    = {}       # symbol → {"ema50", "ema9", "st_dir", "daily_st"}
        self.brief_sent    = False
        self.prev_bars     = {}       # token → last completed bar
        self._lock         = threading.Lock()
        self._active_today = set()   # symbols active today per schedule + daily ST

    # ── Pre-compute daily indicators ──────────────────────────────────────────

    def load_daily_indicators(self):
        """
        Load EMA-50, EMA-9, 5-min SuperTrend, and daily SuperTrend for each instrument.
        Also determine which instruments are active today (schedule + daily ST gate).
        """
        today = datetime.now(IST).date()
        dow   = today.weekday()

        for symbol in ["NIFTY50", "SENSEX"]:
            exp_dow, pre_dow = SCHEDULE[symbol]
            active_dows = {d for d in (exp_dow, pre_dow) if d is not None}

            # Schedule gate — only active on Mon+Tue (Nifty) or Wed+Thu (Sensex)
            if dow not in active_dows:
                log.info(f"{symbol}: not active today (dow={dow})")
                continue

            try:
                df5  = load_historical(symbol, "5min")
                df1d = load_historical(symbol, "1day")
                if df5.index.tz:  df5.index  = df5.index.tz_localize(None)
                if df1d.index.tz: df1d.index = df1d.index.tz_localize(None)

                # 5-min indicators (seeded from history for accurate EMA)
                df5_ind = compute_indicators(df5)
                ema50   = float(df5_ind["ema_trend"].iloc[-1])
                ema9    = float(df5_ind["ema_fast"].iloc[-1])
                st5     = int(df5_ind["st_direction"].iloc[-1])

                # Daily indicators: SuperTrend, 20 DMA, 50 DMA
                past = df1d[df1d.index.date < today]
                if past.empty:
                    daily_st = 1
                    dma20 = dma50 = 0.0
                else:
                    h = past["high"].values
                    l = past["low"].values
                    c = past["close"].values
                    daily_trend, _ = _supertrend(h, l, c, DAILY_ST_ATR, DAILY_ST_MULT)
                    daily_st = int(daily_trend[-1])
                    c_s  = pd.Series(c)
                    dma20 = float(c_s.rolling(20).mean().iloc[-1])
                    dma50 = float(c_s.rolling(50).mean().iloc[-1])

                self.indicators[symbol] = {
                    "ema50":    ema50,
                    "ema9":     ema9,
                    "st5":      st5,
                    "daily_st": daily_st,
                    "dma20":    dma20,
                    "dma50":    dma50,
                }

                if daily_st == 1:
                    self._active_today.add(symbol)
                    log.info(f"{symbol}: ACTIVE today | EMA-50={ema50:.0f} "
                             f"EMA-9={ema9:.0f} 5mST={'BUY' if st5==1 else 'SELL'} "
                             f"DailyST=BUY ✅")
                else:
                    log.info(f"{symbol}: SKIPPED today — Daily SuperTrend is SELL 🔴")

            except Exception as e:
                log.error(f"{symbol}: failed to load indicators — {e}")
                # Default to active with neutral indicators so we don't miss signals
                self.indicators[symbol] = {"ema50": 0, "ema9": 0, "st5": 1, "daily_st": 1}
                self._active_today.add(symbol)

    # ── Subscribe to live option tokens ──────────────────────────────────────

    def refresh_option_tokens(self):
        """Fetch ATM ± N option tokens for all active instruments today."""
        for symbol in self._active_today:
            spot_token = NIFTY_SPOT_TOKEN if symbol == "NIFTY50" else SENSEX_SPOT_TOKEN
            last = self.builder.last_bar(spot_token)
            if last is None:
                continue
            ltp = last["close"]
            log.info(f"Refreshing option tokens for {symbol} ATM≈{ltp:.0f}")
            new_tokens = get_option_tokens(self.kite, symbol, ltp)
            self.opt_tokens.update(new_tokens)
        log.info(f"Total option tokens loaded: {len(self.opt_tokens)}")

    # ── Candle processing ─────────────────────────────────────────────────────

    def _update_ema(self, symbol: str, close: float):
        """Incrementally update rolling EMA-9 and EMA-50 with each new bar."""
        ind = self.indicators.get(symbol)
        if ind is None:
            return
        a9  = 2 / (EMA_FAST  + 1)
        a50 = 2 / (EMA_TREND + 1)
        if ind["ema9"]:  ind["ema9"]  = a9  * close + (1 - a9)  * ind["ema9"]
        if ind["ema50"]: ind["ema50"] = a50 * close + (1 - a50) * ind["ema50"]

    def _process_completed_bar(self, token: int, bar: dict):
        """Called when a 5-min candle completes. Run OEH detection for both instruments."""
        symbol = self.spot_tokens.get(token)
        if symbol is None or symbol not in self._active_today:
            return   # option token or inactive instrument — skip

        now = datetime.now(IST)
        t   = now.time()

        self._update_ema(symbol, bar["close"])
        ind = self.indicators[symbol]

        # Attach 5-min SuperTrend to bar so prev_bar check works
        bar["st_dir"] = ind["st5"]

        prev  = self.prev_bars.get(token)
        sm    = self.state_machines[symbol]
        step  = NIFTY_STRIKE_STEP if symbol == "NIFTY50" else SENSEX_STRIKE_STEP

        alert = sm.on_candle(
            bar      = bar,
            prev_bar = prev,
            st_dir   = ind["st5"],
            ema50    = ind["ema50"],
            ema9     = ind["ema9"],
            t        = t,
        )
        self.prev_bars[token] = bar

        if alert is None:
            return

        log.info(f"{symbol} OEH state → {alert}")

        if alert == "oeh_formed":
            opt_snap = get_option_chain_snapshot(
                self.kite, self.opt_tokens, oeh_level=bar["open"]
            )
            msg = fmt_oeh_alert(
                symbol, bar, sm.oeh_time or now.strftime("%H:%M"),
                opt_snap, ind["st5"], ind["ema50"],
                dma20=ind.get("dma20", 0), dma50=ind.get("dma50", 0),
            )
            send_alert(msg)

        elif alert.startswith("reversal_bar_"):
            n = int(alert.split("_")[-1])
            msg = fmt_reversal_update(
                symbol, bar, sm.oeh_spot or bar["open"], n, REVERSAL_BARS
            )
            send_alert(msg)

        elif alert == "entry_confirmed":
            opt_snap = get_option_chain_snapshot(self.kite, self.opt_tokens)
            msg = fmt_entry_alert(
                symbol, bar, sm.oeh_spot or bar["open"], opt_snap, step,
                dma20=ind.get("dma20", 0), dma50=ind.get("dma50", 0),
            )
            send_alert(msg)

            # ── Journal: log this signal (ATM option, system-default) ──────────
            atm_px   = next(
                (r["ltp"] for r in opt_snap if r.get("atm") and r["type"] == "CE"),
                0.0
            )
            if atm_px > 0:
                atm_strike = int(round(
                    (sm.oeh_spot or bar["open"]) / step
                ) * step)
                target_px = round(atm_px * 1.55, 2)   # +55%
                stop_px   = round(atm_px * 0.50, 2)   # -50%
                from oeh_reversal import FIXED_LOTS_NIFTY, FIXED_LOTS_SENSEX
                lots = FIXED_LOTS_NIFTY if symbol == "NIFTY50" else FIXED_LOTS_SENSEX
                sid = self.journal.log_signal(
                    symbol     = symbol,
                    direction  = "CE",
                    oeh_spot   = sm.oeh_spot or bar["open"],
                    atm_strike = atm_strike,
                    entry_px   = atm_px,
                    target_px  = target_px,
                    stop_px    = stop_px,
                    signal_time= now,
                    lots       = lots,
                )
                sm.journal_sid = sid    # store so we can close it on exit
                log.info(f"Journal: logged signal {sid} — ATM {atm_strike}CE @ {atm_px}")

    # ── WebSocket callbacks ───────────────────────────────────────────────────

    def on_ticks(self, ws, ticks):
        now = datetime.now(IST)
        for tick in ticks:
            token = tick["instrument_token"]
            ltp   = tick.get("last_price") or tick.get("last_traded_price", 0)
            if ltp == 0:
                continue

            completed = self.builder.on_tick(token, ltp, now)
            if completed:
                with self._lock:
                    self._process_completed_bar(token, completed)

    def on_connect(self, ws, response):
        log.info("WebSocket connected")
        # Subscribe to all active spot instruments
        tokens = []
        if "NIFTY50" in self._active_today:
            tokens.append(NIFTY_SPOT_TOKEN)
        if "SENSEX" in self._active_today:
            tokens.append(SENSEX_SPOT_TOKEN)
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_LTP, tokens)
        log.info(f"Subscribed to {len(tokens)} spot token(s): {list(self.spot_tokens[t] for t in tokens if t in self.spot_tokens)}")

    def on_error(self, ws, code, reason):
        log.error(f"WebSocket error {code}: {reason}")

    def on_close(self, ws, code, reason):
        log.warning(f"WebSocket closed {code}: {reason}")

    def on_reconnect(self, ws, attempt):
        log.info(f"WebSocket reconnecting (attempt {attempt})")

    # ── Morning briefing scheduler ────────────────────────────────────────────

    def _briefing_thread(self):
        """Wait until 9:20 AM IST then send morning briefing."""
        today = datetime.now(IST).date()
        brief_dt = datetime.combine(today, MORNING_BRIEF_TIME, tzinfo=IST)
        now_ist  = datetime.now(IST)

        if now_ist < brief_dt:
            wait = (brief_dt - now_ist).total_seconds()
            log.info(f"Briefing scheduled in {wait/60:.0f} min")
            time.sleep(wait)

        log.info("Sending morning briefing…")
        today = datetime.now(IST).date()
        for sym in sorted(self._active_today):
            ind = self.indicators.get(sym, {})
            daily_st_tag = "🟢 BUY" if ind.get("daily_st", 1) == 1 else "🔴 SELL"
            msg = build_morning_brief(sym, today)
            send_alert(msg)
            time.sleep(2)

        self.brief_sent = True
        # Option tokens refresh handled in main loop at 9:22 AM

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self):
        log.info("OEH Alerter starting…")
        self.load_daily_indicators()

        if not self._active_today:
            log.info("No instruments active today — nothing to monitor. Exiting.")
            return

        log.info(f"Active today: {self._active_today}")

        # Start briefing thread
        t = threading.Thread(target=self._briefing_thread, daemon=True)
        t.start()

        # Reset state machines for active instruments
        for sym in self._active_today:
            self.state_machines[sym].reset()

        # Connect WebSocket
        ticker = KiteTicker(settings.KITE_API_KEY, settings.KITE_ACCESS_TOKEN)
        ticker.on_ticks     = self.on_ticks
        ticker.on_connect   = self.on_connect
        ticker.on_error     = self.on_error
        ticker.on_close     = self.on_close
        ticker.on_reconnect = self.on_reconnect

        log.info("Connecting to Kite WebSocket…")
        ticker.connect(threaded=True)

        eod_done = False

        # Main loop — keep alive until market close
        try:
            while True:
                now_ist = datetime.now(IST)
                now     = now_ist.time()
                if now >= MARKET_CLOSE:
                    if not eod_done:
                        self._eod_handler()
                        eod_done = True
                    log.info("Market closed. Exiting.")
                    break
                # Refresh option tokens at 9:22 AM (after spot has ticked a bit)
                if (now >= dtime(9, 22) and now <= dtime(9, 24) and
                        self.brief_sent and not self.opt_tokens):
                    self.refresh_option_tokens()
                time.sleep(10)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
            self._eod_handler()   # send summary even on manual stop
        finally:
            ticker.stop()
            log.info("Stopped.")

    def _eod_handler(self):
        """At market close: close any open journal positions, send daily summary."""
        log.info("EOD: closing open journal positions and sending summary…")

        # Get last known LTP for each instrument's ATM option (approximate)
        ltp_map: dict[str, float] = {}
        for sym, sm in self.state_machines.items():
            if sm.journal_sid and sm.state == "signalled":
                # Fetch last option LTP from the WebSocket cache if available
                spot_token = NIFTY_SPOT_TOKEN if sym == "NIFTY50" else SENSEX_SPOT_TOKEN
                last = self.builder.last_bar(spot_token)
                if last:
                    ltp_map[sym] = last["close"]   # rough proxy; real exit would come from option LTP

        self.journal.close_open_trades(ltp_map, reason="EXPIRED")

        # Build and send daily summary
        summary = self.journal.daily_summary()
        log.info(f"\n{summary}")
        send_alert(summary)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not settings.KITE_API_KEY or not settings.KITE_ACCESS_TOKEN:
        print("ERROR: KITE_API_KEY or KITE_ACCESS_TOKEN missing in .env")
        print("Run  python login.py  first.")
        sys.exit(1)

    # ── Channel status ────────────────────────────────────────────────────────
    print("\n── Alert channels ───────────────────────────────────────────")
    if TG_BOT_TOKEN and TG_CHAT_ID:
        print("  Telegram : ✅ configured")
    else:
        print("  Telegram : ❌ not set  (add TG_BOT_TOKEN + TG_CHAT_ID to .env)")
        print("             Setup: @BotFather → /newbot → copy token")
        print("             Then: https://api.telegram.org/bot<TOKEN>/getUpdates → copy chat id")

    if WA_PHONE and WA_APIKEY and WA_APIKEY not in ("XXXXXXX", ""):
        print("  WhatsApp : ✅ configured")
    else:
        print("  WhatsApp : ⏳ waiting for CallMeBot API key (add WA_APIKEY to .env when received)")

    if not (TG_BOT_TOKEN and TG_CHAT_ID) and not (WA_PHONE and WA_APIKEY):
        print("  ⚠️  No channels configured — alerts will print to this terminal.")
    print("─────────────────────────────────────────────────────────────\n")

    # ── Optional test ─────────────────────────────────────────────────────────
    if "--test" in sys.argv:
        print("Sending test alert…")
        ok = send_alert(
            "✅ *OEH Alerter — test message*\n"
            "If you see this, your alert channel is working!\n"
            "Run  python live_alerter.py  (without --test) on a trading day."
        )
        print("Sent!" if ok else "Printed to console (no channel configured).")
        return

    alerter = OEHAlerter()
    alerter.run()


if __name__ == "__main__":
    main()
