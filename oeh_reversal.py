#!/usr/bin/env python3
"""
oeh_reversal.py — Open=High Reversal Entry Strategy

Exact flow (as described):
  1. OEH candle: spot opens at the high, sells off (drops ≥ MIN_DROP%)
  2. Market continues to fall — we watch, do NOT enter yet
  3. Reversal confirmation: 2 consecutive bars close HIGHER than prior bar
     AND current close > OEH close (mini breakout of the pullback high)
  4. Entry: buy ATM call at the reversal bar's close price
     → By now spot is ~100-200 pts below OEH level → option is cheap (~₹50)
  5. Target: +TARGET_PCT% gain on premium (e.g. ₹50 → ₹77 at 55% gain)
  6. Stop:   -STOP_PCT% loss on premium  (e.g. ₹50 → ₹25 at 50% loss)
  7. Force-close by 3:00 PM

Sizing:
  User targets ~325 qty (≈ 4 Nifty lots × 75 = 300, or ~16 Sensex lots × 20 = 320)
  We use FIXED_LOTS for consistent comparison.

Valid days:
  Mon + Tue  → Nifty  (Tue = expiry, Mon = pre-expiry)
  Wed + Thu  → Sensex (Thu = expiry, Wed = pre-expiry)

Run:
    python oeh_reversal.py
"""

from datetime import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kite_data import load_historical
from options_utils import (
    INSTRUMENTS, RISK_FREE_RATE,
    bs_price, atm_strike,
    intraday_time_to_expiry, overnight_time_to_expiry,
)
from real_options import RealOptionsLoader

# ── Config ────────────────────────────────────────────────────────────────────

OEH_TOLERANCE   = 0.05    # % — open must be within this of high
MIN_DROP        = 0.15    # % — bar must drop from open to qualify as OEH
MIN_SPOT_DROP   = 0.40    # % — spot must pull back ≥ this from OEH before reversal entry
                           #     ≈96pt on Nifty@24k — forces a meaningful pullback
REVERSAL_BARS   = 2       # consecutive higher-close AND green bars to confirm reversal
MAX_WAIT_BARS   = 30      # give up if no reversal within this many bars after OEH
TARGET_PCT      = 55.0    # % gain on premium to exit (₹50 → ₹77.5)
STOP_PCT        = 50.0    # % loss on premium to exit  (₹50 → ₹25)
# Instrument-specific premium range at entry
# Nifty ATM ~24k level → ₹25–100 is the sweet spot
# Sensex ATM ~80k level → ₹100–200 is the right range
OPTION_PX_RANGE = {
    "NIFTY50": (25,  100),
    "SENSEX":  (100, 200),
}

# ── SuperTrend + EMA config ───────────────────────────────────────────────────
ST_ATR_PERIOD   = 10      # ATR period for 5-min SuperTrend
ST_MULTIPLIER   = 3.0     # SuperTrend multiplier (standard: 3)
EMA_FAST        = 9       # Fast EMA (for reversal bar confirmation at entry)
EMA_TREND       = 50      # Slower EMA confirming BROADER uptrend at OEH time

# DAILY SuperTrend — macro trend filter
# Only take OEH setups when the DAILY chart SuperTrend is in BUY mode
# This kills false reversals in bearish macro environments (e.g. 2023-2025 chop)
DAILY_ST_ATR    = 10
DAILY_ST_MULT   = 3.0
REQUIRE_DAILY_ST  = True   # Set False to backtest without daily trend gate
REQUIRE_DAILY_DMA = True   # Require price > 20 DMA on daily chart (trend direction)

# 5-min confirmations
REQUIRE_ST_CONF   = True   # prev bar's 5-min ST must be BUY at OEH bar
REQUIRE_EMA_CONF  = True   # OEH bar open > EMA-50 (broad uptrend)
REQUIRE_ENTRY_EMA = False  # EMA-9 at entry already implicit via MIN_SPOT_DROP + 2-bar reversal

EARLIEST_ENTRY  = time(9, 20)
FIRST_HALF_END  = time(12, 30)
EXPIRY_EXT_END  = time(14, 30)   # expiry days only
FORCE_CLOSE     = time(15, 0)

# Capital-based sizing — all values loaded from .env
# Set in .env:  INITIAL_CAPITAL=200000  DEPLOY_PCT=0.20  MAX_LOTS_NIFTY=10  MAX_LOTS_SENSEX=20
from settings import settings as _settings
INITIAL_CAPITAL   = _settings.INITIAL_CAPITAL   # e.g. 200000
DEPLOY_PCT        = _settings.DEPLOY_PCT         # e.g. 0.20 = 20% per trade
MAX_LOTS_NIFTY    = _settings.MAX_LOTS_NIFTY     # hard cap to avoid runaway sizing
MAX_LOTS_SENSEX   = _settings.MAX_LOTS_SENSEX

HV_WINDOW = 20

# Active weekdays: (expiry_dow, pre_expiry_dow)
# Set pre_expiry_dow to None to trade expiry days only
SCHEDULE = {
    "NIFTY50": (1, None),   # Tue expiry only
    "SENSEX":  (3, None),   # Thu expiry only
}


# ── Indicators ───────────────────────────────────────────────────────────────

def _hv(df1d: pd.DataFrame, as_of: pd.Timestamp) -> float:
    past = df1d[df1d.index < as_of]["close"].tail(HV_WINDOW + 1)
    if len(past) < 5:
        return 0.15
    lr = np.log(past / past.shift(1)).dropna()
    return max(0.08, float(lr.std() * np.sqrt(252)))


def _supertrend(h, l, c, atr_period, multiplier):
    """
    Core SuperTrend logic (Pine Script exact port).
    Returns (direction_array, line_array) where direction: 1=BUY, -1=SELL.
    """
    hl2 = (h + l) / 2
    prev_c = pd.Series(c).shift(1).values
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    # Wilder's smoothing
    atr = np.zeros(len(c))
    atr[0] = tr[0]
    alpha = 1 / atr_period
    for i in range(1, len(c)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    ub = hl2 + multiplier * atr
    lb = hl2 - multiplier * atr

    up   = np.zeros(len(c))
    dn   = np.zeros(len(c))
    trend = np.ones(len(c), dtype=int)
    up[0] = lb[0]
    dn[0] = ub[0]

    for i in range(1, len(c)):
        up[i]    = max(lb[i], up[i-1]) if c[i-1] > up[i-1] else lb[i]
        dn[i]    = min(ub[i], dn[i-1]) if c[i-1] < dn[i-1] else ub[i]
        if   trend[i-1] == -1 and c[i] > dn[i-1]:  trend[i] =  1
        elif trend[i-1] ==  1 and c[i] < up[i-1]:  trend[i] = -1
        else:                                         trend[i] = trend[i-1]

    return trend, np.where(trend == 1, up, dn)


def compute_daily_st(df1d: pd.DataFrame) -> pd.Series:
    """
    Compute SuperTrend on the daily chart.
    Returns a Series (indexed by date) with 1=BUY, -1=SELL.
    """
    h = df1d["high"].values
    l = df1d["low"].values
    c = df1d["close"].values
    trend, _ = _supertrend(h, l, c, DAILY_ST_ATR, DAILY_ST_MULT)
    return pd.Series(trend, index=df1d.index.date, name="daily_st")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SuperTrend and EMA on a 5-min DataFrame.
    Returns df with added columns: ema_fast, ema_trend, st_direction, st_line.
    """
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]

    # EMA
    df["ema_fast"]  = c.ewm(span=EMA_FAST,  adjust=False).mean()
    df["ema_trend"] = c.ewm(span=EMA_TREND, adjust=False).mean()

    # ATR (Wilder's smoothing)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/ST_ATR_PERIOD, adjust=False).mean()

    hl2 = (h + l) / 2
    upper_band = hl2 + ST_MULTIPLIER * atr
    lower_band = hl2 - ST_MULTIPLIER * atr

    # 5-min SuperTrend (Pine Script exact port via shared helper)
    trend5, line5 = _supertrend(
        df["high"].values, df["low"].values, c.values,
        ST_ATR_PERIOD, ST_MULTIPLIER
    )
    df["st_direction"] = trend5
    df["st_line"]      = line5
    return df


def is_oeh(bar: pd.Series) -> bool:
    """Open is within OEH_TOLERANCE% of the high AND bar dropped ≥ MIN_DROP%."""
    if bar["high"] == 0:
        return False
    gap   = abs(bar["open"] - bar["high"]) / bar["high"] * 100
    drop  = (bar["open"] - bar["close"]) / bar["open"] * 100
    return gap <= OEH_TOLERANCE and drop >= MIN_DROP


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:          str
    instrument:    str
    is_expiry:     bool
    oeh_time:      str
    entry_time:    str
    exit_time:     str
    oeh_spot:      float
    entry_spot:    float
    spot_drop_pts: float    # pts spot fell from OEH → entry (larger = cheaper option)
    strike:        int
    entry_prem:    float
    exit_prem:     float
    exit_reason:   str
    pnl_per_unit:  float
    lots:          int
    qty:           int
    pnl_inr:       float
    deploy_amt:    float    # ₹ deployed on this trade (lots × lot_size × entry_px)
    st_signal:     str      # SuperTrend at entry: BUY / SELL
    price_source:  str      # 'real' or 'bs' (Black-Scholes fallback)
    equity:        float


# ── Backtest engine ───────────────────────────────────────────────────────────

def _option_price(loader: "RealOptionsLoader | None", symbol: str,
                  ts: pd.Timestamp, strike: int, expiry_date,
                  spot: float, hv: float, is_expiry: bool,
                  t_str: str) -> tuple[float, str]:
    """
    Return (premium, source) where source is 'real' or 'bs'.
    Tries real option data first; falls back to Black-Scholes.
    """
    if loader is not None:
        real = loader.get_price(symbol, ts, strike, expiry_date)
        if real is not None and real > 0:
            return real, "real"

    # Black-Scholes fallback
    T = max(intraday_time_to_expiry(t_str) if is_expiry
            else overnight_time_to_expiry(1.0), 1e-6)
    prem = bs_price(spot, strike, T, hv, RISK_FREE_RATE, "call")
    return prem, "bs"


def _calc_lots(equity: float, entry_px: float, lot_size: int, max_lots: int) -> int:
    """How many lots can we buy with DEPLOY_PCT of current equity?"""
    deploy   = equity * DEPLOY_PCT
    cost_per = entry_px * lot_size
    if cost_per <= 0:
        return 1
    lots = int(deploy // cost_per)
    return max(1, min(lots, max_lots))


def run(symbol: str, df5: pd.DataFrame, df1d: pd.DataFrame,
        initial_capital: float = INITIAL_CAPITAL,
        opt_loader: "RealOptionsLoader | None" = None) -> list[Trade]:

    cfg       = INSTRUMENTS[symbol]
    lot       = cfg["lot"]
    step      = cfg["strike_step"]
    exp_dow, pre_dow = SCHEDULE[symbol]
    active_dows = {d for d in (exp_dow, pre_dow) if d is not None}
    max_lots   = MAX_LOTS_NIFTY if symbol == "NIFTY50" else MAX_LOTS_SENSEX

    trades: list[Trade] = []
    equity = initial_capital
    all_dates = sorted(set(df5.index.date))
    min_px, max_px = OPTION_PX_RANGE[symbol]

    # Pre-compute indicators on the full dataset (cross-day, so EMAs warm up correctly)
    df5_ind = compute_indicators(df5)

    # Daily SuperTrend — macro trend filter
    daily_st = compute_daily_st(df1d) if REQUIRE_DAILY_ST else None

    # Daily 20 DMA — direction filter (price must be above 20-day SMA to take OEH CE trades)
    daily_close = df1d["close"].copy()
    if df1d.index.tz:
        daily_close.index = daily_close.index.tz_localize(None)
    daily_dma20 = daily_close.rolling(20).mean() if REQUIRE_DAILY_DMA else None

    for trade_date in all_dates:
        dow = trade_date.weekday()
        if dow not in active_dows:
            continue

        is_expiry    = (dow == exp_dow)
        latest_entry = EXPIRY_EXT_END if is_expiry else FIRST_HALF_END
        hv           = _hv(df1d, pd.Timestamp(trade_date))

        # Daily SuperTrend gate — skip days when daily trend is SELL
        if daily_st is not None:
            past_st = daily_st[daily_st.index < trade_date]
            if len(past_st) == 0 or past_st.iloc[-1] != 1:
                continue   # daily ST is SELL → skip this day entirely

        # Daily 20 DMA gate — skip days when price is below 20-day SMA
        if daily_dma20 is not None:
            past_dma = daily_dma20[daily_dma20.index < pd.Timestamp(trade_date)]
            past_close = daily_close[daily_close.index < pd.Timestamp(trade_date)]
            if len(past_dma) > 0 and len(past_close) > 0:
                if past_close.iloc[-1] < past_dma.iloc[-1]:
                    continue   # price below 20 DMA → skip

        day_bars = df5_ind[df5_ind.index.date == trade_date]
        bars     = list(day_bars.iterrows())

        state         = "watch"
        oeh_spot      = None
        oeh_time      = None
        wait_count    = 0
        consec_up     = 0
        last_close    = None
        open_trade    = None
        prev_bar      = None   # bar just before current — used for pre-OEH ST check

        for i, (idx, bar) in enumerate(bars):
            t     = idx.time()
            t_str = idx.strftime("%H:%M")

            # ── Manage open trade ─────────────────────────────────────────────
            if state == "in_trade" and open_trade is not None:
                S = bar["close"]
                # Try real option price at exit bar; fall back to BS
                opt_now, exit_src = _option_price(
                    opt_loader, symbol, idx, open_trade["K"], trade_date,
                    S, hv, is_expiry, t_str
                )

                gain_pct = (opt_now - open_trade["entry"]) / open_trade["entry"] * 100
                reason = None
                if t >= FORCE_CLOSE:
                    reason = "EOD"
                elif gain_pct >= TARGET_PCT:
                    reason = "TARGET"
                elif gain_pct <= -STOP_PCT:
                    reason = "STOP"

                if reason:
                    entry_lots = open_trade["lots"]
                    pnl_u      = opt_now - open_trade["entry"]
                    qty        = entry_lots * lot
                    pnl_inr    = pnl_u * qty
                    deploy_amt = entry_lots * lot * open_trade["entry"]
                    equity    += pnl_inr
                    trades.append(Trade(
                        date          = str(trade_date),
                        instrument    = symbol,
                        is_expiry     = is_expiry,
                        oeh_time      = oeh_time,
                        entry_time    = open_trade["entry_time"],
                        exit_time     = t_str,
                        oeh_spot      = oeh_spot,
                        entry_spot    = open_trade["entry_spot"],
                        spot_drop_pts = round(oeh_spot - open_trade["entry_spot"], 1),
                        strike        = open_trade["K"],
                        entry_prem    = round(open_trade["entry"], 2),
                        exit_prem     = round(opt_now, 2),
                        exit_reason   = reason,
                        pnl_per_unit  = round(pnl_u, 2),
                        lots          = entry_lots,
                        qty           = qty,
                        pnl_inr       = round(pnl_inr, 2),
                        deploy_amt    = round(deploy_amt, 0),
                        st_signal     = open_trade.get("st_at_entry", "?"),
                        price_source  = open_trade.get("price_source", "bs"),
                        equity        = round(equity, 0),
                    ))
                    state = "done"
                continue

            if state == "done":
                continue

            # ── Gate: only look during valid entry window ──────────────────────
            if t < EARLIEST_ENTRY or t > latest_entry:
                continue

            # ── State: watch — look for OEH candle ────────────────────────────
            if state == "watch":
                if is_oeh(bar) and prev_bar is not None:
                    # Check SuperTrend on the BAR BEFORE the OEH — because the OEH
                    # bar itself is bearish (open=high, sells off) so ST will flip
                    # to SELL on that bar. We want to know if the trend was BUY
                    # just before the pullback started.
                    st_buy_before_oeh = prev_bar["st_direction"] == 1
                    # EMA-50 check at OEH bar's OPEN (= the OEH high level)
                    # — OEH open must be above the trend EMA (confirms uptrend context)
                    above_trend_ema   = bar["open"] > bar["ema_trend"]

                    if REQUIRE_ST_CONF and not st_buy_before_oeh:
                        prev_bar = bar
                        continue   # trend was already bearish before OEH → skip
                    if REQUIRE_EMA_CONF and not above_trend_ema:
                        prev_bar = bar
                        continue   # OEH level is below EMA-50 → no uptrend context

                    state      = "oeh_seen"
                    oeh_spot   = bar["open"]
                    oeh_time   = t_str
                    wait_count = 0
                    consec_up  = 0
                    last_close = bar["close"]
                prev_bar = bar
                continue

            # ── State: oeh_seen — wait for genuine reversal from the low ────────
            if state == "oeh_seen":
                wait_count += 1
                if wait_count > MAX_WAIT_BARS:
                    state = "watch"
                    continue

                c = bar["close"]

                # Require a meaningful pullback from OEH before reversal entry
                if MIN_SPOT_DROP > 0:
                    spot_drop_pct = (oeh_spot - c) / oeh_spot * 100
                    if spot_drop_pct < MIN_SPOT_DROP:
                        last_close = c
                        consec_up  = 0
                        continue

                # Count consecutive STRONG reversal bars:
                # bar must be green (close > open) AND higher close than last bar
                is_green      = bar["close"] > bar["open"]
                is_higher_cls = (last_close is not None and c > last_close)

                if is_green and is_higher_cls:
                    consec_up += 1
                else:
                    consec_up = 0
                last_close = c

                if consec_up < REVERSAL_BARS:
                    continue

                # ── Fast EMA confirmation at entry bar ────────────────────────
                # Price recovering above EMA-9 shows momentum is returning.
                # (We already checked SuperTrend + EMA-50 at the OEH bar.)
                if REQUIRE_ENTRY_EMA and c <= bar["ema_fast"]:
                    consec_up = 0   # not back above fast EMA yet, wait
                    continue

                # ── Entry ─────────────────────────────────────────────────────
                K = atm_strike(c, step)
                prem, psrc = _option_price(
                    opt_loader, symbol, idx, K, trade_date,
                    c, hv, is_expiry, t_str
                )

                if prem < min_px or prem > max_px:
                    state = "watch"
                    continue

                entry_lots = _calc_lots(equity, prem, lot, max_lots)
                state      = "in_trade"
                open_trade = {
                    "K":           K,
                    "entry":       prem,
                    "entry_time":  t_str,
                    "entry_spot":  round(c, 1),
                    "price_source": psrc,
                    "st_at_entry": "BUY",
                    "lots":        entry_lots,
                }

    return trades


# ── Display ───────────────────────────────────────────────────────────────────

def show(trades: list[Trade], symbol: str, cap: float) -> dict:
    if not trades:
        print(f"\n  {symbol}: NO TRADES")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["won"]  = df["pnl_inr"] > 0

    total  = len(df)
    won    = df["won"].sum()
    wr     = won / total * 100
    pnl    = df["pnl_inr"].sum()
    avg_w  = df.loc[df["won"],  "pnl_inr"].mean() if won > 0 else 0
    avg_l  = df.loc[~df["won"], "pnl_inr"].mean() if (total - won) > 0 else 0
    exp    = (wr/100)*avg_w + (1-wr/100)*avg_l
    yrs    = max((df["date"].max()-df["date"].min()).days/365.25, 0.1)
    cagr   = ((df["equity"].iloc[-1] / cap) ** (1/yrs) - 1) * 100

    eq_arr = df["equity"].values
    peak = cap; max_dd = 0
    for e in eq_arr:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    print(f"\n{'='*65}")
    print(f"  {symbol} | OEH Reversal Entry | ATM Call | 5-min")
    print(f"{'='*65}")
    print(f"  ₹{cap:,.0f} → ₹{df['equity'].iloc[-1]:,.0f}  ({(df['equity'].iloc[-1]-cap)/cap*100:+.1f}%)")
    print(f"  CAGR: {cagr:+.1f}%   MaxDD: {max_dd:.1f}%")
    print(f"  Trades: {total}  ({total/yrs:.0f}/yr)   Win: {wr:.0f}%  ({int(won)}W/{total-int(won)}L)")
    print(f"  Avg win: ₹{avg_w:,.0f}   Avg loss: ₹{avg_l:,.0f}")
    print(f"  Expectancy: ₹{exp:+,.0f}/trade")
    print(f"  Avg entry premium: ₹{df['entry_prem'].mean():.1f}  "
          f"(range ₹{df['entry_prem'].min():.0f}–₹{df['entry_prem'].max():.0f})")
    print(f"  Avg spot drop OEH→entry: {df['spot_drop_pts'].mean():.0f} pts")

    yr_pnl  = df.groupby("year")["pnl_inr"].sum()
    max_abs = max(abs(yr_pnl).max(), 1)
    print(f"\n  {'Year':<6} {'T':>4} {'W%':>5}  {'P&L ₹':>12}  Bar")
    for yr, grp in df.groupby("year"):
        n   = len(grp)
        w   = grp["won"].sum()
        p   = grp["pnl_inr"].sum()
        bw  = int(abs(p) / max_abs * 18)
        ch  = "█" if p >= 0 else "░"
        sg  = "+" if p >= 0 else ""
        print(f"  {yr:<6} {n:>4} {w/n*100:>4.0f}%  {sg}{p:>11,.0f}  {ch*bw}")

    print(f"\n  Exit breakdown:")
    for r, g in df.groupby("exit_reason"):
        n   = len(g)
        p   = g["pnl_inr"].sum()
        wr_ = g["won"].mean() * 100
        print(f"    {r:<8} {n:>4}  win={wr_:3.0f}%  P&L=₹{p:+,.0f}")

    print(f"\n  Expiry day vs Pre-expiry:")
    for lbl, mask in [("Expiry", df["is_expiry"]), ("Pre-exp", ~df["is_expiry"])]:
        g = df[mask]
        if not g.empty:
            print(f"    {lbl:<8} {len(g):>4} trades  win={g['won'].mean()*100:.0f}%  "
                  f"P&L=₹{g['pnl_inr'].sum():+,.0f}  "
                  f"avg entry ₹{g['entry_prem'].mean():.0f}")

      # Per-trade log (last 20 as sample)
    real_trades = (df["price_source"] == "real").sum() if "price_source" in df.columns else 0
    bs_trades   = total - real_trades
    src_note    = (f"  Pricing: {real_trades} real option prices / {bs_trades} Black-Scholes"
                   if real_trades > 0 else "  Pricing: Black-Scholes only (run download_options.py for real data)")
    print(src_note)

    print(f"\n  Recent trades (last 20):")
    print(f"  {'Date':<12} {'Exp':>4} {'Src':>4} {'Drop':>5} "
          f"{'Prem':>5} {'Exit':>5} {'Lots':>4} {'Deployed':>9} {'Gain%':>6} {'P&L':>9} {'Rsn':>7}")
    for _, r in df.tail(20).iterrows():
        gain_pct = (r["exit_prem"] - r["entry_prem"]) / r["entry_prem"] * 100
        exp_tag  = "EXP" if r["is_expiry"] else "pre"
        src_tag  = r.get("price_source", "bs")[:4]
        lots_val = int(r.get("lots", r["qty"] // 65))
        deployed = r.get("deploy_amt", r["entry_prem"] * r["qty"])
        print(f"  {r['date'].strftime('%d %b %y'):<12} {exp_tag:>4} "
              f"{src_tag:>4} "
              f"{r['spot_drop_pts']:>5.0f} "
              f"{r['entry_prem']:>5.1f} {r['exit_prem']:>5.1f} "
              f"{lots_val:>4}  ₹{deployed:>7,.0f} "
              f"{gain_pct:>+6.0f}% {r['pnl_inr']:>+9,.0f}  {r['exit_reason']:>6}")

    return {"cagr": cagr, "max_dd": max_dd, "win_rate": wr,
            "exp": exp, "trades": total, "trades_yr": total/yrs,
            "avg_prem": df["entry_prem"].mean()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading spot data…")
    n5  = load_historical("NIFTY50", "5min")
    n1d = load_historical("NIFTY50", "1day")
    s5  = load_historical("SENSEX",  "5min")
    s1d = load_historical("SENSEX",  "1day")

    for df in [n5, n1d, s5, s1d]:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    print(f"  Nifty  5min: {len(n5):,} bars  ({n5.index[0].date()} → {n5.index[-1].date()})")
    print(f"  Sensex 5min: {len(s5):,} bars")

    # Load real option prices if available
    loader = RealOptionsLoader()
    cov    = loader.coverage_report()
    if cov.empty:
        print("\n  [!] No real option data found — using Black-Scholes pricing.")
        print("      Run  python download_options.py  to download real option prices.")
        loader = None
    else:
        n_nifty  = len(cov[cov["symbol"] == "NIFTY50"])
        n_sensex = len(cov[cov["symbol"] == "SENSEX"])
        total    = cov["strikes"].sum()
        print(f"\n  Real option data: {n_nifty} Nifty expiries, {n_sensex} Sensex expiries "
              f"({total} strike files)")

    CAP = INITIAL_CAPITAL
    results = []

    t1 = run("NIFTY50", n5, n1d, CAP, opt_loader=loader)
    r1 = show(t1, "NIFTY50", CAP)
    if r1: results.append(("NIFTY50", r1))

    t2 = run("SENSEX", s5, s1d, CAP, opt_loader=loader)
    r2 = show(t2, "SENSEX", CAP)
    if r2: results.append(("SENSEX", r2))

    print(f"\n\n{'='*65}")
    print("  COMBINED SUMMARY")
    print(f"{'='*65}")
    for sym, r in results:
        sign = "+" if r["cagr"] >= 0 else ""
        print(f"  {sym:<10} {r['trades_yr']:.0f} trades/yr  "
              f"Win {r['win_rate']:.0f}%  "
              f"CAGR {sign}{r['cagr']:.1f}%  DD {r['max_dd']:.1f}%  "
              f"Exp ₹{r['exp']:+,.0f}  "
              f"Avg entry ₹{r['avg_prem']:.0f}")

    print(f"\n  Strategy rules:")
    print(f"  1. OEH candle: open≈high on 5-min bar, spot drops ≥{MIN_DROP}%")
    print(f"  2. Pullback: spot must drop ≥{MIN_SPOT_DROP}% from OEH (≈{MIN_SPOT_DROP/100*24000:.0f}pt on Nifty@24k)")
    print(f"  3. Reversal: {REVERSAL_BARS} consecutive GREEN + higher-close bars")
    print(f"  4. At OEH bar: {'SuperTrend=BUY' if REQUIRE_ST_CONF else 'any ST'} + {'close>EMA-'+str(EMA_TREND) if REQUIRE_EMA_CONF else 'any EMA'} (broad uptrend)")
    print(f"  4b.At entry bar: {'close>EMA-'+str(EMA_FAST) if REQUIRE_ENTRY_EMA else 'no EMA check'} (momentum returning)")
    print(f"  0. Daily ST gate:  {'ON — skip bearish daily trend days' if REQUIRE_DAILY_ST else 'OFF'}")
    print(f"  0. Daily 20 DMA:   {'ON — skip days when price < 20-day SMA' if REQUIRE_DAILY_DMA else 'OFF'}")
    n_lo, n_hi = OPTION_PX_RANGE["NIFTY50"]
    s_lo, s_hi = OPTION_PX_RANGE["SENSEX"]
    print(f"  5. Premium filter: Nifty ₹{n_lo}–₹{n_hi}  |  Sensex ₹{s_lo}–₹{s_hi}")
    print(f"     Min spot drop from OEH: {MIN_SPOT_DROP}%")
    print(f"  6. Target: +{TARGET_PCT:.0f}% on premium  |  Stop: -{STOP_PCT:.0f}% on premium")
    print(f"  7. Sizing: {DEPLOY_PCT*100:.0f}% of equity per trade  "
          f"(max {MAX_LOTS_NIFTY} lots Nifty / {MAX_LOTS_SENSEX} lots Sensex)")
    print(f"     Capital: ₹{CAP:,.0f}")
    print(f"  8. Active: Mon+Tue (Nifty), Wed+Thu (Sensex). Force-close 15:00.")


if __name__ == "__main__":
    main()
