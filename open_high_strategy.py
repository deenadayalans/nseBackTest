#!/usr/bin/env python3
"""
open_high_strategy.py — "Open = High / Open = Low" ATM Options Strategy

Setup (CALL buy — bullish):
  1. Spot 3-min or 5-min bar: open ≈ high  (bar opened at top, sold off)
  2. ATM call price at open > call price now  (implied — call also "open=high")
  3. Current call price > 50% of call's open price  (not decayed too far)
  4. Direction filter: uptrend OR price reversing upward into this level
  → Buy ATM CALL. Target: call returns above its open level.
    Stop: call falls below 50% of its opening price.

Setup (PUT buy — bearish, mirror image):
  1. Spot bar: open ≈ low  (bar opened at bottom, rallied up)
  2. ATM put price at open > put price now  (put also "open=high" since spot low→high)
  3. Current put price > 50% of put's open price
  4. Direction filter: downtrend OR reversing downward
  → Buy ATM PUT. Same stop/target logic.

Why it works:
  When both spot and option show "open=high" (opened at their candle's extreme),
  institutional supply/demand has paused. If price hasn't fallen more than 50%,
  the level acts as a magnet — price typically revisits and breaks through the
  open=high level before exhausting.

Timeframes: 3-min and 5-min (both tested here).
Instruments: Nifty 50 and Sensex (ATM only, no OTMs).

Run:
    python open_high_strategy.py
"""

import sys
from datetime import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kite_data import load_historical
from options_utils import (
    INSTRUMENTS, RISK_FREE_RATE,
    bs_price, atm_strike, hist_vol,
    intraday_time_to_expiry, overnight_time_to_expiry,
)

# ── Config ────────────────────────────────────────────────────────────────────

# Candle setup
OEH_TOLERANCE   = 0.05    # % — open must be within this of high to count as "open=high"
MIN_CANDLE_DROP = 0.15    # % — open-to-close drop must be at least this (real pullback)
CMP_FLOOR_PCT   = 0.50    # call/put must be > 50% of its opening price (user's filter)

# Prior trend confirmation: of the N bars before the OEH signal,
# at least TREND_AGREE must be up-bars (close > open) for a CALL setup
PRIOR_BARS      = 4       # look-back window
TREND_AGREE     = 3       # minimum up-bars required out of PRIOR_BARS

# Entry window (from charts observation)
#   First half (9:15–13:00) → works on ALL trading days
#   Second half (13:00–14:30) → works on EXPIRY days only (Tue/Thu)
#   Pre-expiry afternoons → market too quiet, skip
EARLIEST_ENTRY       = time(9, 20)
FIRST_HALF_END       = time(13, 0)    # first half ends at 1 PM for all days
EXPIRY_LATEST_ENTRY  = time(14, 30)   # on expiry days, signals up to 2:30 PM
FORCE_CLOSE          = time(15, 0)    # close all by 3 PM

# Expiry day-of-week map (0=Mon … 6=Sun)
EXPIRY_DOW = {
    "NIFTY50": 1,   # Tuesday
    "SENSEX":  3,   # Thursday
}

# Risk / reward
STOP_PRICE_PCT  = 0.50    # stop: option falls to 50% of signal-bar's open price
RR              = 2.0     # default 1:2 RR (also test 1:3)
RISK_PCT        = 1.5     # % of equity to risk per trade
MAX_LOTS        = 4       # hard cap — prevents runaway sizing on late-day entries

# Direction filter EMAs
FAST_EMA        = 9
SLOW_EMA        = 21

# HV window for B-S pricing
HV_WINDOW       = 20

# Consecutive bars confirming reversal
REVERSAL_BARS   = 3       # last N bars must be in the trade direction to confirm reversal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _hv_daily(df1d: pd.DataFrame, as_of: pd.Timestamp) -> float:
    past = df1d[df1d.index < as_of]["close"].tail(HV_WINDOW + 1)
    if len(past) < 5:
        return 0.15
    lr = np.log(past / past.shift(1)).dropna()
    return max(0.08, float(lr.std() * np.sqrt(252)))


def is_open_eq_high(bar: pd.Series, tol_pct: float = OEH_TOLERANCE) -> bool:
    """True when bar's open is within tol_pct% of its high."""
    if bar["high"] == 0:
        return False
    return abs(bar["open"] - bar["high"]) / bar["high"] * 100 <= tol_pct


def is_open_eq_low(bar: pd.Series, tol_pct: float = OEH_TOLERANCE) -> bool:
    """True when bar's open is within tol_pct% of its low."""
    if bar["low"] == 0:
        return False
    return abs(bar["open"] - bar["low"]) / bar["low"] * 100 <= tol_pct


def direction_filter(hist_closes: pd.Series, day_closes: pd.Series) -> str:
    """
    Returns 'up', 'down', or 'neutral'.

    Uses cross-day historical closes for EMAs so the signal can fire
    at any time of day (not just after 115 min of warmup within the day).
    Then confirms with recent intraday bars for reversal detection.
    """
    if len(hist_closes) < SLOW_EMA + 5:
        return "neutral"

    fast = _ema(hist_closes, FAST_EMA).iloc[-1]
    slow = _ema(hist_closes, SLOW_EMA).iloc[-1]

    trending_up   = fast > slow
    trending_down = fast < slow

    # Reversal check: last few intraday bars moving against the EMA trend
    if len(day_closes) >= REVERSAL_BARS:
        recent = day_closes.iloc[-REVERSAL_BARS:]
        reversing_up   = not trending_up   and (recent.iloc[-1] > recent.iloc[0])
        reversing_down = not trending_down and (recent.iloc[-1] < recent.iloc[0])
    else:
        reversing_up = reversing_down = False

    if trending_up or reversing_up:
        return "up"
    if trending_down or reversing_down:
        return "down"
    return "neutral"


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:         str
    instrument:   str
    timeframe:    str
    is_expiry:    bool
    direction:    str       # call / put
    signal_time:  str
    entry_time:   str
    spot_at_sig:  float
    strike:       int
    call_open:    float     # option price at signal bar's open (the "open=high" level)
    entry_prem:   float     # option price when we enter (< call_open for calls)
    stop_prem:    float     # 50% of call_open
    target_prem:  float     # entry + 2× risk (or 3×)
    exit_prem:    float
    exit_reason:  str       # TARGET / STOP / EOD
    pnl_pts:      float
    pnl_inr:      float
    lots:         int
    lot_size:     int
    equity:       float


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_oeh(
    symbol: str,
    df_intraday: pd.DataFrame,    # 3-min or 5-min OHLCV
    df_daily: pd.DataFrame,
    timeframe: str,
    initial_capital: float = 500_000,
    rr: float = RR,
) -> tuple[list[Trade], float]:

    cfg      = INSTRUMENTS[symbol]
    lot      = cfg["lot"]
    step     = cfg["strike_step"]

    equity = initial_capital
    trades: list[Trade] = []

    all_dates = sorted(set(df_intraday.index.date))

    # Pre-compute cross-day EMA on ALL intraday closes (for direction filter)
    all_closes = df_intraday["close"]
    ema_fast_series = _ema(all_closes, FAST_EMA)
    ema_slow_series = _ema(all_closes, SLOW_EMA)

    expiry_dow   = EXPIRY_DOW.get(symbol, -1)

    for trade_date in all_dates:
        day_bars   = df_intraday[df_intraday.index.date == trade_date]
        hv         = _hv_daily(df_daily, pd.Timestamp(trade_date))
        is_expiry  = (trade_date.weekday() == expiry_dow)
        # Effective latest entry: 2:30 PM on expiry, 1 PM otherwise
        latest_entry = EXPIRY_LATEST_ENTRY if is_expiry else FIRST_HALF_END

        entry_done  = False
        open_trade: dict | None = None

        for idx, bar in day_bars.iterrows():
            t     = idx.time()
            t_str = idx.strftime("%H:%M")

            # ── Manage open position ──────────────────────────────────────────
            if open_trade is not None:
                S     = bar["close"]
                T_now = max(intraday_time_to_expiry(t_str), 1e-6)

                opt_now = bs_price(S, open_trade["strike"], T_now, hv,
                                   RISK_FREE_RATE, open_trade["opt_type"])

                reason = None
                if t >= FORCE_CLOSE:
                    reason = "EOD"
                elif opt_now <= open_trade["stop_prem"]:
                    reason = "STOP"
                elif (open_trade["opt_type"] == "call" and
                      bar["close"] > open_trade["breakout_level"]):
                    # Spot has closed above the OEH candle's open=high level — exit
                    reason = "TARGET"
                elif (open_trade["opt_type"] == "put" and
                      bar["close"] < open_trade["breakout_level"]):
                    # Spot has closed below the OEL candle's open=low level — exit
                    reason = "TARGET"

                if reason:
                    pnl_pts = (opt_now - open_trade["entry_prem"]) * open_trade["lots"]
                    pnl_inr = pnl_pts * lot
                    equity += pnl_inr
                    trades.append(Trade(
                        date        = str(trade_date),
                        instrument  = symbol,
                        timeframe   = timeframe,
                        is_expiry   = is_expiry,
                        direction   = open_trade["opt_type"],
                        signal_time = open_trade["signal_time"],
                        entry_time  = t_str,
                        spot_at_sig = open_trade["spot_at_sig"],
                        strike      = open_trade["strike"],
                        call_open   = open_trade["call_open"],
                        entry_prem  = open_trade["entry_prem"],
                        stop_prem   = open_trade["stop_prem"],
                        target_prem = open_trade["target_prem"],
                        exit_prem   = round(opt_now, 2),
                        exit_reason = reason,
                        pnl_pts     = round(pnl_pts, 2),
                        pnl_inr     = round(pnl_inr, 2),
                        lots        = open_trade["lots"],
                        lot_size    = lot,
                        equity      = round(equity, 0),
                    ))
                    open_trade = None
                continue

            # ── Look for OEH / OEL signal — first qualifying signal per day ──
            if entry_done or t < EARLIEST_ENTRY or t > latest_entry:
                continue

            # Cross-day EMAs at this exact bar (computed across all history)
            e_fast = ema_fast_series.loc[idx]
            e_slow = ema_slow_series.loc[idx]
            trending_up   = e_fast > e_slow
            trending_down = e_fast < e_slow

            # Short-term reversal from current day's bars so far
            day_closes_so_far = day_bars["close"].loc[:idx]
            if len(day_closes_so_far) >= REVERSAL_BARS:
                recent_slice = day_closes_so_far.iloc[-REVERSAL_BARS:]
                reversing_up   = not trending_up   and recent_slice.iloc[-1] > recent_slice.iloc[0]
                reversing_down = not trending_down and recent_slice.iloc[-1] < recent_slice.iloc[0]
            else:
                reversing_up = reversing_down = False

            if not (trending_up or trending_down or reversing_up or reversing_down):
                continue

            mkt_dir = ("up"   if (trending_up or reversing_up) else
                       "down" if (trending_down or reversing_down) else "neutral")

            # Need at least PRIOR_BARS of history within the day
            day_bars_so_far = list(day_bars["close"].loc[:idx].values)
            if len(day_bars_so_far) <= PRIOR_BARS:
                continue
            prior_candles = day_bars.loc[:idx].iloc[-(PRIOR_BARS+1):-1]  # bars before this one

            # ── CALL setup: spot Open = High, uptrend / reversing up ──────────
            if mkt_dir == "up" and is_open_eq_high(bar):
                spot_open = bar["open"]
                spot_cmp  = bar["close"]

                # Require a real pullback (not just 1-tick move)
                drop_pct = (spot_open - spot_cmp) / spot_open * 100
                if drop_pct < MIN_CANDLE_DROP:
                    continue

                # Require prior bars to have been rising (uptrend just before signal)
                up_bars = (prior_candles["close"] > prior_candles["open"]).sum()
                if up_bars < TREND_AGREE:
                    continue

                K         = atm_strike(spot_open, step)
                T_sig     = max(intraday_time_to_expiry(t_str), 1e-6)

                call_at_open  = bs_price(spot_open, K, T_sig, hv, RISK_FREE_RATE, "call")
                call_at_close = bs_price(spot_cmp,  K, T_sig, hv, RISK_FREE_RATE, "call")

                if call_at_close < CMP_FLOOR_PCT * call_at_open:
                    continue
                if call_at_close < 10:
                    continue

                entry_prem  = call_at_close
                stop_prem   = round(CMP_FLOOR_PCT * call_at_open, 2)
                risk        = entry_prem - stop_prem
                if risk <= 0:
                    continue
                # Target: spot breaks ABOVE the signal bar's open=high level
                # (user's description) — option price is then computed dynamically
                # target_prem is used only for Trade record display; exit is spot-based
                target_prem = round(entry_prem + rr * risk, 2)

                risk_inr         = equity * RISK_PCT / 100
                max_loss_per_lot = risk * lot
                n_lots = min(MAX_LOTS,
                             max(1, int(risk_inr / max_loss_per_lot))
                             if max_loss_per_lot > 0 else 1)

                open_trade = {
                    "opt_type": "call", "strike": K,
                    "call_open": round(call_at_open, 2),
                    "entry_prem": round(entry_prem, 2),
                    "stop_prem": stop_prem, "target_prem": target_prem,
                    "breakout_level": spot_open,   # spot must close ABOVE this for TARGET
                    "spot_at_sig": round(spot_open, 1), "signal_time": t_str,
                    "lots": n_lots,
                }
                entry_done = True
                continue

            # ── PUT setup: spot Open = Low, downtrend / reversing down ────────
            if mkt_dir == "down" and is_open_eq_low(bar):
                spot_open = bar["open"]
                spot_cmp  = bar["close"]

                # Require a real move up (not just 1 tick)
                rise_pct = (spot_cmp - spot_open) / spot_open * 100
                if rise_pct < MIN_CANDLE_DROP:
                    continue

                # Require prior bars to have been falling (downtrend just before signal)
                down_bars = (prior_candles["close"] < prior_candles["open"]).sum()
                if down_bars < TREND_AGREE:
                    continue

                K         = atm_strike(spot_open, step)
                T_sig     = max(intraday_time_to_expiry(t_str), 1e-6)

                put_at_open  = bs_price(spot_open, K, T_sig, hv, RISK_FREE_RATE, "put")
                put_at_close = bs_price(spot_cmp,  K, T_sig, hv, RISK_FREE_RATE, "put")

                if put_at_close < CMP_FLOOR_PCT * put_at_open:
                    continue
                if put_at_close < 10:
                    continue

                entry_prem  = put_at_close
                stop_prem   = round(CMP_FLOOR_PCT * put_at_open, 2)
                risk        = entry_prem - stop_prem
                if risk <= 0:
                    continue
                target_prem = round(entry_prem + rr * risk, 2)

                risk_inr         = equity * RISK_PCT / 100
                max_loss_per_lot = risk * lot
                n_lots = min(MAX_LOTS,
                             max(1, int(risk_inr / max_loss_per_lot))
                             if max_loss_per_lot > 0 else 1)

                open_trade = {
                    "opt_type": "put", "strike": K,
                    "call_open": round(put_at_open, 2),
                    "entry_prem": round(entry_prem, 2),
                    "stop_prem": stop_prem, "target_prem": target_prem,
                    "breakout_level": spot_open,   # spot must close BELOW this for TARGET
                    "spot_at_sig": round(spot_open, 1), "signal_time": t_str,
                    "lots": n_lots,
                }
                entry_done = True

    return trades, equity


# ── Display ────────────────────────────────────────────────────────────────────

def show(trades: list[Trade], end_equity: float, cap: float,
         label: str, rr: float) -> dict:
    if not trades:
        print(f"\n  {label}: NO TRADES FOUND")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["won"]  = df["pnl_inr"] > 0

    total  = len(df)
    won    = df["won"].sum()
    wr     = won / total * 100
    avg_w  = df.loc[df["won"],  "pnl_inr"].mean() if won > 0 else 0
    avg_l  = df.loc[~df["won"], "pnl_inr"].mean() if (total - won) > 0 else 0
    exp    = (wr/100) * avg_w + (1 - wr/100) * avg_l
    ret    = (end_equity - cap) / cap * 100
    years  = max((df["date"].max() - df["date"].min()).days / 365.25, 0.1)
    cagr   = ((end_equity / cap) ** (1 / years) - 1) * 100

    eq_arr = df["equity"].values
    peak = eq_arr[0]; max_dd = 0
    for e in eq_arr:
        peak   = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    print(f"\n{'='*65}")
    print(f"  {label}  [RR 1:{rr:.0f}]")
    print(f"{'='*65}")
    print(f"  ₹{cap:,.0f} → ₹{end_equity:,.0f}  ({ret:+.1f}%)")
    print(f"  CAGR: {cagr:+.1f}%   MaxDD: {max_dd:.1f}%")
    print(f"  Trades: {total}  ({total/years:.0f}/yr)  Win: {wr:.0f}%  ({int(won)}W/{total-int(won)}L)")
    print(f"  Avg win: ₹{avg_w:,.0f}   Avg loss: ₹{avg_l:,.0f}")
    print(f"  Expectancy: ₹{exp:,.0f}/trade")

    # Yearly bar chart
    yr_pnl = df.groupby("year")["pnl_inr"].sum()
    max_abs = max(abs(yr_pnl).max(), 1)
    print(f"\n  {'Year':<6} {'T':>4} {'W%':>5}  {'P&L ₹':>12}  Bar")
    for yr, grp in df.groupby("year"):
        n   = len(grp)
        w   = grp["won"].sum()
        pnl = grp["pnl_inr"].sum()
        bar_w = int(abs(pnl) / max_abs * 18)
        ch    = "█" if pnl >= 0 else "░"
        sign  = "+" if pnl >= 0 else ""
        print(f"  {yr:<6} {n:>4} {w/n*100:>4.0f}%  {sign}{pnl:>11,.0f}  {ch*bar_w}")

    # Exit breakdown
    print(f"\n  Exit breakdown:")
    for reason, grp in df.groupby("exit_reason"):
        n   = len(grp)
        pnl = grp["pnl_inr"].sum()
        wr_ = grp["won"].mean() * 100
        print(f"    {reason:<8} {n:>4}  win={wr_:3.0f}%  P&L=₹{pnl:+,.0f}")

    # Call vs Put split
    for side, grp in df.groupby("direction"):
        print(f"  {side.upper():>4}  {len(grp):>4} trades  win={grp['won'].mean()*100:.0f}%  "
              f"P&L=₹{grp['pnl_inr'].sum():+,.0f}")

    # Expiry vs pre-expiry breakdown
    if "is_expiry" in df.columns:
        for day_label, mask in [("Expiry day", df["is_expiry"]), ("Pre-expiry", ~df["is_expiry"])]:
            g = df[mask]
            if not g.empty:
                print(f"  {day_label:<12} {len(g):>4} trades  "
                      f"win={g['won'].mean()*100:.0f}%  P&L=₹{g['pnl_inr'].sum():+,.0f}")

    # Signal time distribution
    df["sig_hr"] = pd.to_datetime(df["signal_time"], format="%H:%M").dt.hour
    print(f"\n  Signal time distribution:")
    for hr, grp in df.groupby("sig_hr"):
        w   = grp["won"].mean() * 100
        pnl = grp["pnl_inr"].sum()
        print(f"    {hr:02d}:xx  {len(grp):>3} trades  win={w:.0f}%  P&L=₹{pnl:+,.0f}")

    return {
        "label": label, "cagr": cagr, "max_dd": max_dd,
        "win_rate": wr, "exp": exp, "total_return": ret,
        "trades": total, "rr": rr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")

    def _try_load(sym, interval):
        try:
            df = load_historical(sym, interval)
            return df if df is not None and len(df) > 0 else None
        except Exception:
            return None

    n3  = _try_load("NIFTY50", "3min")
    n5  = load_historical("NIFTY50", "5min")
    n1d = load_historical("NIFTY50", "1day")
    s3  = _try_load("SENSEX",  "3min")
    s5  = load_historical("SENSEX",  "5min")
    s1d = load_historical("SENSEX",  "1day")

    for df in [n3, n5, n1d, s3, s5, s1d]:
        if df is not None and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    has_3min = (n3 is not None and len(n3) > 0 and
                s3 is not None and len(s3) > 0)

    if has_3min:
        print(f"  Nifty  3min: {len(n3):,} bars")
        print(f"  Sensex 3min: {len(s3):,} bars")
    print(f"  Nifty  5min: {len(n5):,} bars")
    print(f"  Sensex 5min: {len(s5):,} bars")

    CAP = 500_000
    all_results = []

    for rr in [2.0, 3.0]:
        print(f"\n{'─'*65}")
        print(f"  TESTING RR = 1:{rr:.0f}")
        print(f"{'─'*65}")

        if has_3min:
            t1, eq1 = run_oeh("NIFTY50", n3, n1d, "3min", CAP, rr)
            r = show(t1, eq1, CAP, "NIFTY50 | OEH/OEL | ATM | 3-min", rr)
            if r: all_results.append(r)

            t2, eq2 = run_oeh("SENSEX", s3, s1d, "3min", CAP, rr)
            r = show(t2, eq2, CAP, "SENSEX  | OEH/OEL | ATM | 3-min", rr)
            if r: all_results.append(r)

        t3, eq3 = run_oeh("NIFTY50", n5, n1d, "5min", CAP, rr)
        r = show(t3, eq3, CAP, "NIFTY50 | OEH/OEL | ATM | 5-min", rr)
        if r: all_results.append(r)

        t4, eq4 = run_oeh("SENSEX", s5, s1d, "5min", CAP, rr)
        r = show(t4, eq4, CAP, "SENSEX  | OEH/OEL | ATM | 5-min", rr)
        if r: all_results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        print(f"\n\n{'='*72}")
        print("  SUMMARY — Open=High / Open=Low ATM Strategy")
        print(f"{'='*72}")
        print(f"  {'Strategy':<40} {'RR':>4} {'CAGR':>7} {'DD':>6} {'Win%':>6} {'Exp':>9}")
        print(f"  {'─'*70}")
        for r in all_results:
            sign = "+" if r["cagr"] >= 0 else ""
            print(f"  {r['label'][:40]:<40} 1:{r['rr']:.0f}  "
                  f"{sign}{r['cagr']:>5.1f}%  {r['max_dd']:>5.1f}%  "
                  f"{r['win_rate']:>5.0f}%  ₹{r['exp']:>+7,.0f}")

    print(f"\n  Notes:")
    print(f"  • Open=High tolerance: {OEH_TOLERANCE}% — spot opens at top, sells off ≥{MIN_CANDLE_DROP}%")
    print(f"  • CMP filter: option price > {CMP_FLOOR_PCT*100:.0f}% of signal-bar opening price")
    print(f"  • Prior trend: {TREND_AGREE}/{PRIOR_BARS} bars must be up (call) or down (put)")
    print(f"  • Timing: first half 9:20–13:00 all days │ 13:00–14:30 expiry days only")
    print(f"  • Stop: option falls to 50% of signal-bar open. Target: spot breaks OEH level.")
    print(f"  • ATM only. Force-close: {FORCE_CLOSE.strftime('%H:%M')}")


if __name__ == "__main__":
    main()
