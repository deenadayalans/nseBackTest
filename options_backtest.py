#!/usr/bin/env python3
"""
options_backtest.py — Index Options Strategy Backtester

Three strategies, day-of-week filtered per instrument:

  Strategy 1 — SELL ATM STRADDLE (expiry day)
      Sell ATM call + ATM put at 9:30 AM on expiry day.
      Collect full theta decay. Stop if combined premium hits 2× collected.
      Close at 3:15 PM.

  Strategy 2 — SELL OTM STRANGLE (pre-expiry day)
      Day before expiry: sell ±1.5% OTM call + put.
      Overnight hold. Close on expiry at 3:15 PM or stop at 2× premium.

  Strategy 3 — BUY ATM DIRECTIONAL (ORB + expiry day)
      On expiry day: check if Nifty/Sensex breaks above the 9:15 opening range.
      If yes → buy ATM call. If breaks down → buy ATM put.
      Target 2× premium. Stop at 50% premium loss.

Day-of-week schedule:
  NIFTY50  → Monday (pre) + Tuesday (expiry)
  SENSEX   → Wednesday (pre) + Thursday (expiry)
  General  → Friday  (for BANKNIFTY placeholder)

Pricing: Black-Scholes with 20-day rolling historical volatility.
         This is a SIMULATION — actual options prices differ slightly.
         Use as strategy research, not absolute P&L prediction.

Run:
    python options_backtest.py
"""

import sys
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kite_data import load_historical, add_features
from options_utils import (
    INSTRUMENTS, RISK_FREE_RATE,
    bs_price, atm_strike, otm_call_strike, otm_put_strike,
    hist_vol, straddle_premium, strangle_premium,
    intraday_time_to_expiry, overnight_time_to_expiry,
)

MARKET_OPEN_T  = time(9, 15)
ORB_END_T      = time(9, 30)
ENTRY_TIME_T   = time(9, 30)   # straddle/strangle entry after ORB
FORCE_CLOSE_T  = time(15, 15)
HV_WINDOW      = 20
OTM_PCT        = 1.5           # % away from ATM for strangle
STRADDLE_STOP  = 2.0           # stop when position moves to 2× premium collected
DIRECTIONAL_STOP_PCT = 0.50    # stop when option loses 50% of cost
DIRECTIONAL_TARGET   = 2.0     # target when option gains 2× cost


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:        str
    instrument:  str
    strategy:    str
    legs:        str          # description of what was sold/bought
    premium_collected: float  # positive = collected (sell), negative = paid (buy)
    lots:        int
    entry_spot:  float
    exit_spot:   float
    pnl_pts:     float        # in index points
    pnl_inr:     float        # in rupees (pnl_pts × lot_size)
    exit_reason: str
    hv:          float
    equity_after: float


# ── Core helpers ──────────────────────────────────────────────────────────────

def _compute_hv(daily_df: pd.DataFrame, as_of_date: pd.Timestamp) -> float:
    """20-day HV from daily close prices, annualised."""
    past = daily_df[daily_df.index.date < as_of_date.date()]["close"].tail(HV_WINDOW + 1)
    if len(past) < HV_WINDOW:
        return 0.15   # fallback 15%
    log_ret = np.log(past / past.shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(252))


def _spot_at(intraday_df: pd.DataFrame, date, t: time) -> float:
    """Get close price at a specific time on a specific date."""
    day_bars = intraday_df[intraday_df.index.date == date]
    matches  = day_bars[day_bars.index.time == t]
    if matches.empty:
        # fallback: last bar of that date
        return day_bars["close"].iloc[-1] if not day_bars.empty else np.nan
    return float(matches["close"].iloc[-1])


def _opening_range(intraday_df: pd.DataFrame, date) -> tuple:
    """High/low of 9:15 bar. Returns (high, low) or (nan, nan)."""
    day_bars = intraday_df[intraday_df.index.date == date]
    open_bar = day_bars[day_bars.index.time == MARKET_OPEN_T]
    if open_bar.empty:
        return np.nan, np.nan
    return float(open_bar["high"].iloc[0]), float(open_bar["low"].iloc[0])


def _lots_from_risk(equity: float, risk_pct: float, premium_pts: float,
                    lot_size: int) -> int:
    """Risk-based lot sizing. Min 1 lot."""
    if premium_pts <= 0:
        return 1
    risk_inr = equity * risk_pct / 100
    raw_lots = risk_inr / (premium_pts * lot_size)
    return max(1, int(raw_lots))


# ── Strategy 1: Sell ATM Straddle on Expiry Day ───────────────────────────────

def run_straddle_sell(
    symbol: str,
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    initial_capital: float = 500_000,
    risk_pct: float = 2.0,
) -> tuple[list[Trade], float]:
    """
    Sell ATM straddle at 9:30 on every expiry day.
    Stop: combined mark-to-market hits 2× premium.
    Close: 3:15 PM.
    """
    cfg      = INSTRUMENTS[symbol]
    lot      = cfg["lot"]
    step     = cfg["strike_step"]
    exp_dow  = cfg["expiry_dow"]   # weekday of expiry

    equity  = initial_capital
    trades  = []

    # Get all expiry days in the data
    all_dates = sorted(set(intraday_df.index.date))
    expiry_dates = [d for d in all_dates if d.weekday() == exp_dow]

    for exp_date in expiry_dates:
        exp_ts = pd.Timestamp(exp_date)
        hv     = _compute_hv(daily_df, exp_ts)
        if hv < 0.05:
            hv = 0.15   # floor

        # Entry at 9:30 (after ORB bar)
        entry_spot = _spot_at(intraday_df, exp_date, ENTRY_TIME_T)
        if np.isnan(entry_spot):
            continue

        K     = atm_strike(entry_spot, step)
        T_entry = intraday_time_to_expiry("09:30")
        sd    = straddle_premium(entry_spot, K, T_entry, hv)
        coll  = sd["total"]   # premium collected per unit (index points)
        if coll < 1:
            continue

        n_lots    = _lots_from_risk(equity, risk_pct, coll, lot)
        stop_mark = coll * STRADDLE_STOP   # stop if mark-to-market loss > this

        # Simulate bar by bar
        day_bars = intraday_df[intraday_df.index.date == exp_date]
        day_bars = day_bars[day_bars.index.time >= ENTRY_TIME_T]

        exit_spot   = entry_spot
        exit_reason = "EOD"
        final_prem  = 0.0

        for idx, bar in day_bars.iterrows():
            S      = bar["close"]
            t_str  = idx.strftime("%H:%M")
            T_now  = intraday_time_to_expiry(t_str)

            call_now = bs_price(S, K, T_now, hv, RISK_FREE_RATE, "call")
            put_now  = bs_price(S, K, T_now, hv, RISK_FREE_RATE, "put")
            mark     = call_now + put_now   # current cost to buy back

            if idx.time() >= FORCE_CLOSE_T:
                exit_spot   = S
                final_prem  = mark
                exit_reason = "EOD"
                break

            if mark >= stop_mark:
                exit_spot   = S
                final_prem  = mark
                exit_reason = "STOP"
                break
        else:
            exit_spot  = _spot_at(intraday_df, exp_date, FORCE_CLOSE_T)
            if np.isnan(exit_spot):
                exit_spot = entry_spot
            T_close   = intraday_time_to_expiry("15:15")
            final_prem = (bs_price(exit_spot, K, T_close, hv, RISK_FREE_RATE, "call")
                        + bs_price(exit_spot, K, T_close, hv, RISK_FREE_RATE, "put"))

        # P&L: we sold at coll, bought back at final_prem
        pnl_pts = (coll - final_prem) * n_lots
        pnl_inr = pnl_pts * lot
        equity  += pnl_inr

        trades.append(Trade(
            date        = str(exp_date),
            instrument  = symbol,
            strategy    = "ATM Straddle Sell",
            legs        = f"Sell {K}C + {K}P",
            premium_collected = round(coll, 2),
            lots        = n_lots,
            entry_spot  = round(entry_spot, 1),
            exit_spot   = round(exit_spot, 1),
            pnl_pts     = round(pnl_pts, 2),
            pnl_inr     = round(pnl_inr, 2),
            exit_reason = exit_reason,
            hv          = round(hv * 100, 1),
            equity_after= round(equity, 0),
        ))

    return trades, equity


# ── Strategy 2: Sell OTM Strangle Pre-Expiry ─────────────────────────────────

def run_strangle_sell(
    symbol: str,
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    initial_capital: float = 500_000,
    risk_pct: float = 2.0,
    otm_pct: float = OTM_PCT,
) -> tuple[list[Trade], float]:
    """
    Day before expiry: sell OTM call + put (strangle).
    Entry at 9:30. Hold overnight. Close on expiry day at 3:15.
    Stop: combined mark-to-market hits 2× premium at any intraday bar.
    """
    cfg      = INSTRUMENTS[symbol]
    lot      = cfg["lot"]
    step     = cfg["strike_step"]
    exp_dow  = cfg["expiry_dow"]
    pre_dow  = cfg["pre_dow"]

    equity  = initial_capital
    trades  = []

    all_dates   = sorted(set(intraday_df.index.date))
    expiry_dates = [d for d in all_dates if d.weekday() == exp_dow]
    pre_dates    = [d for d in all_dates if d.weekday() == pre_dow]

    for exp_date in expiry_dates:
        # Find most recent pre-expiry day
        candidates = [d for d in pre_dates if d < exp_date]
        if not candidates:
            continue
        pre_date = candidates[-1]

        # Check it's actually the day before (not further back)
        if (exp_date - pre_date).days > 4:
            continue

        exp_ts = pd.Timestamp(pre_date)
        hv     = _compute_hv(daily_df, exp_ts)
        if hv < 0.05:
            hv = 0.15

        # Entry on pre-expiry day at 9:30
        entry_spot = _spot_at(intraday_df, pre_date, ENTRY_TIME_T)
        if np.isnan(entry_spot):
            continue

        call_K = otm_call_strike(entry_spot, step, otm_pct)
        put_K  = otm_put_strike(entry_spot,  step, otm_pct)

        # T = 1 + fraction of day remaining (overnight + next day)
        T_entry = overnight_time_to_expiry(1.5)   # ~1.5 trading days
        sg  = strangle_premium(entry_spot, call_K, put_K, T_entry, hv)
        coll = sg["total"]
        if coll < 0.5:
            continue

        n_lots    = _lots_from_risk(equity, risk_pct, coll, lot)
        stop_mark = coll * STRADDLE_STOP

        # Simulate — overnight gap risk then expiry day bars
        exit_spot   = entry_spot
        exit_reason = "EOD"
        final_prem  = coll  # default: no change

        # Check next day bars (expiry day)
        exp_bars = intraday_df[intraday_df.index.date == exp_date]

        for idx, bar in exp_bars.iterrows():
            S     = bar["close"]
            t_str = idx.strftime("%H:%M")
            T_now = intraday_time_to_expiry(t_str)

            call_now = bs_price(S, call_K, T_now, hv, RISK_FREE_RATE, "call")
            put_now  = bs_price(S, put_K,  T_now, hv, RISK_FREE_RATE, "put")
            mark     = call_now + put_now

            if idx.time() >= FORCE_CLOSE_T:
                exit_spot   = S
                final_prem  = mark
                exit_reason = "EOD"
                break

            if mark >= stop_mark:
                exit_spot   = S
                final_prem  = mark
                exit_reason = "STOP"
                break

        pnl_pts = (coll - final_prem) * n_lots
        pnl_inr = pnl_pts * lot
        equity  += pnl_inr

        trades.append(Trade(
            date        = str(pre_date),
            instrument  = symbol,
            strategy    = "OTM Strangle Sell",
            legs        = f"Sell {call_K}C + {put_K}P",
            premium_collected = round(coll, 2),
            lots        = n_lots,
            entry_spot  = round(entry_spot, 1),
            exit_spot   = round(exit_spot, 1),
            pnl_pts     = round(pnl_pts, 2),
            pnl_inr     = round(pnl_inr, 2),
            exit_reason = exit_reason,
            hv          = round(hv * 100, 1),
            equity_after= round(equity, 0),
        ))

    return trades, equity


# ── Strategy 3: Buy ATM Directional (ORB + Expiry) ───────────────────────────

def run_directional_atm(
    symbol: str,
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    initial_capital: float = 500_000,
    risk_pct: float = 1.0,
    max_range_pct: float = 0.5,
) -> tuple[list[Trade], float]:
    """
    On expiry day: use ORB to pick direction.
    Tight opening range (< max_range_pct%) → buy ATM call (breakout up)
                                            or ATM put (breakout down).
    Target: 2× premium. Stop: 50% loss of premium paid.
    """
    cfg      = INSTRUMENTS[symbol]
    lot      = cfg["lot"]
    step     = cfg["strike_step"]
    exp_dow  = cfg["expiry_dow"]

    equity  = initial_capital
    trades  = []

    all_dates    = sorted(set(intraday_df.index.date))
    expiry_dates = [d for d in all_dates if d.weekday() == exp_dow]

    for exp_date in expiry_dates:
        exp_ts = pd.Timestamp(exp_date)
        hv     = _compute_hv(daily_df, exp_ts)
        if hv < 0.05:
            hv = 0.15

        or_high, or_low = _opening_range(intraday_df, exp_date)
        if np.isnan(or_high):
            continue

        or_range = or_high - or_low
        or_mid   = (or_high + or_low) / 2

        # Filter: only trade tight opens
        if or_range / or_low * 100 > max_range_pct:
            continue

        # Entry price: close of first bar after ORB (9:30 bar)
        entry_spot = _spot_at(intraday_df, exp_date, ENTRY_TIME_T)
        if np.isnan(entry_spot):
            continue

        # Direction from ORB
        if entry_spot > or_high:
            direction   = "CALL"
            option_type = "call"
        elif entry_spot < or_low:
            direction   = "PUT"
            option_type = "put"
        else:
            continue   # no clear break

        K      = atm_strike(entry_spot, step)
        T_entry = intraday_time_to_expiry("09:30")
        prem   = bs_price(entry_spot, K, T_entry, hv, RISK_FREE_RATE, option_type)
        if prem < 1:
            continue

        stop_prem   = prem * (1 - DIRECTIONAL_STOP_PCT)
        target_prem = prem * DIRECTIONAL_TARGET
        n_lots      = _lots_from_risk(equity, risk_pct, prem, lot)

        day_bars = intraday_df[intraday_df.index.date == exp_date]
        day_bars = day_bars[day_bars.index.time >= ENTRY_TIME_T]

        exit_spot   = entry_spot
        exit_reason = "EOD"
        final_prem  = 0.0

        for idx, bar in day_bars.iterrows():
            S     = bar["close"]
            t_str = idx.strftime("%H:%M")
            T_now = intraday_time_to_expiry(t_str)

            opt_now = bs_price(S, K, T_now, hv, RISK_FREE_RATE, option_type)

            if idx.time() >= FORCE_CLOSE_T:
                exit_spot   = S
                final_prem  = opt_now
                exit_reason = "EOD"
                break

            if opt_now <= stop_prem:
                exit_spot   = S
                final_prem  = opt_now
                exit_reason = "STOP"
                break

            if opt_now >= target_prem:
                exit_spot   = S
                final_prem  = opt_now
                exit_reason = "TARGET"
                break

        # P&L: we bought at prem, sold at final_prem
        pnl_pts = (final_prem - prem) * n_lots
        pnl_inr = pnl_pts * lot
        equity  += pnl_inr

        trades.append(Trade(
            date        = str(exp_date),
            instrument  = symbol,
            strategy    = f"Directional ATM {direction}",
            legs        = f"Buy {K}{direction[0]}",
            premium_collected = -round(prem, 2),   # negative = paid
            lots        = n_lots,
            entry_spot  = round(entry_spot, 1),
            exit_spot   = round(exit_spot, 1),
            pnl_pts     = round(pnl_pts, 2),
            pnl_inr     = round(pnl_inr, 2),
            exit_reason = exit_reason,
            hv          = round(hv * 100, 1),
            equity_after= round(equity, 0),
        ))

    return trades, equity


# ── Analysis & display ────────────────────────────────────────────────────────

def analyse(trades: list[Trade], end_equity: float,
            initial_capital: float, label: str) -> dict:
    if not trades:
        print(f"\n  {label}: NO TRADES")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["won"]   = df["pnl_inr"] > 0

    total   = len(df)
    won     = df["won"].sum()
    wr      = won / total * 100
    avg_w   = df.loc[df["won"], "pnl_inr"].mean()
    avg_l   = df.loc[~df["won"], "pnl_inr"].mean()
    exp     = (wr/100 * avg_w) + ((1 - wr/100) * avg_l)
    ret_pct = (end_equity - initial_capital) / initial_capital * 100
    years   = (df["date"].max() - df["date"].min()).days / 365.25
    cagr    = ((end_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0

    # Max drawdown
    equity_curve = df["equity_after"].values
    peak = equity_curve[0]
    max_dd = 0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  ₹{initial_capital:,.0f} → ₹{end_equity:,.0f}  ({ret_pct:+.1f}%)")
    print(f"  CAGR         : {cagr:+.1f}%   MaxDD: {max_dd:.1f}%")
    print(f"  Trades       : {total}  Win: {wr:.0f}%  ({int(won)}W/{total-int(won)}L)")
    print(f"  Avg win      : ₹{avg_w:,.0f}   Avg loss: ₹{avg_l:,.0f}")
    print(f"  Expectancy   : ₹{exp:,.0f} / trade")

    # Yearly
    print(f"\n  {'Year':<6} {'T':>4} {'W%':>5}  {'P&L ₹':>12}  Bar")
    for yr, grp in df.groupby("year"):
        n   = len(grp)
        w   = grp["won"].sum()
        pnl = grp["pnl_inr"].sum()
        bar_w = int(abs(pnl) / max(abs(df.groupby("year")["pnl_inr"].sum()).max(), 1) * 18)
        ch    = "█" if pnl >= 0 else "░"
        sign  = "+" if pnl >= 0 else ""
        print(f"  {yr:<6} {n:>4} {w/n*100:>4.0f}%  {sign}{pnl:>11,.0f}  {ch*bar_w}")

    # Exit reasons
    print(f"\n  Exit reasons:")
    for reason, grp in df.groupby("exit_reason"):
        n   = len(grp)
        pnl = grp["pnl_inr"].sum()
        print(f"    {reason:<8} {n:>4} trades  P&L=₹{pnl:+,.0f}")

    return {
        "label": label, "trades": total, "win_rate": wr,
        "cagr": cagr, "max_dd": max_dd, "expectancy": exp,
        "total_return": ret_pct,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")

    nifty_15  = load_historical("NIFTY50",  "15min")
    nifty_1d  = load_historical("NIFTY50",  "1day")
    sensex_15 = load_historical("SENSEX",   "15min")
    sensex_1d = load_historical("SENSEX",   "1day")

    # Strip timezone from Sensex too
    for df in [sensex_15, sensex_1d]:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    print(f"  Nifty  15min: {len(nifty_15):,} bars")
    print(f"  Sensex 15min: {len(sensex_15):,} bars\n")

    CAP = 500_000

    # ── NIFTY — Sell ATM Straddle (Tuesday expiry) ────────────────────────────
    t1, eq1 = run_straddle_sell("NIFTY50", nifty_15, nifty_1d, CAP)
    r1 = analyse(t1, eq1, CAP, "NIFTY50 | Sell ATM Straddle | Tue Expiry")

    # ── NIFTY — Sell OTM Strangle (Monday pre-expiry) ────────────────────────
    t2, eq2 = run_strangle_sell("NIFTY50", nifty_15, nifty_1d, CAP)
    r2 = analyse(t2, eq2, CAP, "NIFTY50 | Sell OTM Strangle ±1.5% | Mon+Tue")

    # ── NIFTY — Buy ATM Directional (ORB on Tuesday expiry) ──────────────────
    t3, eq3 = run_directional_atm("NIFTY50", nifty_15, nifty_1d, CAP)
    r3 = analyse(t3, eq3, CAP, "NIFTY50 | Buy ATM Directional (ORB) | Tue Expiry")

    # ── SENSEX — Sell ATM Straddle (Thursday expiry) ─────────────────────────
    t4, eq4 = run_straddle_sell("SENSEX", sensex_15, sensex_1d, CAP)
    r4 = analyse(t4, eq4, CAP, "SENSEX  | Sell ATM Straddle | Thu Expiry")

    # ── SENSEX — Sell OTM Strangle (Wednesday pre-expiry) ────────────────────
    t5, eq5 = run_strangle_sell("SENSEX", sensex_15, sensex_1d, CAP)
    r5 = analyse(t5, eq5, CAP, "SENSEX  | Sell OTM Strangle ±1.5% | Wed+Thu")

    # ── SENSEX — Buy ATM Directional (ORB on Thursday expiry) ────────────────
    t6, eq6 = run_directional_atm("SENSEX", sensex_15, sensex_1d, CAP)
    r6 = analyse(t6, eq6, CAP, "SENSEX  | Buy ATM Directional (ORB) | Thu Expiry")

    # ── Master comparison ─────────────────────────────────────────────────────
    results = [r for r in [r1, r2, r3, r4, r5, r6] if r]
    if not results:
        return

    print(f"\n\n{'='*75}")
    print("  MASTER COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Strategy':<42} {'CAGR':>7} {'MaxDD':>7} {'Win%':>6} {'Exp ₹':>9} {'Total%':>8}")
    print(f"  {'─'*72}")
    for r in results:
        lbl  = r["label"][:42]
        sign = "+" if r["cagr"] >= 0 else ""
        print(f"  {lbl:<42} {sign}{r['cagr']:>5.1f}%  {r['max_dd']:>5.1f}%  "
              f"{r['win_rate']:>5.0f}%  {r['expectancy']:>+8,.0f}  {r['total_return']:>+7.1f}%")

    print(f"\n  ⚠  Pricing via Black-Scholes + 20d HV. Actual options prices differ.")
    print(f"     This shows strategy DIRECTION, not precise P&L.")
    print(f"     Validate with real options chain data before going live.")


if __name__ == "__main__":
    main()
