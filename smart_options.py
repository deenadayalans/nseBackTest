#!/usr/bin/env python3
"""
smart_options.py — ORB Options Buyer (Nifty + Sensex)

Concept
-------
On expiry and pre-expiry days, wait for the Opening Range (first 30 min)
to break, then buy an option in the breakout direction.
5-strategy aggregator provides directional confirmation (3/5 minimum).

Strike selection (user-defined):
  Expiry day   → ATM (maximum gamma; premium ~₹100–300)
  Pre-expiry   → 1–2 strikes OTM (cheaper, still has overnight time value)

Why ORB entry instead of open-of-day?
  • Avoids entering into early-session noise that triggers stops
  • Breakout gives a clear directional catalyst to enter on
  • EOD hold rate improves: only enter when momentum has started

Schedule:
  NIFTY50 → Tuesday (expiry) + Monday (pre-expiry)
  SENSEX  → Thursday (expiry) + Wednesday (pre-expiry)

Run:
    python smart_options.py
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

ORB_BARS        = 6              # 6 × 5-min = 30-min opening range (9:15–9:44)
LATEST_ENTRY    = time(11, 0)    # no new entries after 11 AM
FORCE_CLOSE     = time(15, 0)    # close by 3 PM (liquidity thins after)
MIN_PREMIUM     = 30             # min premium to enter (₹) — wider for ATM expiry
STOP_PCT        = 0.60           # exit if premium loses 60% (wider: ORB gives cleaner bias)
RR              = 2.0            # reward-to-risk ratio (2 = 1:2, 3 = 1:3)
AGG_THRESHOLD   = 65             # per-strategy min confidence to count as "agree"
MIN_AGREE       = 3              # relaxed to 3/5: ORB is the primary filter
MARGIN_GAP      = 10             # buy_conf must beat sell_conf by this
BASE_LOTS       = 1              # base position in lots
DOUBLE_LOTS     = 2              # lots when 5/5 unanimous
RISK_PCT        = 2.0            # % of capital to risk per trade
HV_WINDOW       = 20


# ── Strategy signal dataclass ─────────────────────────────────────────────────

@dataclass
class Signal:
    buy_conf:  float   # 0–100
    sell_conf: float   # 0–100
    regime:    str
    reason:    str


# ── 5 Strategies (each scores 4 × 25 pts) ────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"]  - df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _macd(s: pd.Series):
    m  = _ema(s, 12) - _ema(s, 26)
    sig = _ema(m, 9)
    return m, sig, m - sig

def _vwap(df: pd.DataFrame) -> pd.Series:
    """Daily-reset VWAP (typical price mean since 9:15 each day)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = tp.copy()
    last_date = None
    cum_sum = 0.0
    cnt = 0
    vals = []
    for ts, v in tp.items():
        d = ts.date()
        if d != last_date:
            cum_sum = v; cnt = 1; last_date = d
        else:
            cum_sum += v; cnt += 1
        vals.append(cum_sum / cnt)
    vwap[:] = vals
    return vwap


def strategy_ema_trend(df: pd.DataFrame) -> Signal:
    """EMA Trend Pullback — 4 × 25 pts."""
    if len(df) < 205:
        return Signal(0, 0, "warmup", "not enough bars")
    c    = df["close"]
    e20  = _ema(c, 20).iloc[-1]
    e200 = _ema(c, 200).iloc[-1]
    atr  = _atr(df).iloc[-1]
    last = c.iloc[-1]
    prev = c.iloc[-2]

    buy = 0
    buy += 25 if last > e200 else 0           # HTF trend up
    buy += 25 if e20 > e200 else 0            # EMA alignment bullish
    buy += 25 if abs(last - e20) < 0.5*atr else 0  # pullback zone
    buy += 25 if last > prev else 0           # bullish candle

    sell = 0
    sell += 25 if last < e200 else 0
    sell += 25 if e20 < e200 else 0
    sell += 25 if abs(last - e20) < 0.5*atr else 0
    sell += 25 if last < prev else 0

    regime = "uptrend" if last > e200 else "downtrend" if last < e200 else "transitioning"
    return Signal(buy, sell, regime, f"e20={e20:.0f} e200={e200:.0f} atr={atr:.0f}")


def strategy_rsi_reversion(df: pd.DataFrame) -> Signal:
    """RSI Mean Reversion — 4 × 25 pts."""
    if len(df) < 60:
        return Signal(0, 0, "warmup", "not enough bars")
    c    = df["close"]
    rsi  = _rsi(c, 14)
    r    = rsi.iloc[-1]
    r_1  = rsi.iloc[-2]
    e50  = _ema(c, 50).iloc[-1]
    atr  = _atr(df).iloc[-1]
    last = c.iloc[-1]
    prev = c.iloc[-2]

    buy = 0
    buy += 25 if r < 35 else 0              # oversold
    buy += 25 if r > r_1 else 0             # RSI turning up
    buy += 25 if abs(last - e50) < 1.5*atr else 0  # near mean
    buy += 25 if last > prev else 0         # bullish candle

    sell = 0
    sell += 25 if r > 65 else 0
    sell += 25 if r < r_1 else 0
    sell += 25 if abs(last - e50) < 1.5*atr else 0
    sell += 25 if last < prev else 0

    regime = "oversold" if r < 35 else "overbought" if r > 65 else "ranging"
    return Signal(buy, sell, regime, f"RSI={r:.1f}")


def strategy_bb_squeeze(df: pd.DataFrame) -> Signal:
    """Bollinger Band Squeeze — 4 × 25 pts."""
    if len(df) < 55:
        return Signal(0, 0, "warmup", "not enough bars")
    c     = df["close"]
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper = (sma20 + 2*std20).iloc[-1]
    lower = (sma20 - 2*std20).iloc[-1]
    width = upper - lower
    widths = (sma20 + 2*std20) - (sma20 - 2*std20)
    pct_rank = (width < widths.rolling(50).quantile(0.30)).iloc[-1]
    last = c.iloc[-1]
    prev = c.iloc[-2]

    buy = 0
    buy += 25 if pct_rank else 0             # squeeze active
    buy += 25 if last > upper else 0         # breaking up
    vol_ratio = (df["volume"].iloc[-1] /
                 df["volume"].rolling(20).mean().iloc[-1]) if "volume" in df else 1.0
    buy += min(25, int(vol_ratio * 12)) if vol_ratio > 1 else 0
    buy += 25 if last > prev else 0

    sell = 0
    sell += 25 if pct_rank else 0
    sell += 25 if last < lower else 0
    sell += min(25, int(vol_ratio * 12)) if vol_ratio > 1 else 0
    sell += 25 if last < prev else 0

    regime = "squeeze" if pct_rank else "expansion" if last > upper or last < lower else "normal"
    return Signal(buy, sell, regime, f"BB_width={width:.0f} vol_ratio={vol_ratio:.1f}")


def strategy_vwap_momentum(df: pd.DataFrame) -> Signal:
    """
    VWAP Momentum (replaces FVG — more relevant for Indian intraday).
    4 × 25 pts. Replaces FVG from the PDF because:
      • VWAP is used by every institutional desk in India
      • FVG had weakest results and needed 2 redesigns in the PDF
    """
    if len(df) < 25:
        return Signal(0, 0, "warmup", "not enough bars")
    c    = df["close"]
    vwap = _vwap(df).iloc[-1]
    atr  = _atr(df).iloc[-1]
    last = c.iloc[-1]
    prev = c.iloc[-2]
    e20  = _ema(c, 20).iloc[-1]

    dev = last - vwap

    buy = 0
    buy += 25 if last > vwap else 0                # above VWAP (institutional bias up)
    buy += 25 if dev > 0.3*atr else 0              # confirmed separation, not just touching
    buy += 25 if e20 > vwap else 0                 # trend aligned with VWAP
    buy += 25 if last > prev else 0                # momentum candle

    sell = 0
    sell += 25 if last < vwap else 0
    sell += 25 if dev < -0.3*atr else 0
    sell += 25 if e20 < vwap else 0
    sell += 25 if last < prev else 0

    regime = "above_vwap" if last > vwap else "below_vwap"
    return Signal(buy, sell, regime, f"vwap={vwap:.0f} dev={dev:+.0f} atr={atr:.0f}")


def strategy_macd_momentum(df: pd.DataFrame) -> Signal:
    """MACD Momentum — 4 × 25 pts."""
    if len(df) < 230:
        return Signal(0, 0, "warmup", "not enough bars")
    c        = df["close"]
    m, sig, hist = _macd(c)
    e200     = _ema(c, 200).iloc[-1]
    last     = c.iloc[-1]
    prev     = c.iloc[-2]
    m_now    = m.iloc[-1]
    s_now    = sig.iloc[-1]
    h_now    = hist.iloc[-1]
    h_prev   = hist.iloc[-2]

    # Fresh crossover: MACD crossed signal within last 3 bars
    crossed_up   = any(m.iloc[-i] > sig.iloc[-i] and m.iloc[-i-1] <= sig.iloc[-i-1]
                       for i in range(1, 4) if len(m) > i+1)
    crossed_down = any(m.iloc[-i] < sig.iloc[-i] and m.iloc[-i-1] >= sig.iloc[-i-1]
                       for i in range(1, 4) if len(m) > i+1)

    buy = 0
    buy += 25 if last > e200 else 0          # HTF bullish
    buy += 25 if m_now > s_now else 0        # MACD above signal
    buy += 25 if h_now > 0 and h_now > h_prev else 0  # accelerating
    buy += 25 if crossed_up else 0           # fresh crossover

    sell = 0
    sell += 25 if last < e200 else 0
    sell += 25 if m_now < s_now else 0
    sell += 25 if h_now < 0 and h_now < h_prev else 0
    sell += 25 if crossed_down else 0

    regime = "momentum_up" if m_now > s_now else "momentum_down"
    return Signal(buy, sell, regime, f"MACD={m_now:.1f} sig={s_now:.1f}")


# ── Aggregator ────────────────────────────────────────────────────────────────

STRATEGIES = [
    strategy_ema_trend,
    strategy_rsi_reversion,
    strategy_bb_squeeze,
    strategy_vwap_momentum,
    strategy_macd_momentum,
]
STRATEGY_NAMES = ["EMA", "RSI", "BB", "VWAP", "MACD"]


def aggregate(df: pd.DataFrame, threshold: int = AGG_THRESHOLD,
              min_agree: int = MIN_AGREE) -> dict:
    """
    Run all 5 strategies. Return recommendation + agreement count + scores.
    Returns: {direction: 'long'|'short'|'stand_aside', agree: int, unanimous: bool,
              scores: [(name, buy, sell)], avg_buy, avg_sell}
    """
    sigs = [s(df) for s in STRATEGIES]

    agree_buy  = sum(1 for s in sigs if s.buy_conf  >= threshold)
    agree_sell = sum(1 for s in sigs if s.sell_conf >= threshold)
    avg_buy    = np.mean([s.buy_conf  for s in sigs])
    avg_sell   = np.mean([s.sell_conf for s in sigs])

    direction = "stand_aside"
    if agree_buy >= min_agree and avg_buy >= threshold*0.8 and avg_buy > avg_sell + MARGIN_GAP:
        direction = "long"
    elif agree_sell >= min_agree and avg_sell >= threshold*0.8 and avg_sell > avg_buy + MARGIN_GAP:
        direction = "short"

    unanimous = (agree_buy == 5 and direction == "long") or \
                (agree_sell == 5 and direction == "short")

    return {
        "direction": direction,
        "agree":     agree_buy if direction == "long" else agree_sell,
        "unanimous": unanimous,
        "scores":    list(zip(STRATEGY_NAMES,
                              [s.buy_conf for s in sigs],
                              [s.sell_conf for s in sigs])),
        "avg_buy":   round(avg_buy, 1),
        "avg_sell":  round(avg_sell, 1),
        "regimes":   [s.regime for s in sigs],
    }


# ── OTM option selector ───────────────────────────────────────────────────────

def select_strike(spot: float, step: int, direction: str,
                  T: float, hv: float,
                  is_expiry: bool) -> tuple[int, float]:
    """
    Strike selection:
      Expiry day   → ATM (maximum gamma, premium ~₹100-300)
      Pre-expiry   → 1-2 strikes OTM (cheaper, still has time value)
    Returns (strike, premium).
    """
    opt_type = "call" if direction == "long" else "put"
    atm = atm_strike(spot, step)

    if is_expiry:
        # ATM on expiry day — highest gamma, biggest % move per spot point
        return atm, round(bs_price(spot, atm, T, hv, RISK_FREE_RATE, opt_type), 2)

    # Pre-expiry: try 1 strike OTM first, then 2 strikes
    for n in [1, 2]:
        K = (atm + n * step) if direction == "long" else (atm - n * step)
        p = bs_price(spot, K, T, hv, RISK_FREE_RATE, opt_type)
        if p >= MIN_PREMIUM:   # don't go so far out the option is worthless
            return K, round(p, 2)

    # Fallback: ATM if strikes are too cheap
    return atm, round(bs_price(spot, atm, T, hv, RISK_FREE_RATE, opt_type), 2)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:        str
    instrument:  str
    is_expiry:   bool
    direction:   str    # long/short
    option_type: str    # call/put
    strike:      int
    entry_prem:  float
    target_prem: float
    stop_prem:   float
    lots:        int
    lot_size:    int
    entry_spot:  float
    exit_spot:   float
    exit_prem:   float
    pnl_pts:     float
    pnl_inr:     float
    exit_reason: str
    agree:       int
    unanimous:   bool
    hv:          float
    equity:      float


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_smart_options(
    symbol: str,
    df5: pd.DataFrame,         # 5-min OHLCV
    df1d: pd.DataFrame,        # daily OHLCV (for HV)
    initial_capital: float = 500_000,
    rr: float = RR,
) -> tuple[list[Trade], float]:
    """
    Entry logic: Opening Range Breakout + 5-strategy confirmation.

    Phase 1 (first 30 min, bars 1-6): Build opening range (OR_high, OR_low).
    Phase 2 (9:45–11:00 AM): On first bar that closes outside the OR, check
      if the 5-strategy aggregator agrees (≥ MIN_AGREE). If yes, buy ATM
      (expiry) or 1-2 strike OTM (pre-expiry).

    Rationale: Pure open-of-day entries were stopped by early-session noise;
    the ORB ensures momentum has already started before we commit premium.
    """
    cfg      = INSTRUMENTS[symbol]
    lot      = cfg["lot"]
    step     = cfg["strike_step"]
    exp_dow  = cfg["expiry_dow"]
    pre_dow  = cfg["pre_dow"]

    equity = initial_capital
    trades = []

    all_dates    = sorted(set(df5.index.date))
    active_dates = [(d, d.weekday() == exp_dow) for d in all_dates
                    if d.weekday() in (exp_dow, pre_dow)]

    for trade_date, is_expiry in active_dates:
        day_bars = df5[df5.index.date == trade_date]
        past_df  = df5[df5.index.date <= trade_date].tail(250 + len(day_bars))
        hv       = _hv_daily(df1d, pd.Timestamp(trade_date))

        entry_done = False
        open_trade: dict | None = None
        or_high    = None
        or_low     = None
        bar_count  = 0

        for idx, bar in day_bars.iterrows():
            t = idx.time()

            # ── Phase 1: accumulate opening range (first ORB_BARS bars) ──────
            if bar_count < ORB_BARS:
                if or_high is None:
                    or_high = bar["high"]; or_low = bar["low"]
                else:
                    or_high = max(or_high, bar["high"])
                    or_low  = min(or_low,  bar["low"])
                bar_count += 1
                continue

            # ── Manage open position ──────────────────────────────────────────
            if open_trade is not None:
                S     = bar["close"]
                t_str = idx.strftime("%H:%M")
                T_now = (intraday_time_to_expiry(t_str)
                         if is_expiry else overnight_time_to_expiry(0.5))
                opt_now = bs_price(S, open_trade["strike"], T_now, hv,
                                   RISK_FREE_RATE, open_trade["opt_type"])

                reason = None
                if t >= FORCE_CLOSE:
                    reason = "EOD"
                elif opt_now <= open_trade["stop_prem"]:
                    reason = "STOP"
                elif opt_now >= open_trade["target_prem"]:
                    reason = "TARGET"

                if reason:
                    pnl_pts = (opt_now - open_trade["entry_prem"]) * open_trade["lots"]
                    pnl_inr = pnl_pts * lot
                    equity += pnl_inr
                    trades.append(Trade(
                        date        = str(trade_date),
                        instrument  = symbol,
                        is_expiry   = is_expiry,
                        direction   = open_trade["direction"],
                        option_type = open_trade["opt_type"],
                        strike      = open_trade["strike"],
                        entry_prem  = open_trade["entry_prem"],
                        target_prem = open_trade["target_prem"],
                        stop_prem   = open_trade["stop_prem"],
                        lots        = open_trade["lots"],
                        lot_size    = lot,
                        entry_spot  = open_trade["entry_spot"],
                        exit_spot   = round(S, 1),
                        exit_prem   = round(opt_now, 2),
                        pnl_pts     = round(pnl_pts, 2),
                        pnl_inr     = round(pnl_inr, 2),
                        exit_reason = reason,
                        agree       = open_trade["agree"],
                        unanimous   = open_trade["unanimous"],
                        hv          = round(hv * 100, 1),
                        equity      = round(equity, 0),
                    ))
                    open_trade = None
                continue

            # ── Phase 2: look for ORB breakout entry ─────────────────────────
            if entry_done or t > LATEST_ENTRY:
                continue

            close = bar["close"]
            if close > or_high:
                orb_direction = "long"
            elif close < or_low:
                orb_direction = "short"
            else:
                continue   # still inside opening range

            # Confirm with 5-strategy aggregator
            bars_so_far = past_df[past_df.index <= idx]
            if len(bars_so_far) < 230:
                continue

            agg = aggregate(bars_so_far)
            # ORB and aggregator must agree on direction
            agg_direction = agg["direction"]
            if agg_direction == "stand_aside":
                continue
            if agg_direction != orb_direction:
                continue   # aggregator disagrees — skip conflicting signals

            direction = orb_direction
            opt_type  = "call" if direction == "long" else "put"
            spot      = close
            t_str     = idx.strftime("%H:%M")

            T_entry = (intraday_time_to_expiry(t_str)
                       if is_expiry else overnight_time_to_expiry(1.0))

            strike, prem = select_strike(spot, step, direction, T_entry, hv, is_expiry)

            if prem < MIN_PREMIUM:
                continue

            stop_prem   = round(prem * (1 - STOP_PCT), 2)
            target_prem = round(prem * (1 + STOP_PCT * rr), 2)

            risk_inr         = equity * RISK_PCT / 100
            max_loss_per_lot = prem * lot * STOP_PCT
            n_lots = max(BASE_LOTS, int(risk_inr / max_loss_per_lot)) if max_loss_per_lot > 0 else BASE_LOTS
            if agg["unanimous"]:
                n_lots = max(n_lots, DOUBLE_LOTS)

            open_trade = {
                "direction":   direction,
                "opt_type":    opt_type,
                "strike":      strike,
                "entry_prem":  prem,
                "stop_prem":   stop_prem,
                "target_prem": target_prem,
                "entry_spot":  round(spot, 1),
                "lots":        n_lots,
                "agree":       agg["agree"],
                "unanimous":   agg["unanimous"],
            }
            entry_done = True

    return trades, equity


def _hv_daily(df1d: pd.DataFrame, as_of: pd.Timestamp) -> float:
    past = df1d[df1d.index < as_of]["close"].tail(HV_WINDOW + 1)
    if len(past) < 5:
        return 0.15
    lr = np.log(past / past.shift(1)).dropna()
    return max(0.08, float(lr.std() * np.sqrt(252)))


# ── Display ───────────────────────────────────────────────────────────────────

def show_results(trades: list[Trade], end_equity: float,
                 cap: float, label: str, rr: float):
    if not trades:
        print(f"\n  {label}: NO TRADES")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["won"]   = df["pnl_inr"] > 0

    total  = len(df)
    won    = df["won"].sum()
    wr     = won / total * 100
    avg_w  = df.loc[df["won"],  "pnl_inr"].mean()
    avg_l  = df.loc[~df["won"], "pnl_inr"].mean()
    exp    = (wr/100*avg_w) + ((1-wr/100)*avg_l)
    ret    = (end_equity - cap) / cap * 100
    years  = (df["date"].max() - df["date"].min()).days / 365.25
    cagr   = ((end_equity / cap) ** (1/years) - 1) * 100 if years > 0 else 0

    eq_arr = df["equity"].values
    peak   = eq_arr[0]; max_dd = 0
    for e in eq_arr:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    unani = df["unanimous"].sum()

    print(f"\n{'='*65}")
    print(f"  {label}  [RR 1:{rr:.0f}]")
    print(f"{'='*65}")
    print(f"  ₹{cap:,.0f} → ₹{end_equity:,.0f}  ({ret:+.1f}%)")
    print(f"  CAGR: {cagr:+.1f}%   MaxDD: {max_dd:.1f}%")
    print(f"  Trades: {total}  ({total/years:.0f}/yr)  Win: {wr:.0f}%  ({int(won)}W/{total-int(won)}L)")
    print(f"  Avg win: ₹{avg_w:,.0f}   Avg loss: ₹{avg_l:,.0f}")
    print(f"  Expectancy: ₹{exp:,.0f}/trade")
    print(f"  Unanimous (5/5): {unani} trades")

    print(f"\n  {'Year':<6} {'T':>4} {'W%':>5}  {'P&L ₹':>12}  Bar")
    for yr, grp in df.groupby("year"):
        n   = len(grp)
        w   = grp["won"].sum()
        pnl = grp["pnl_inr"].sum()
        bar_w = int(abs(pnl) / max(abs(df.groupby("year")["pnl_inr"].sum()).max(), 1) * 18)
        ch = "█" if pnl >= 0 else "░"
        sign = "+" if pnl >= 0 else ""
        print(f"  {yr:<6} {n:>4} {w/n*100:>4.0f}%  {sign}{pnl:>11,.0f}  {ch*bar_w}")

    print(f"\n  Exit breakdown:")
    for reason, grp in df.groupby("exit_reason"):
        n   = len(grp)
        pnl = grp["pnl_inr"].sum()
        wr_ = grp["won"].mean()*100
        print(f"    {reason:<8} {n:>4}  win={wr_:3.0f}%  P&L=₹{pnl:+,.0f}")

    expiry_df = df[df["is_expiry"]]
    pre_df    = df[~df["is_expiry"]]
    if not expiry_df.empty:
        print(f"\n  Expiry day:  {len(expiry_df)} trades  win={expiry_df['won'].mean()*100:.0f}%  "
              f"P&L=₹{expiry_df['pnl_inr'].sum():+,.0f}")
    if not pre_df.empty:
        print(f"  Pre-expiry:  {len(pre_df)} trades  win={pre_df['won'].mean()*100:.0f}%  "
              f"P&L=₹{pre_df['pnl_inr'].sum():+,.0f}")

    return {"label": label, "cagr": cagr, "max_dd": max_dd,
            "win_rate": wr, "exp": exp, "total_return": ret,
            "trades": total}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")

    n5   = load_historical("NIFTY50", "5min")
    n1d  = load_historical("NIFTY50", "1day")
    s5   = load_historical("SENSEX",  "5min")
    s1d  = load_historical("SENSEX",  "1day")

    for df in [n5, n1d, s5, s1d]:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    print(f"  Nifty  5min: {len(n5):,} bars")
    print(f"  Sensex 5min: {len(s5):,} bars\n")

    CAP = 500_000

    all_results = []

    for rr in [2.0, 3.0]:
        print(f"\n{'─'*65}")
        print(f"  TESTING RR = 1:{rr:.0f}")
        print(f"{'─'*65}")

        t1, eq1 = run_smart_options("NIFTY50", n5, n1d, CAP, rr)
        r1 = show_results(t1, eq1, CAP, "NIFTY50 | ORB+Signal | ATM/OTM", rr)
        if r1: all_results.append({**r1, "rr": rr})

        t2, eq2 = run_smart_options("SENSEX", s5, s1d, CAP, rr)
        r2 = show_results(t2, eq2, CAP, "SENSEX  | ORB+Signal | ATM/OTM", rr)
        if r2: all_results.append({**r2, "rr": rr})

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        print(f"\n\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Strategy':<38} {'RR':>4} {'CAGR':>7} {'DD':>6} {'Win%':>6} {'Exp':>9}")
        print(f"  {'─'*68}")
        for r in all_results:
            sign = "+" if r["cagr"] >= 0 else ""
            print(f"  {r['label'][:38]:<38} 1:{r['rr']:.0f}  "
                  f"{sign}{r['cagr']:>5.1f}%  {r['max_dd']:>5.1f}%  "
                  f"{r['win_rate']:>5.0f}%  ₹{r['exp']:>+7,.0f}")

    print(f"\n  Note: Premium-based on Black-Scholes + 20d HV.")
    print(f"  Entry: ORB (first 30 min) breakout confirmed by 3/5 strategy aggregator.")
    print(f"  Expiry day → ATM | Pre-expiry → 1-2 strikes OTM.")
    print(f"  Stop: 60% premium loss. Max entry: 11 AM. Force-close: 3 PM.")


if __name__ == "__main__":
    main()
