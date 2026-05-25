#!/usr/bin/env python3
"""
intraday_backtest.py — Three intraday strategies on Nifty 50 (15-min bars)

Strategies
----------
1. ORB   — Opening Range Breakout: first 15-min bar sets the high/low range;
           trade the breakout with a stop at the opposite side.
2. VWAP  — VWAP Reversion: fade moves > 1.5×ATR away from the daily VWAP,
           target a reversion back to VWAP.
3. EMA   — EMA Pullback: 9/21 EMA trend filter; enter on pullbacks to the 9 EMA.

All strategies
--------------
  • Long AND short (assumes Nifty Futures execution — spot is just the price feed)
  • Force-close any open position at 15:15 IST (last 15-min bar open)
  • No overnight positions
  • Commission: 0.03 % per side (Zerodha intraday)
  • Risk per trade: 0.5 % of portfolio

Run
---
    python intraday_backtest.py
"""

import sys
from datetime import time

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np

from kite_data import load_historical, add_features


# ── Constants ────────────────────────────────────────────────────────────────

MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 15)   # force-close at the last 15-min bar
ORB_END      = time(9, 30)    # opening range = first bar only (9:15–9:30)


# ── Shared base ──────────────────────────────────────────────────────────────

class IntradayBase(bt.Strategy):
    """
    Common intraday plumbing:
      - ATR-based position sizing (risk 0.5 % of equity per trade)
      - Force-close at 15:15
      - Trade logging
    """
    params = (
        ("risk_pct",    0.5),    # % of equity to risk per trade
        ("atr_period",  14),
        ("verbose",     True),
    )

    def __init__(self):
        self.atr        = bt.indicators.ATR(period=self.p.atr_period)
        self.order      = None
        self.stop_price = None
        self.trades_log = []

    def log(self, txt: str):
        if self.p.verbose:
            dt = self.data.datetime.datetime(0).strftime("%Y-%m-%d %H:%M")
            print(f"  [{dt}] {txt}")

    def position_size(self, stop_distance: float) -> int:
        if stop_distance <= 0:
            return 1
        equity    = self.broker.getvalue()
        risk_amt  = equity * (self.p.risk_pct / 100)
        size      = int(risk_amt / stop_distance)
        return max(1, size)

    def _force_close_if_needed(self) -> bool:
        """Return True (and close) if it's time to flatten."""
        if self.data.datetime.time(0) >= MARKET_CLOSE and self.position:
            self.log("Force-close EOD")
            self.close()
            return True
        return False

    def notify_order(self, order):
        if order.status == order.Completed:
            side = "BUY " if order.isbuy() else "SELL"
            self.log(f"{side} exec @ {order.executed.price:.2f}  size={abs(order.executed.size)}")
        elif order.status in (order.Canceled, order.Rejected, order.Margin):
            self.log(f"Order {order.getstatusname()}")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            cost = trade.price * abs(trade.size)
            pnl_pct = round(trade.pnl / cost * 100, 2) if cost else 0.0
            self.trades_log.append({
                "open_dt":  trade.dtopen,
                "close_dt": trade.dtclose,
                "pnl":      round(trade.pnl, 2),
                "pnl_pct":  pnl_pct,
                "side":     "long" if trade.size > 0 else "short",
            })
            emoji = "✅" if trade.pnl > 0 else "❌"
            self.log(f"{emoji} Closed  PnL={trade.pnl:.2f}  ({pnl_pct:+.2f}%)")


# ── Strategy 1: Opening Range Breakout ───────────────────────────────────────

class ORBStrategy(IntradayBase):
    """
    Opening Range Breakout (ORB)
    ----------------------------
    • The first 15-min candle (9:15–9:30) defines the opening range.
    • A close above the range high   → long  with stop at range low.
    • A close below the range low    → short with stop at range high.
    • Target = 2 × range.
    • One trade per day (first signal only).
    • Force-close at 15:15.
    • Filter: only trade when the range is < max_range_pct of price
      (tight ranges produce more explosive, cleaner breakouts).
    """
    params = (
        ("rr_target",      2.0),   # reward-to-risk ratio for target
        ("max_range_pct",  0.4),   # skip wide/choppy opens (% of price)
    )

    def __init__(self):
        super().__init__()
        self._or_high    = None
        self._or_low     = None
        self._or_set     = False
        self._traded_today = False
        self._last_date  = None

    def next(self):
        if self.order:
            return

        current_dt   = self.data.datetime.datetime(0)
        current_date = current_dt.date()
        current_time = current_dt.time()

        # New day → reset state
        if current_date != self._last_date:
            self._last_date    = current_date
            self._or_high      = None
            self._or_low       = None
            self._or_set       = False
            self._traded_today = False

        # Capture opening range bar (9:15)
        if current_time == MARKET_OPEN:
            self._or_high = self.data.high[0]
            self._or_low  = self.data.low[0]
            return

        # Range is set on the bar after MARKET_OPEN
        if self._or_high is not None and not self._or_set:
            self._or_set = True

        if self._force_close_if_needed():
            return

        if not self._or_set or self._traded_today:
            return

        close = self.data.close[0]
        rng   = self._or_high - self._or_low
        if rng <= 0:
            return

        # Skip wide/noisy opens — tight ranges give cleaner breakouts
        range_pct = rng / self._or_low * 100
        if range_pct > self.p.max_range_pct:
            return

        if not self.position:
            if close > self._or_high:
                stop = self._or_low
                size = self.position_size(close - stop)
                self.log(f"ORB LONG  break={close:.2f}  range=[{self._or_low:.1f},{self._or_high:.1f}]")
                self.order = self.buy(size=size)
                self.stop_price   = stop
                self.target_price = close + rng * self.p.rr_target
                self._traded_today = True

            elif close < self._or_low:
                stop = self._or_high
                size = self.position_size(stop - close)
                self.log(f"ORB SHORT break={close:.2f}  range=[{self._or_low:.1f},{self._or_high:.1f}]")
                self.order = self.sell(size=size)
                self.stop_price   = stop
                self.target_price = close - rng * self.p.rr_target
                self._traded_today = True

        else:
            # Manage open position
            if self.position.size > 0:   # long
                if close <= self.stop_price or close >= self.target_price:
                    self.order = self.close()
            else:                         # short
                if close >= self.stop_price or close <= self.target_price:
                    self.order = self.close()


# ── Strategy 2: VWAP Reversion ───────────────────────────────────────────────

class VWAPReversionStrategy(IntradayBase):
    """
    VWAP Reversion
    --------------
    • Computes cumulative VWAP (typical price mean) resetting each day.
    • Long  when price < VWAP − 1.5×ATR  and  RSI < 42  (oversold dip)
    • Short when price > VWAP + 1.5×ATR  and  RSI > 58  (overbought spike)
    • Stop : 1×ATR from entry.
    • Target: VWAP itself (mean reversion).
    • Force-close at 15:15. No new entries after 14:45.
    """
    params = (
        ("vwap_band",    1.5),   # ATR multiples away from VWAP to trigger
        ("rsi_period",   14),
        ("rsi_oversold", 42),
        ("rsi_overbought", 58),
    )

    def __init__(self):
        super().__init__()
        self.rsi         = bt.indicators.RSI(period=self.p.rsi_period)
        self._vwap_sum   = 0.0
        self._bar_count  = 0
        self._last_date  = None
        self._vwap       = None

    def _update_vwap(self):
        date = self.data.datetime.date(0)
        typ  = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
        if date != self._last_date:
            self._vwap_sum  = typ
            self._bar_count = 1
            self._last_date = date
        else:
            self._vwap_sum  += typ
            self._bar_count += 1
        self._vwap = self._vwap_sum / self._bar_count

    def next(self):
        if self.order:
            return

        self._update_vwap()
        vwap         = self._vwap
        current_time = self.data.datetime.time(0)

        if self._force_close_if_needed():
            return

        # No new entries in the opening range or too close to close
        if current_time <= ORB_END or current_time >= time(14, 45):
            return

        close = self.data.close[0]
        atr   = self.atr[0]
        rsi   = self.rsi[0]
        band  = self.p.vwap_band * atr

        if not self.position:
            if close < vwap - band and rsi < self.p.rsi_oversold:
                stop = close - atr
                size = self.position_size(atr)
                self.log(f"VWAP LONG  price={close:.2f}  vwap={vwap:.2f}  RSI={rsi:.1f}")
                self.order = self.buy(size=size)
                self.stop_price   = stop
                self.target_price = vwap

            elif close > vwap + band and rsi > self.p.rsi_overbought:
                stop = close + atr
                size = self.position_size(atr)
                self.log(f"VWAP SHORT price={close:.2f}  vwap={vwap:.2f}  RSI={rsi:.1f}")
                self.order = self.sell(size=size)
                self.stop_price   = stop
                self.target_price = vwap

        else:
            if self.position.size > 0:   # long
                if close <= self.stop_price or close >= self.target_price:
                    self.order = self.close()
            else:                         # short
                if close >= self.stop_price or close <= self.target_price:
                    self.order = self.close()


# ── Strategy 3: EMA Pullback ─────────────────────────────────────────────────

class EMAPullbackStrategy(IntradayBase):
    """
    EMA Pullback (Trend Following)
    ------------------------------
    • Trend defined by 9 EMA vs 21 EMA on 15-min bars.
    • Uptrend  (9 > 21): wait for price to pull back within 0.5×ATR of the 9 EMA, then long.
    • Downtrend(9 < 21): wait for price to bounce within 0.5×ATR of the 9 EMA, then short.
    • Stop : 1.5×ATR from entry.
    • Target: 2.5×ATR from entry (R:R ≈ 1.67).
    • Force-close at 15:15. One trade per day.
    """
    params = (
        ("fast",       9),
        ("slow",      21),
        ("touch_atr", 0.5),   # how close price must be to EMA to count as "pullback"
        ("stop_atr",  1.5),
        ("target_atr",2.5),
        ("rsi_period", 14),
    )

    def __init__(self):
        super().__init__()
        self.ema_fast  = bt.indicators.EMA(period=self.p.fast)
        self.ema_slow  = bt.indicators.EMA(period=self.p.slow)
        self.rsi       = bt.indicators.RSI(period=self.p.rsi_period)
        self._traded_today = False
        self._last_date    = None

    def next(self):
        if self.order:
            return

        current_dt   = self.data.datetime.datetime(0)
        current_date = current_dt.date()
        current_time = current_dt.time()

        if current_date != self._last_date:
            self._last_date    = current_date
            self._traded_today = False

        if self._force_close_if_needed():
            return

        if current_time <= ORB_END or current_time >= time(14, 45):
            return

        close    = self.data.close[0]
        atr      = self.atr[0]
        fast_ema = self.ema_fast[0]
        slow_ema = self.ema_slow[0]
        rsi      = self.rsi[0]
        touch    = self.p.touch_atr * atr

        uptrend   = fast_ema > slow_ema
        downtrend = fast_ema < slow_ema

        if not self.position and not self._traded_today:
            if uptrend and abs(close - fast_ema) <= touch and 40 < rsi < 65:
                stop = close - self.p.stop_atr * atr
                size = self.position_size(self.p.stop_atr * atr)
                self.log(f"EMA LONG   price={close:.2f}  ema9={fast_ema:.2f}  RSI={rsi:.1f}")
                self.order = self.buy(size=size)
                self.stop_price   = stop
                self.target_price = close + self.p.target_atr * atr
                self._traded_today = True

            elif downtrend and abs(close - fast_ema) <= touch and 35 < rsi < 60:
                stop = close + self.p.stop_atr * atr
                size = self.position_size(self.p.stop_atr * atr)
                self.log(f"EMA SHORT  price={close:.2f}  ema9={fast_ema:.2f}  RSI={rsi:.1f}")
                self.order = self.sell(size=size)
                self.stop_price   = stop
                self.target_price = close - self.p.target_atr * atr
                self._traded_today = True

        elif self.position:
            if self.position.size > 0:   # long
                if close <= self.stop_price or close >= self.target_price:
                    self.order = self.close()
            else:                         # short
                if close >= self.stop_price or close <= self.target_price:
                    self.order = self.close()


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_intraday_backtest(
    df: pd.DataFrame,
    strategy_cls,
    initial_capital: float = 500_000,
    commission: float = 0.0003,   # 0.03% per side — Zerodha intraday
    verbose: bool = False,
) -> dict:
    cerebro = bt.Cerebro(stdstats=False)

    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,   # use index
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )
    cerebro.adddata(data)

    cerebro.addstrategy(strategy_cls, verbose=verbose)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addanalyzer(btanalyzers.SharpeRatio,
                        _name="sharpe", riskfreerate=0.065,
                        annualize=True, timeframe=bt.TimeFrame.Minutes,
                        compression=15)
    cerebro.addanalyzer(btanalyzers.DrawDown,      _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(btanalyzers.Returns,       _name="returns")

    name = strategy_cls.__name__.replace("Strategy", "")
    print(f"\n{'='*60}")
    print(f"  Strategy : {name}")
    print(f"  Capital  : ₹{initial_capital:,.0f}")
    print(f"  Bars     : {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"{'='*60}")

    results  = cerebro.run()
    strat    = results[0]
    end_val  = cerebro.broker.getvalue()

    total_return = round((end_val - initial_capital) / initial_capital * 100, 2)
    years        = (df.index[-1] - df.index[0]).days / 365.25
    cagr         = round(((end_val / initial_capital) ** (1 / years) - 1) * 100, 2) if years > 0 else 0

    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("closed", 0)
    won          = ta.get("won",   {}).get("total", 0)
    lost         = ta.get("lost",  {}).get("total", 0)
    win_rate     = round(won / total_trades * 100, 1) if total_trades else 0

    avg_win  = round(ta.get("won",  {}).get("pnl", {}).get("average", 0), 2)
    avg_loss = round(ta.get("lost", {}).get("pnl", {}).get("average", 0), 2)
    expectancy = round((win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss), 2)

    try:
        sharpe = round(strat.analyzers.sharpe.get_analysis()["sharperatio"] or 0, 2)
    except Exception:
        sharpe = None

    max_dd = round(strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0), 2)

    # Trades per day
    trading_days = len(set(df.index.date))
    trades_per_day = round(total_trades / trading_days, 2) if trading_days else 0

    metrics = {
        "total_return_pct": total_return,
        "cagr_pct":         cagr,
        "sharpe":           sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades":     total_trades,
        "win_rate_pct":     win_rate,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "expectancy":       expectancy,
        "trades_per_day":   trades_per_day,
        "end_value":        round(end_val, 2),
    }

    print(f"\n  Total return : {total_return:+.2f}%")
    print(f"  CAGR         : {cagr:+.2f}%")
    print(f"  Sharpe       : {sharpe}")
    print(f"  Max drawdown : {max_dd:.2f}%")
    print(f"  Trades       : {total_trades}  ({trades_per_day}/day avg)")
    print(f"  Win rate     : {win_rate}%  ({won}W / {lost}L)")
    print(f"  Avg win      : ₹{avg_win:,.2f}")
    print(f"  Avg loss     : ₹{avg_loss:,.2f}")
    print(f"  Expectancy   : ₹{expectancy:,.2f} per trade")
    print(f"  End capital  : ₹{end_val:,.0f}")

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading 15-min Nifty 50 data...")
    try:
        df = load_historical("NIFTY50", "15min")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 'python download_data.py' first.")
        sys.exit(1)

    df = add_features(df)
    print(f"Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    capital = 500_000

    r_orb  = run_intraday_backtest(df, ORBStrategy,           capital, verbose=False)
    r_vwap = run_intraday_backtest(df, VWAPReversionStrategy, capital, verbose=False)
    r_ema  = run_intraday_backtest(df, EMAPullbackStrategy,   capital, verbose=False)

    # ── Comparison table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Metric':<22} {'ORB':>12} {'VWAP Rev':>12} {'EMA Pull':>12}"
    print(header)
    print(f"  {'-'*64}")

    rows = [
        ("Total return %",   "total_return_pct",  True),
        ("CAGR %",           "cagr_pct",           True),
        ("Sharpe",           "sharpe",             False),
        ("Max drawdown %",   "max_drawdown_pct",   False),
        ("Trades",           "total_trades",       False),
        ("Trades/day",       "trades_per_day",     False),
        ("Win rate %",       "win_rate_pct",       False),
        ("Avg win ₹",        "avg_win",            True),
        ("Avg loss ₹",       "avg_loss",           False),
        ("Expectancy ₹",     "expectancy",         True),
    ]
    for label, key, higher_is_better in rows:
        vals = [r_orb.get(key), r_vwap.get(key), r_ema.get(key)]
        formatted = []
        for v in vals:
            if v is None:
                formatted.append("N/A")
            elif isinstance(v, float):
                formatted.append(f"{v:>+.2f}" if higher_is_better else f"{v:>.2f}")
            else:
                formatted.append(str(v))
        print(f"  {label:<22} {formatted[0]:>12} {formatted[1]:>12} {formatted[2]:>12}")

    print(f"\n  Note: Execution via Nifty Futures (lot=75). Spot used as price feed only.")
    print(f"        Slippage not modelled — real results will be slightly lower.")
