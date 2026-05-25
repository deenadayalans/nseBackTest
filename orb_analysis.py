#!/usr/bin/env python3
"""
orb_analysis.py — Deep-dive on the winning ORB (Long-only, range ≤ 0.5%) strategy.

Outputs
-------
  1. Trade-by-trade log
  2. Yearly breakdown
  3. Monthly P&L heatmap
  4. Distribution stats (best/worst trades, streaks)

Run
---
    python orb_analysis.py
"""

import sys
from datetime import time
from collections import defaultdict

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np

from kite_data import load_historical, add_features
from intraday_backtest import IntradayBase

MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 15)


# ── Instrumented ORB (long-only, records full trade detail) ───────────────────

LOT_SIZE = 75   # Nifty Futures lot size (NSE standard)


class ORBDetailed(IntradayBase):
    params = (
        ("max_range_pct", 0.5),
        ("rr_target",     2.0),
    )

    def __init__(self):
        super().__init__()
        self._or_high      = None
        self._or_low       = None
        self._or_set       = False
        self._traded_today = False
        self._last_date    = None
        self._entry_price  = None
        self._or_range     = None
        self._entry_lots   = 0     # lots traded (tracked at entry)
        self.detailed_log  = []

    def _lots(self, stop_distance: float) -> int:
        """
        Position size in whole Nifty Futures lots.
        Minimum 1 lot. Risk ~0.5% of equity per trade.
        """
        equity   = self.broker.getvalue()
        risk_amt = equity * (self.p.risk_pct / 100)
        if stop_distance <= 0:
            return LOT_SIZE
        raw_units = risk_amt / stop_distance
        lots = max(1, int(raw_units / LOT_SIZE))
        return lots * LOT_SIZE   # always a whole-lot quantity

    def next(self):
        if self.order:
            return

        dt           = self.data.datetime.datetime(0)
        current_date = dt.date()
        current_time = dt.time()

        if current_date != self._last_date:
            self._last_date    = current_date
            self._or_high      = None
            self._or_low       = None
            self._or_set       = False
            self._traded_today = False

        if current_time == MARKET_OPEN:
            self._or_high = self.data.high[0]
            self._or_low  = self.data.low[0]
            return

        if self._or_high is not None and not self._or_set:
            self._or_set = True

        if current_time >= MARKET_CLOSE and self.position:
            self.order = self.close()
            return

        if not self._or_set or self._traded_today:
            return

        close = self.data.close[0]
        rng   = self._or_high - self._or_low
        if rng <= 0:
            return
        if (rng / self._or_low * 100) > self.p.max_range_pct:
            return

        if not self.position:
            if close > self._or_high:
                stop              = self._or_low
                size              = self._lots(close - stop)
                self._entry_lots  = size
                self._entry_price = close
                self._or_range    = rng
                self.stop_price   = stop
                self.target_price = close + rng * self.p.rr_target
                self.order = self.buy(size=size)
                self._traded_today = True
        else:
            if self.position.size > 0:
                if close <= self.stop_price or close >= self.target_price:
                    self.order = self.close()

    def notify_trade(self, trade):
        if trade.isclosed:
            lots     = self._entry_lots // LOT_SIZE
            entry_dt = bt.num2date(trade.dtopen)
            exit_dt  = bt.num2date(trade.dtclose)

            # exit price from P&L
            if self._entry_lots:
                exit_px = trade.price + (trade.pnl + trade.commission) / self._entry_lots
            else:
                exit_px = trade.price

            pts_moved = round(exit_px - trade.price, 1)

            if trade.pnl > 0:
                reason = "TARGET"
            elif (exit_dt.hour * 60 + exit_dt.minute) >= (15 * 60 + 10):
                reason = "EOD"
            else:
                reason = "STOP"

            cost    = trade.price * self._entry_lots
            pnl_pct = round(trade.pnl / cost * 100, 3) if cost else 0.0

            self.detailed_log.append({
                "date":      entry_dt.strftime("%Y-%m-%d"),
                "entry_dt":  entry_dt.strftime("%H:%M"),
                "exit_dt":   exit_dt.strftime("%H:%M"),
                "entry_px":  round(trade.price, 1),
                "exit_px":   round(exit_px, 1),
                "pts":       pts_moved,
                "lots":      lots,
                "or_range":  round(self._or_range, 1) if self._or_range else 0,
                "pnl":       round(trade.pnl, 2),
                "pnl_pct":   pnl_pct,
                "reason":    reason,
                "equity":    round(self.broker.getvalue(), 0),
            })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _streak(results: list[bool]) -> tuple[int, int]:
    """Return (max_win_streak, max_loss_streak)."""
    max_w = max_l = cur_w = cur_l = 0
    for w in results:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _bar(value: float, max_val: float, width: int = 20, pos_char="█", neg_char="░") -> str:
    if max_val == 0:
        return " " * width
    ratio = value / max_val
    filled = int(abs(ratio) * width)
    char = pos_char if value >= 0 else neg_char
    return (char * filled).ljust(width)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading 15-min Nifty 50 data...")
    try:
        df = load_historical("NIFTY50", "15min")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    df = add_features(df)
    print(f"Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    cerebro = bt.Cerebro(stdstats=False)
    feed = bt.feeds.PandasData(
        dataname=df, datetime=None,
        open="open", high="high", low="low",
        close="close", volume="volume", openinterest=-1,
    )
    cerebro.adddata(feed)
    cerebro.addstrategy(ORBDetailed, verbose=False)
    cerebro.broker.setcash(500_000)

    # Futures mode: broker only locks margin per unit, not full notional.
    # Nifty Futures margin ≈ ₹70,000 / lot → ₹933/unit.
    # Commission: ~0.013% per side on notional ≈ ₹150/order on 1 lot
    # (Zerodha ₹20 brokerage + STT 0.01% sell-side + exchange fees + GST)
    cerebro.broker.setcommission(
        commission=0.00013,        # ~0.013% per side on notional
        margin=70_000 / LOT_SIZE,  # ₹933 margin per unit → ₹70K per lot
        mult=1.0,
    )

    cerebro.addanalyzer(btanalyzers.SharpeRatio,
                        _name="sharpe", riskfreerate=0.065, annualize=True,
                        timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")

    results  = cerebro.run()
    strat    = results[0]
    end_val  = cerebro.broker.getvalue()
    log      = strat.detailed_log

    if not log:
        print("No trades recorded.")
        return

    trades_df = pd.DataFrame(log)
    trades_df["date"]  = pd.to_datetime(trades_df["date"])
    trades_df["year"]  = trades_df["date"].dt.year
    trades_df["month"] = trades_df["date"].dt.month
    trades_df["won"]   = trades_df["pnl"] > 0

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = len(trades_df)
    won     = trades_df["won"].sum()
    wr      = won / total * 100
    avg_win  = trades_df.loc[trades_df["won"],  "pnl"].mean()
    avg_loss = trades_df.loc[~trades_df["won"], "pnl"].mean()
    expectancy = (wr/100 * avg_win) + ((1 - wr/100) * avg_loss)
    total_ret  = (end_val - 500_000) / 500_000 * 100
    years      = (trades_df["date"].max() - trades_df["date"].min()).days / 365.25
    cagr       = ((end_val / 500_000) ** (1 / years) - 1) * 100 if years > 0 else 0
    max_dd     = strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0)
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()["sharperatio"] or 0
    except Exception:
        sharpe = 0
    max_w, max_l = _streak(trades_df["won"].tolist())

    print("=" * 65)
    print("  ORB STRATEGY — Long Only | Range ≤ 0.5% | Target 2× Range")
    print("=" * 65)
    print(f"  Period       : {trades_df['date'].min().date()} → {trades_df['date'].max().date()}")
    print(f"  Capital      : ₹5,00,000  →  ₹{end_val:,.0f}")
    print(f"  Total return : {total_ret:+.2f}%")
    print(f"  CAGR         : {cagr:+.2f}%")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Max drawdown : {max_dd:.2f}%")
    print(f"  Trades       : {total}  ({total/years:.1f}/year)")
    print(f"  Win rate     : {wr:.1f}%  ({int(won)}W / {total - int(won)}L)")
    print(f"  Avg win      : ₹{avg_win:,.2f}")
    print(f"  Avg loss     : ₹{avg_loss:,.2f}")
    print(f"  Expectancy   : ₹{expectancy:,.2f} per trade")
    print(f"  Best trade   : ₹{trades_df['pnl'].max():,.2f}  ({trades_df.loc[trades_df['pnl'].idxmax(), 'date'].strftime('%Y-%m-%d')})")
    print(f"  Worst trade  : ₹{trades_df['pnl'].min():,.2f}  ({trades_df.loc[trades_df['pnl'].idxmin(), 'date'].strftime('%Y-%m-%d')})")
    print(f"  Max win streak  : {max_w}")
    print(f"  Max loss streak : {max_l}")
    avg_lots = trades_df["lots"].mean()
    print(f"  Avg lots/trade  : {avg_lots:.1f}  (₹{avg_lots*70_000:,.0f} avg margin deployed)")

    # ── Yearly breakdown ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  YEARLY BREAKDOWN")
    print(f"{'─'*65}")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>6} {'P&L ₹':>10} {'Ret%':>7}  Bar")
    print(f"  {'─'*60}")

    yearly_pnl = trades_df.groupby("year")["pnl"].sum()
    max_abs_pnl = yearly_pnl.abs().max()

    for year, grp in trades_df.groupby("year"):
        n   = len(grp)
        w   = grp["won"].sum()
        wr_ = w / n * 100
        pnl = grp["pnl"].sum()
        # Approximate % return on starting equity that year
        start_eq = grp["equity"].iloc[0] - grp["pnl"].iloc[0]  # rough
        ret_pct  = pnl / 500_000 * 100
        bar = _bar(pnl, max_abs_pnl)
        sign = "+" if pnl >= 0 else ""
        print(f"  {year:<6} {n:>7} {wr_:>5.0f}%  {sign}{pnl:>9,.0f}  {sign}{ret_pct:>5.1f}%  {bar}")

    # ── Monthly heatmap ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  MONTHLY P&L HEATMAP  (₹)")
    print(f"{'─'*65}")

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    years_list = sorted(trades_df["year"].unique())

    header = f"  {'':6}" + "".join(f"{m:>7}" for m in months)
    print(header)

    for yr in years_list:
        row = f"  {yr:<6}"
        for mo in range(1, 13):
            mask = (trades_df["year"] == yr) & (trades_df["month"] == mo)
            pnl  = trades_df.loc[mask, "pnl"].sum()
            n    = mask.sum()
            if n == 0:
                row += f"  {'---':>5}"
            else:
                sign = "+" if pnl >= 0 else ""
                row += f"  {sign}{pnl:>4,.0f}"[: 7]
        print(row)

    # ── Trade-by-trade log ────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  TRADE LOG  (1 lot = 75 units of Nifty Futures)")
    print(f"{'─'*80}")
    print(f"  {'Date':<12} {'In':>5} {'Out':>5} {'Entry':>8} {'Exit':>8} "
          f"{'Pts':>6} {'Lots':>5} {'Rng':>5} {'P&L ₹':>10}  {'Reason':<7}  {'Equity':>10}")
    print(f"  {'─'*90}")

    for _, r in trades_df.iterrows():
        icon  = "✅" if r["pnl"] >= 0 else "❌"
        pnl_s = f"{r['pnl']:+,.0f}"
        pts_s = f"{r['pts']:+.0f}"
        print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {r['entry_dt']:>5} {r['exit_dt']:>5} "
              f"{r['entry_px']:>8.1f} {r['exit_px']:>8.1f} "
              f"{pts_s:>6} {r['lots']:>5} {r['or_range']:>5.0f} "
              f"{pnl_s:>10}  {icon} {r['reason']:<5}  ₹{r['equity']:>9,.0f}")

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  EXIT BREAKDOWN")
    print(f"{'─'*65}")
    for reason, grp in trades_df.groupby("reason"):
        n   = len(grp)
        pnl = grp["pnl"].sum()
        wr_ = grp["won"].mean() * 100
        print(f"  {reason:<10}  {n:>4} trades  win={wr_:.0f}%  total P&L=₹{pnl:+,.0f}")


if __name__ == "__main__":
    main()
