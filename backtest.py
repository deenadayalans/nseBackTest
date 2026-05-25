# strategies/backtest.py
# Layer 2: Strategy engine built on backtrader
#
# Two strategies included:
#   1. MomentumStrategy  — EMA crossover + ATR stop loss (trend following)
#   2. MeanReversionStrategy — Bollinger Band squeeze + RSI reversal
#
# Both use ATR-based position sizing and strict stop-loss rules.
# Run the file directly to backtest on synthetic data.

import argparse
import sys
import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.analyzers as btanalyzers
from datetime import datetime


# ── Reusable base strategy with risk controls ─────────────────────────────────

class NiftyBaseStrategy(bt.Strategy):
    """
    Base class for all strategies.
    Handles: ATR-based stop loss, position sizing, trade logging.
    Override: next() with your entry/exit logic.
    """
    params = (
        ("risk_pct",    1.0),    # max % of portfolio to risk per trade
        ("atr_period",  14),
        ("atr_sl_mult", 1.5),    # stop = entry ± (ATR × multiplier)
        ("verbose",     True),
    )

    def __init__(self):
        self.atr     = bt.indicators.ATR(period=self.p.atr_period)
        self.order   = None
        self.stop_price  = None
        self.entry_price = None
        self.trades_log  = []

    def log(self, txt: str):
        if self.p.verbose:
            dt = self.datas[0].datetime.date(0)
            print(f"  [{dt}] {txt}")

    def position_size(self) -> int:
        """
        Risk-based position sizing.
        Units = (portfolio_value × risk_pct) / (ATR × atr_sl_mult)
        This ensures one losing trade never costs more than risk_pct of capital.
        """
        portfolio_val = self.broker.getvalue()
        risk_amount   = portfolio_val * (self.p.risk_pct / 100)
        stop_distance = self.atr[0] * self.p.atr_sl_mult
        if stop_distance == 0:
            return 1
        size = int(risk_amount / stop_distance)
        return max(1, size)

    def notify_order(self, order):
        if order.status == order.Completed:
            side = "BUY " if order.isbuy() else "SELL"
            self.log(f"{side} @ {order.executed.price:.2f}  size={order.executed.size}  "
                     f"commission={order.executed.comm:.2f}")
        elif order.status in (order.Canceled, order.Rejected):
            self.log(f"Order {order.status}")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades_log.append({
                "open_dt":  bt.num2date(trade.dtopen),
                "close_dt": bt.num2date(trade.dtclose),
                "pnl":      round(trade.pnl, 2),
                "pnl_pct":  round(trade.pnl / (trade.price * trade.size) * 100, 2)
                            if trade.price and trade.size else 0.0,
            })
            emoji = "✅" if trade.pnl > 0 else "❌"
            self.log(f"{emoji} Trade closed  PnL={trade.pnl:.2f}")


# ── Strategy 1: EMA Momentum ──────────────────────────────────────────────────

class MomentumStrategy(NiftyBaseStrategy):
    """
    EMA Crossover Momentum with ATR stop.

    Entry conditions (LONG only, index — no short selling spot):
      • Fast EMA (9) crosses above Slow EMA (21)
      • Price is above 50-day EMA (trend filter)
      • RSI > 40 (not oversold entry)
      • Volume above 20-day average

    Exit conditions:
      • Fast EMA crosses below Slow EMA  (signal reversal)
      • Price hits ATR-based stop loss

    Why this works (when it does):
      Nifty tends to trend strongly post-budget, earnings seasons, and
      global risk-on/off episodes. Trend-following captures those runs.
    """
    params = (
        ("fast_period", 9),
        ("slow_period", 21),
        ("trend_period",50),
        ("rsi_period",  14),
        ("vol_period",  20),
    )

    def __init__(self):
        super().__init__()
        self.ema_fast  = bt.indicators.EMA(period=self.p.fast_period)
        self.ema_slow  = bt.indicators.EMA(period=self.p.slow_period)
        self.ema_trend = bt.indicators.EMA(period=self.p.trend_period)
        self.rsi       = bt.indicators.RSI(period=self.p.rsi_period)
        self.vol_sma   = bt.indicators.SMA(self.datas[0].volume, period=self.p.vol_period)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)

    def next(self):
        if self.order:
            return

        close = self.data.close[0]

        if not self.position:
            # ── Entry ────────────────────────────────────────────
            trend_ok  = close > self.ema_trend[0]
            rsi_ok    = self.rsi[0] > 40
            volume_ok = self.datas[0].volume[0] > self.vol_sma[0]
            cross_up  = self.crossover[0] > 0

            if cross_up and trend_ok and rsi_ok and volume_ok:
                size = self.position_size()
                self.stop_price  = close - (self.atr[0] * self.p.atr_sl_mult)
                self.entry_price = close
                self.order = self.buy(size=size)
                self.log(f"BUY signal | EMA9={self.ema_fast[0]:.1f} > EMA21={self.ema_slow[0]:.1f} "
                         f"| RSI={self.rsi[0]:.1f} | Stop={self.stop_price:.1f}")
        else:
            # ── Exit ─────────────────────────────────────────────
            stop_hit    = close <= self.stop_price
            cross_down  = self.crossover[0] < 0

            if stop_hit:
                self.log(f"STOP LOSS hit @ {close:.1f}  (stop was {self.stop_price:.1f})")
                self.order = self.close()
            elif cross_down:
                self.log(f"EMA cross-down exit @ {close:.1f}")
                self.order = self.close()
            else:
                # Trail the stop up (never move it down)
                new_stop = close - (self.atr[0] * self.p.atr_sl_mult)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop


# ── Strategy 2: Mean Reversion (Bollinger + RSI) ──────────────────────────────

class MeanReversionStrategy(NiftyBaseStrategy):
    """
    Bollinger Band mean reversion with RSI confirmation.

    Entry conditions:
      • Price touches or crosses below lower Bollinger Band
      • RSI < 35 (oversold)
      • Price is above 200-day EMA (only trade pullbacks in bull market)

    Exit conditions:
      • Price reaches Bollinger Band midline (take profit)
      • Price hits ATR-based stop loss
      • RSI > 70 (overbought — optional early exit)

    Why this works (when it does):
      Nifty frequently reverts to mean after sharp news-driven selloffs.
      This strategy tries to buy those dips with confirmation.
    """
    params = (
        ("bb_period", 20),
        ("bb_dev",    2.0),
        ("rsi_period",14),
        ("trend_period", 200),
        ("rsi_oversold", 35),
        ("rsi_overbought", 65),
    )

    def __init__(self):
        super().__init__()
        self.bb       = bt.indicators.BollingerBands(period=self.p.bb_period,
                                                     devfactor=self.p.bb_dev)
        self.rsi      = bt.indicators.RSI(period=self.p.rsi_period)
        self.ema_trend= bt.indicators.EMA(period=self.p.trend_period)
        self.target_price = None

    def next(self):
        if self.order:
            return

        close = self.data.close[0]

        if not self.position:
            # ── Entry ────────────────────────────────────────────
            bull_market   = close > self.ema_trend[0]
            below_bb_low  = close <= self.bb.lines.bot[0]
            rsi_oversold  = self.rsi[0] < self.p.rsi_oversold

            if bull_market and below_bb_low and rsi_oversold:
                size = self.position_size()
                self.stop_price   = close - (self.atr[0] * self.p.atr_sl_mult)
                self.entry_price  = close
                self.target_price = self.bb.lines.mid[0]  # target: mean
                self.order = self.buy(size=size)
                self.log(f"MEAN REV BUY | Price={close:.1f} BBLow={self.bb.lines.bot[0]:.1f} "
                         f"RSI={self.rsi[0]:.1f} | Target={self.target_price:.1f}")
        else:
            # ── Exit ─────────────────────────────────────────────
            stop_hit     = close <= self.stop_price
            target_hit   = close >= self.target_price
            rsi_extended = self.rsi[0] > self.p.rsi_overbought

            if stop_hit:
                self.log(f"STOP LOSS @ {close:.1f}")
                self.order = self.close()
            elif target_hit:
                self.log(f"TARGET hit @ {close:.1f}  (entry={self.entry_price:.1f})")
                self.order = self.close()
            elif rsi_extended:
                self.log(f"RSI overbought exit @ {close:.1f}  RSI={self.rsi[0]:.1f}")
                self.order = self.close()


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy_class,
    strategy_params: dict = None,
    initial_capital: float = 500_000.0,   # ₹5 lakh starting capital
    commission_pct: float  = 0.0003,      # 0.03% per trade (Zerodha futures)
) -> dict:
    """
    Run a full backtest and return performance metrics.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        strategy_class: MomentumStrategy or MeanReversionStrategy
        strategy_params: dict of param overrides
        initial_capital: starting cash in ₹
        commission_pct: broker commission rate

    Returns:
        dict with metrics: sharpe, cagr, max_drawdown, win_rate, total_trades
    """
    cerebro = bt.Cerebro()

    # Feed data
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)

    # Add strategy
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)

    # Capital and commission
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission_pct)

    # Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio,   _name="sharpe",
                        riskfreerate=0.065,         # RBI repo rate ~6.5%
                        annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(btanalyzers.Returns,       _name="returns", tann=252)
    cerebro.addanalyzer(btanalyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")

    print(f"\nRunning {strategy_class.__name__}...")
    print(f"  Capital: ₹{initial_capital:,.0f}  |  Commission: {commission_pct*100:.3f}%")
    print(f"  Data: {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    print("-" * 60)

    result = cerebro.run()
    strat  = result[0]

    # Extract metrics
    sharpe_raw = strat.analyzers.sharpe.get_analysis()
    sharpe     = sharpe_raw.get("sharperatio", None)
    
    returns_a  = strat.analyzers.returns.get_analysis()
    cagr_pct   = returns_a.get("rnorm100", 0)
    
    dd         = strat.analyzers.drawdown.get_analysis()
    max_dd_pct = dd.get("max", {}).get("drawdown", 0)
    
    ta         = strat.analyzers.trades.get_analysis()
    total      = ta.get("total", {}).get("closed", 0)
    won        = ta.get("won",   {}).get("total",  0)
    win_rate   = (won / total * 100) if total > 0 else 0

    final_val  = cerebro.broker.getvalue()
    total_return = (final_val - initial_capital) / initial_capital * 100

    metrics = {
        "strategy":      strategy_class.__name__,
        "initial_capital": initial_capital,
        "final_value":   round(final_val, 2),
        "total_return":  round(total_return, 2),
        "cagr_pct":      round(cagr_pct, 2),
        "sharpe":        round(sharpe, 3) if sharpe else None,
        "max_drawdown":  round(max_dd_pct, 2),
        "total_trades":  total,
        "win_rate":      round(win_rate, 2),
        "wins":          won,
        "losses":        total - won,
    }

    _print_metrics(metrics)
    return metrics


def _print_metrics(m: dict):
    print(f"\n{'='*60}")
    print(f"  Strategy:       {m['strategy']}")
    print(f"  Capital:        ₹{m['initial_capital']:>12,.0f}")
    print(f"  Final value:    ₹{m['final_value']:>12,.0f}  ({m['total_return']:+.1f}%)")
    print(f"  CAGR:           {m['cagr_pct']:>+8.2f}%")
    print(f"  Sharpe ratio:   {m['sharpe']}")
    print(f"  Max drawdown:   {m['max_drawdown']:>8.2f}%")
    print(f"  Total trades:   {m['total_trades']}")
    print(f"  Win rate:       {m['win_rate']:>8.2f}%  ({m['wins']}W / {m['losses']}L)")
    print(f"{'='*60}\n")


# ── Synthetic data test (no Kite needed) ──────────────────────────────────────

def make_synthetic_nifty(periods: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic-ish Nifty price data for testing strategies
    without a Kite API connection.
    Uses geometric Brownian motion with a slight upward drift.
    """
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")

    # GBM parameters calibrated loosely to Nifty (annualised)
    mu    = 0.12  # 12% annual drift
    sigma = 0.18  # 18% annual vol
    dt    = 1/252

    log_returns = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.randn(periods)
    price       = 12000 * np.exp(np.cumsum(log_returns))

    daily_vol = sigma * np.sqrt(dt) * price
    opens  = price + np.random.randn(periods) * daily_vol * 0.3
    highs  = price + np.abs(np.random.randn(periods)) * daily_vol
    lows   = price - np.abs(np.random.randn(periods)) * daily_vol
    vols   = np.random.randint(50_000_000, 200_000_000, periods).astype(float)

    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  price,
        "volume": vols,
    }, index=dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty Strategy Backtester")
    parser.add_argument(
        "--real", action="store_true",
        help="Use real Nifty data from ./data/historical/ (run download_data.py first)"
    )
    parser.add_argument(
        "--interval", default="1day",
        help="Bar interval for real data: 1day | 1hr | 15min  (default: 1day)"
    )
    parser.add_argument(
        "--capital", type=int, default=500_000,
        help="Starting capital in INR (default: 500000)"
    )
    args = parser.parse_args()

    if args.real:
        from kite_data import load_historical, add_features
        print(f"Nifty Strategy Backtester — REAL DATA ({args.interval})\n" + "="*60)
        try:
            df = load_historical("NIFTY50", args.interval)
            df = add_features(df)
            print(f"Loaded {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Run 'python download_data.py' first to fetch real data.")
            sys.exit(1)
    else:
        print("Nifty Strategy Backtester — SYNTHETIC DATA\n" + "="*60)
        print("(Run with --real to backtest on actual Nifty prices)\n")
        df = make_synthetic_nifty(periods=1000)

    m1 = run_backtest(df, MomentumStrategy,      initial_capital=args.capital)
    m2 = run_backtest(df, MeanReversionStrategy, initial_capital=args.capital)

    print("\nComparison:")
    print(f"  {'Metric':<20} {'Momentum':>12} {'MeanReversion':>14}")
    print(f"  {'-'*46}")
    for key in ["total_return", "cagr_pct", "sharpe", "max_drawdown", "win_rate", "total_trades"]:
        v1 = m1.get(key)
        v2 = m2.get(key)
        print(f"  {key:<20} {str(v1):>12} {str(v2):>14}")
