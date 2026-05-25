# execution/paper_trader.py
# Layer 4: Paper trading engine
#
# Simulates live trading WITHOUT real money.
# Combines: strategy signals + LLM confirmation + risk management
#
# Run this for weeks/months on live data BEFORE touching real capital.
# Only move to live execution when you have:
#   • Win rate > 50% in paper trading
#   • Sharpe ratio > 1.0
#   • At least 30 closed trades
#   • Max drawdown you're genuinely comfortable with
#
# To go live later: replace _paper_execute() with kite.place_order()

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    id:           int
    symbol:       str
    direction:    str           # "LONG" (only — no shorting spot for now)
    entry_price:  float
    entry_time:   str
    stop_loss:    float
    target:       float
    quantity:     int
    strategy:     str
    llm_action:   str           # what the LLM said
    llm_confidence: float
    
    # Filled after close
    exit_price:   Optional[float] = None
    exit_time:    Optional[str]   = None
    exit_reason:  Optional[str]   = None   # "STOP", "TARGET", "SIGNAL", "MANUAL"
    pnl:          Optional[float] = None
    status:       str = "OPEN"


# ── Risk manager ──────────────────────────────────────────────────────────────

class RiskManager:
    """
    Hard rules that CANNOT be overridden — not even by a high-confidence LLM signal.
    This is the last line of defence.
    """
    def __init__(
        self,
        capital: float,
        max_risk_pct: float = 1.0,     # max % of capital at risk per trade
        max_open_trades: int = 1,       # keep to 1 for Nifty — it's volatile
        min_llm_confidence: float = 0.65,  # ignore signals below this
        min_strategy_llm_agree: bool = True,  # strategy + LLM must agree
    ):
        self.capital              = capital
        self.max_risk_pct         = max_risk_pct
        self.max_open_trades      = max_open_trades
        self.min_llm_confidence   = min_llm_confidence
        self.min_strategy_llm_agree = min_strategy_llm_agree

    def position_size(self, entry: float, stop: float) -> int:
        """
        Units = (capital × risk%) / (entry - stop)
        e.g.: ₹5L × 1% / (20000 - 19700) = ₹5000 / 300 = 16 units
        """
        risk_amount   = self.capital * (self.max_risk_pct / 100)
        stop_distance = abs(entry - stop)
        if stop_distance < 1:
            return 0
        return max(1, int(risk_amount / stop_distance))

    def approve_trade(
        self,
        strategy_signal: str,
        llm_signal,         # TradingSignal object
        open_trade_count: int,
        current_price: float,
    ) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str)
        All conditions must pass for a trade to be placed.
        """
        # Rule 1: Max open trades
        if open_trade_count >= self.max_open_trades:
            return False, f"Max open trades reached ({self.max_open_trades})"
        
        # Rule 2: LLM confidence threshold
        if llm_signal.confidence < self.min_llm_confidence:
            return False, f"LLM confidence too low ({llm_signal.confidence:.0%} < {self.min_llm_confidence:.0%})"
        
        # Rule 3: Strategy + LLM must agree
        if self.min_strategy_llm_agree:
            if strategy_signal == "BUY" and llm_signal.action != "BUY":
                return False, f"Strategy says BUY but LLM says {llm_signal.action}"
            if strategy_signal == "SELL" and llm_signal.action != "SELL":
                return False, f"Strategy says SELL but LLM says {llm_signal.action}"
        
        # Rule 4: LLM risk level
        if llm_signal.risk_level == "HIGH":
            return False, "LLM flagged HIGH risk — skipping trade"
        
        # Rule 5: Stop loss must be set
        if llm_signal.suggested_stop <= 0:
            return False, "No valid stop loss from LLM"
        
        return True, "All checks passed"


# ── Trade database ────────────────────────────────────────────────────────────

class TradeDB:
    """SQLite store for all paper trades."""
    
    def __init__(self, db_path: str = "./data/paper_trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT,
                direction       TEXT,
                entry_price     REAL,
                entry_time      TEXT,
                stop_loss       REAL,
                target          REAL,
                quantity        INTEGER,
                strategy        TEXT,
                llm_action      TEXT,
                llm_confidence  REAL,
                exit_price      REAL,
                exit_time       TEXT,
                exit_reason     TEXT,
                pnl             REAL,
                status          TEXT DEFAULT 'OPEN'
            )
        """)
        self.conn.commit()

    def save_trade(self, trade: Trade) -> int:
        row = asdict(trade)
        row.pop("id")
        cols   = ", ".join(row.keys())
        values = tuple(row.values())
        placeholders = ", ".join("?" * len(row))
        cur = self.conn.execute(
            f"INSERT INTO trades ({cols}) VALUES ({placeholders})", values
        )
        self.conn.commit()
        return cur.lastrowid

    def close_trade(self, trade_id: int, exit_price: float,
                     exit_reason: str, pnl: float):
        self.conn.execute("""
            UPDATE trades
            SET exit_price=?, exit_time=?, exit_reason=?, pnl=?, status='CLOSED'
            WHERE id=?
        """, (exit_price, datetime.now().isoformat(), exit_reason, pnl, trade_id))
        self.conn.commit()

    def get_open_trades(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM trades WHERE status='OPEN'")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def performance_summary(self) -> dict:
        cur = self.conn.execute("""
            SELECT COUNT(*) total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) wins,
                   SUM(pnl) total_pnl,
                   AVG(pnl) avg_pnl,
                   MIN(pnl) worst_trade,
                   MAX(pnl) best_trade
            FROM trades WHERE status='CLOSED'
        """)
        row = cur.fetchone()
        total, wins, total_pnl, avg_pnl, worst, best = row
        if not total or total == 0:
            return {"message": "No closed trades yet"}
        return {
            "total_trades": total,
            "wins":         wins,
            "losses":       total - wins,
            "win_rate":     f"{wins/total*100:.1f}%",
            "total_pnl":    f"₹{total_pnl:,.2f}",
            "avg_pnl":      f"₹{avg_pnl:,.2f}",
            "best_trade":   f"₹{best:,.2f}",
            "worst_trade":  f"₹{worst:,.2f}",
        }


# ── Paper trader ──────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Main loop: gets market data → runs strategy → asks LLM → places paper trade.
    
    In live mode, this runs on a schedule (every 5 or 15 minutes for intraday,
    or daily for positional trades).
    """
    
    def __init__(
        self,
        capital: float         = 500_000,
        risk_pct: float        = 1.0,
        llm_mode: str          = "claude",    # "claude" or "ollama"
        llm_api_key: str       = None,
        db_path: str           = "./data/paper_trades.db",
    ):
        self.capital   = capital
        self.llm_mode  = llm_mode
        self.api_key   = llm_api_key
        self.db        = TradeDB(db_path)
        self.risk      = RiskManager(capital=capital, max_risk_pct=risk_pct)
        self.open_trades: list[Trade] = []
        self._reload_open_trades()

    def _reload_open_trades(self):
        """Restore open trades from DB (survives restarts)."""
        rows = self.db.get_open_trades()
        self.open_trades = []
        for row in rows:
            t = Trade(**{k: row[k] for k in Trade.__dataclass_fields__})
            self.open_trades.append(t)
        if self.open_trades:
            print(f"  Restored {len(self.open_trades)} open trade(s) from DB")

    def run_once(self, df: pd.DataFrame, strategy_signal: str, symbol: str = "NIFTY50"):
        """
        Process one bar of data. Call this on every new candle.
        
        Args:
            df:               DataFrame with OHLCV + features (latest bar = last row)
            strategy_signal:  "BUY", "SELL", or "HOLD" from your strategy
            symbol:           instrument name
        """
        current_price = df["close"].iloc[-1]
        timestamp     = df.index[-1]
        
        print(f"\n[{timestamp}]  {symbol}  price={current_price:.2f}  "
              f"strategy={strategy_signal}")

        # ── Check open trade exits ────────────────────────────────
        self._check_exits(current_price)

        # ── Get LLM signal ────────────────────────────────────────
        if strategy_signal in ("BUY", "SELL") or self.open_trades:
            print(f"  Requesting LLM signal ({self.llm_mode})...")
            try:
                from llm.signal_generator import generate_signal
                llm_signal = generate_signal(
                    df,
                    mode    = self.llm_mode,
                    api_key = self.api_key,
                )
                print(llm_signal)
            except Exception as e:
                print(f"  LLM error: {e}. Skipping LLM confirmation.")
                return

        else:
            print("  Strategy: HOLD — skipping LLM call")
            return

        # ── Evaluate new entry ────────────────────────────────────
        if strategy_signal == "BUY" and not self.open_trades:
            approved, reason = self.risk.approve_trade(
                strategy_signal, llm_signal,
                len(self.open_trades), current_price
            )
            
            if approved:
                self._enter_trade(
                    symbol, current_price, llm_signal, 
                    strategy=f"{strategy_signal}_strategy"
                )
            else:
                print(f"  Trade REJECTED: {reason}")

    def _enter_trade(self, symbol: str, price: float, llm_signal, strategy: str):
        stop   = llm_signal.suggested_stop
        target = llm_signal.suggested_target
        qty    = self.risk.position_size(price, stop)
        
        if qty == 0:
            print("  Position size = 0. Skipping.")
            return
        
        trade = Trade(
            id=0, symbol=symbol, direction="LONG",
            entry_price=price, entry_time=datetime.now().isoformat(),
            stop_loss=stop, target=target, quantity=qty,
            strategy=strategy,
            llm_action=llm_signal.action,
            llm_confidence=llm_signal.confidence,
        )
        trade.id = self.db.save_trade(trade)
        self.open_trades.append(trade)
        
        risk_amount = qty * abs(price - stop)
        print(f"\n  ✅ PAPER TRADE ENTERED")
        print(f"     {symbol}  LONG  {qty} units @ ₹{price:.2f}")
        print(f"     Stop:   ₹{stop:.2f}  |  Target: ₹{target:.2f}")
        print(f"     Risk:   ₹{risk_amount:,.0f}  ({risk_amount/self.capital*100:.2f}% of capital)")

    def _check_exits(self, current_price: float):
        """Check each open trade for stop/target hits."""
        for trade in self.open_trades[:]:  # copy to allow removal
            hit_stop   = current_price <= trade.stop_loss
            hit_target = current_price >= trade.target
            
            if hit_stop or hit_target:
                reason = "STOP" if hit_stop else "TARGET"
                pnl    = (current_price - trade.entry_price) * trade.quantity
                
                self.db.close_trade(trade.id, current_price, reason, pnl)
                self.open_trades.remove(trade)
                
                emoji = "❌" if hit_stop else "🎯"
                print(f"\n  {emoji} TRADE CLOSED ({reason})")
                print(f"     Exit: ₹{current_price:.2f}  |  PnL: ₹{pnl:+,.2f}")

    def print_summary(self):
        """Print overall paper trading performance."""
        summary = self.db.performance_summary()
        print("\n" + "="*50)
        print("  PAPER TRADING PERFORMANCE SUMMARY")
        print("="*50)
        for k, v in summary.items():
            print(f"  {k:<20} {v}")
        print("="*50)


# ── Scheduled live feed loop ──────────────────────────────────────────────────

def start_live_paper_trading(
    kite,                         # KiteConnect instance
    trader: PaperTrader,
    strategy_class,               # your strategy class
    instrument_token: int,
    interval: str  = "15min",
    run_hours: int = 6,           # stop after N hours
):
    """
    Live paper trading loop. Fetches new data every `interval` and processes.
    
    This is the bridge between backtesting and live trading.
    Run during market hours (9:15 AM – 3:30 PM IST).
    
    Args:
        kite:             authenticated KiteConnect instance
        trader:           PaperTrader instance
        strategy_class:   MomentumStrategy or MeanReversionStrategy
        instrument_token: Nifty token (256265 for spot)
        interval:         "15min" or "1hr" for intraday, "1day" for positional
        run_hours:        auto-stop after this many hours
    """
    import schedule
    from data.kite_data import add_features

    INTERVAL_MINUTES = {"15min": 15, "1hr": 60, "1day": 1440}
    sleep_mins = INTERVAL_MINUTES.get(interval, 15)

    print(f"\nStarting live paper trading — {interval} bars")
    print(f"Auto-stop after {run_hours} hours")
    print("Press Ctrl+C to stop manually\n")

    start_time = datetime.now()

    def tick():
        if (datetime.now() - start_time).seconds > run_hours * 3600:
            print("Run time limit reached. Stopping.")
            return schedule.CancelJob

        from datetime import date, timedelta
        from_dt = datetime.now() - timedelta(days=30)
        to_dt   = datetime.now()

        try:
            records = kite.historical_data(
                instrument_token, from_dt, to_dt,
                interval=interval, continuous=False
            )
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df.columns = [c.lower() for c in df.columns]
            df = add_features(df)

            # Simple signal from strategy (just EMA crossover for demo)
            ema9  = df["ema9"].iloc[-1]
            ema21 = df["ema21"].iloc[-1]
            prev_ema9  = df["ema9"].iloc[-2]
            prev_ema21 = df["ema21"].iloc[-2]

            if ema9 > ema21 and prev_ema9 <= prev_ema21:
                strategy_signal = "BUY"
            elif ema9 < ema21 and prev_ema9 >= prev_ema21:
                strategy_signal = "SELL"
            else:
                strategy_signal = "HOLD"

            trader.run_once(df, strategy_signal)

        except Exception as e:
            print(f"Tick error: {e}")

    schedule.every(sleep_mins).minutes.do(tick)
    tick()  # run immediately on start

    while True:
        schedule.run_pending()
        time.sleep(30)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.kite_data import make_synthetic_nifty, add_features

    print("Paper Trader Demo — simulating 50 bars on synthetic data\n")

    df_full = make_synthetic_nifty(periods=300)
    df_full = add_features(df_full)

    # Simulate without real LLM by monkey-patching signal generator
    class MockSignal:
        action="BUY"; confidence=0.75; risk_level="MEDIUM"
        key_factors=["EMA bullish","RSI normal","Volume ok"]
        reasoning="Mocked signal for demo"; suggested_stop=0; suggested_target=0

    trader = PaperTrader(capital=500_000, llm_mode="mock")

    print("Simulating trades (no real LLM — mocked signals)...")

    for i in range(250, 300):
        df_slice = df_full.iloc[:i+1]
        price    = df_slice["close"].iloc[-1]
        ema9     = df_slice["ema9"].iloc[-1]
        ema21    = df_slice["ema21"].iloc[-1]
        prev9    = df_slice["ema9"].iloc[-2]
        prev21   = df_slice["ema21"].iloc[-2]
        
        if ema9 > ema21 and prev9 <= prev21:
            sig = "BUY"
        elif ema9 < ema21 and prev9 >= prev21:
            sig = "SELL"
        else:
            sig = "HOLD"
        
        # Check exits manually (no LLM calls in demo)
        trader._check_exits(price)

    trader.print_summary()
    print("\nNext steps:")
    print("  1. Connect Kite API and download real data")
    print("  2. Add your API key and switch llm_mode='claude'")
    print("  3. Run start_live_paper_trading() during market hours")
    print("  4. Watch for 30+ trades before drawing conclusions")
