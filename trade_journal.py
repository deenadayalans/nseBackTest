"""
trade_journal.py
────────────────
Logs every OEH signal and its outcome to a CSV file.
The system auto-tracks the ATM CE (for OEH) strike — you don't have to.
You can override with your own strike in the summary, but the journal always
records the system's ATM-based entry so you can compare performance.

CSV schema (one row per signal):
  date, time, symbol, direction, oeh_spot, atm_strike, option_type,
  entry_px, target_px, stop_px,
  exit_px, exit_time, exit_reason,    ← filled when outcome known
  pnl_pct, pnl_rs, lots, status

Usage from live_alerter:
    from trade_journal import TradeJournal
    journal = TradeJournal()
    sid = journal.log_signal(...)
    journal.log_outcome(sid, exit_px, exit_time, exit_reason)
"""

from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

JOURNAL_DIR  = Path("journal")
JOURNAL_DIR.mkdir(exist_ok=True)

# Fixed lots used for P&L estimate (same as backtest)
_LOT_QTY = {"NIFTY50": 65, "SENSEX": 5}


class TradeJournal:
    """Thread-safe CSV trade journal."""

    COLUMNS = [
        "signal_id", "date", "signal_time", "symbol", "direction",
        "oeh_spot", "atm_strike", "option_type",
        "entry_px", "target_px", "stop_px",
        "exit_px", "exit_time", "exit_reason",
        "pnl_pct", "pnl_rs", "lots",
        "status",          # OPEN / HIT_TARGET / HIT_STOP / EXPIRED / MANUAL
    ]

    def __init__(self, journal_dir: Path = JOURNAL_DIR):
        self._dir  = journal_dir
        self._lock = threading.Lock()
        self._open: dict[str, dict] = {}   # signal_id → row dict (for open trades)

    def _csv_path(self, trade_date: date) -> Path:
        return self._dir / f"{trade_date.strftime('%Y-%m-%d')}.csv"

    def _next_id(self, trade_date: date) -> str:
        """Generate an incremental signal ID for the day."""
        path = self._csv_path(trade_date)
        if not path.exists():
            return f"{trade_date.strftime('%Y%m%d')}-01"
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        n = len(rows) + 1
        return f"{trade_date.strftime('%Y%m%d')}-{n:02d}"

    def _write_row(self, trade_date: date, row: dict):
        path = self._csv_path(trade_date)
        is_new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if is_new:
                w.writeheader()
            w.writerow({c: row.get(c, "") for c in self.COLUMNS})

    def _update_row(self, trade_date: date, signal_id: str, updates: dict):
        """Rewrite a specific row in the CSV with updated fields."""
        path = self._csv_path(trade_date)
        if not path.exists():
            return
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["signal_id"] == signal_id:
                r.update(updates)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            w.writeheader()
            w.writerows(rows)

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_signal(
        self,
        symbol: str,
        direction: str,          # "CE" or "PE"
        oeh_spot: float,
        atm_strike: int,
        entry_px: float,         # ATM option price at entry
        target_px: float,
        stop_px: float,
        signal_time: Optional[datetime] = None,
        lots: int = 1,
    ) -> str:
        """Record a new OEH signal. Returns the signal_id."""
        now       = signal_time or datetime.now()
        trade_date = now.date()
        with self._lock:
            sid = self._next_id(trade_date)
            row = {
                "signal_id":   sid,
                "date":        trade_date.isoformat(),
                "signal_time": now.strftime("%H:%M:%S"),
                "symbol":      symbol,
                "direction":   direction,
                "oeh_spot":    round(oeh_spot, 2),
                "atm_strike":  atm_strike,
                "option_type": direction,
                "entry_px":    round(entry_px, 2),
                "target_px":   round(target_px, 2),
                "stop_px":     round(stop_px, 2),
                "exit_px":     "",
                "exit_time":   "",
                "exit_reason": "",
                "pnl_pct":     "",
                "pnl_rs":      "",
                "lots":        lots,
                "status":      "OPEN",
            }
            self._write_row(trade_date, row)
            self._open[sid] = row.copy()
        return sid

    def log_outcome(
        self,
        signal_id: str,
        exit_px: float,
        exit_time: Optional[datetime] = None,
        exit_reason: str = "MANUAL",    # HIT_TARGET | HIT_STOP | EXPIRED | MANUAL
    ):
        """Update a signal row with exit data and compute P&L."""
        with self._lock:
            row = self._open.get(signal_id)
            if row is None:
                return   # already closed or unknown

            now   = exit_time or datetime.now()
            entry = float(row["entry_px"])
            qty   = int(row["lots"]) * _LOT_QTY.get(row["symbol"], 1)

            pnl_pct = (exit_px - entry) / entry * 100 if entry else 0
            pnl_rs  = (exit_px - entry) * qty

            updates = {
                "exit_px":     round(exit_px, 2),
                "exit_time":   now.strftime("%H:%M:%S"),
                "exit_reason": exit_reason,
                "pnl_pct":     round(pnl_pct, 2),
                "pnl_rs":      round(pnl_rs, 2),
                "status":      exit_reason if exit_reason in ("HIT_TARGET", "HIT_STOP", "EXPIRED") else "CLOSED",
            }
            trade_date = datetime.fromisoformat(row["date"]).date()
            self._update_row(trade_date, signal_id, updates)
            self._open.pop(signal_id, None)

    def close_open_trades(self, ltp_map: dict[str, float], reason: str = "EXPIRED"):
        """
        Called at 3:30 PM. For any signals still OPEN, mark them closed
        using the last known option LTP from ltp_map (keyed by symbol).
        """
        with self._lock:
            for sid, row in list(self._open.items()):
                symbol = row["symbol"]
                ltp    = ltp_map.get(symbol)
                if ltp is None:
                    ltp = float(row["entry_px"])   # fallback: flat P&L
                now    = datetime.now()
                entry  = float(row["entry_px"])
                qty    = int(row["lots"]) * _LOT_QTY.get(symbol, 1)
                pnl_pct = (ltp - entry) / entry * 100 if entry else 0
                pnl_rs  = (ltp - entry) * qty
                updates = {
                    "exit_px":     round(ltp, 2),
                    "exit_time":   now.strftime("%H:%M:%S"),
                    "exit_reason": reason,
                    "pnl_pct":     round(pnl_pct, 2),
                    "pnl_rs":      round(pnl_rs, 2),
                    "status":      reason,
                }
                trade_date = datetime.fromisoformat(row["date"]).date()
                self._update_row(trade_date, sid, updates)
            self._open.clear()

    # ── Reporting ──────────────────────────────────────────────────────────────

    def daily_summary(self, trade_date: Optional[date] = None) -> str:
        """Return a formatted summary string for the day."""
        trade_date = trade_date or date.today()
        path = self._csv_path(trade_date)
        if not path.exists():
            return f"📋 No trades recorded on {trade_date}"

        df = pd.read_csv(path)
        total    = len(df)
        closed   = df[df["status"] != "OPEN"]
        wins     = closed[closed["pnl_pct"].astype(float) > 0]
        losses   = closed[closed["pnl_pct"].astype(float) <= 0]
        open_cnt = len(df[df["status"] == "OPEN"])

        total_pnl_rs = closed["pnl_rs"].astype(float).sum() if len(closed) else 0
        avg_pnl_pct  = closed["pnl_pct"].astype(float).mean() if len(closed) else 0
        win_rate     = len(wins) / len(closed) * 100 if len(closed) else 0

        lines = [
            f"📊 *Trade Journal — {trade_date.strftime('%d %b %Y')}*\n",
            f"Total signals: {total}  |  Closed: {len(closed)}  |  Open: {open_cnt}",
            f"Win rate: {win_rate:.0f}%  ({len(wins)}W / {len(losses)}L)",
            f"Avg P&L: {avg_pnl_pct:+.1f}%  |  Net P&L: ₹{total_pnl_rs:+,.0f}\n",
        ]

        for _, r in df.iterrows():
            pnl = f"{float(r['pnl_pct']):+.1f}%" if r["pnl_pct"] != "" else "—"
            pnl_rs = f"₹{float(r['pnl_rs']):+,.0f}" if r["pnl_rs"] != "" else "—"
            em = "✅" if r["status"] == "HIT_TARGET" else (
                 "❌" if r["status"] == "HIT_STOP"   else (
                 "⏳" if r["status"] == "OPEN"        else "🔲"))
            lines.append(
                f"{em} {r['signal_time']} {r['symbol']} {r['atm_strike']}{r['direction']}  "
                f"Entry:{r['entry_px']}  Exit:{r['exit_px'] or '—'}  "
                f"{pnl} ({pnl_rs})  [{r['status']}]"
            )

        return "\n".join(lines)

    def monthly_summary(self) -> str:
        """Summarise all CSV files in the journal directory."""
        files = sorted(self._dir.glob("*.csv"))
        if not files:
            return "📋 No journal data yet."

        rows = []
        for f in files:
            try:
                rows.append(pd.read_csv(f))
            except Exception:
                pass
        if not rows:
            return "📋 No data."

        df = pd.concat(rows, ignore_index=True)
        closed = df[df["status"] != "OPEN"].copy()
        closed["pnl_rs"]  = closed["pnl_rs"].astype(float)
        closed["pnl_pct"] = closed["pnl_pct"].astype(float)

        total    = len(df)
        wins     = len(closed[closed["pnl_pct"] > 0])
        losses   = len(closed[closed["pnl_pct"] <= 0])
        net_rs   = closed["pnl_rs"].sum()
        avg_pct  = closed["pnl_pct"].mean() if len(closed) else 0
        win_rate = wins / len(closed) * 100 if len(closed) else 0
        best     = closed["pnl_rs"].max() if len(closed) else 0
        worst    = closed["pnl_rs"].min() if len(closed) else 0

        return (
            f"📈 *Journal Summary ({files[0].stem} → {files[-1].stem})*\n"
            f"Total signals: {total}  |  Closed: {len(closed)}\n"
            f"Win rate: {win_rate:.0f}%  ({wins}W / {losses}L)\n"
            f"Avg P&L: {avg_pct:+.1f}%\n"
            f"Net P&L: ₹{net_rs:+,.0f}\n"
            f"Best: ₹{best:+,.0f}  |  Worst: ₹{worst:+,.0f}"
        )
