"""
auto_trader.py — Fully automated OEH trade execution via Kite Connect.

Activated by setting AUTO_TRADE=true in .env.
Set AUTO_TRADE=false (default) for alert-only mode.

Exit logic (trailing stop):
  • Hard stop  : -50% from entry
  • Trail trigger: once +30% reached → lock floor at +10%
  • Trail:       floor rises to 85% of peak price as price climbs
  • Target:      +55% from entry → full exit

Usage:
    from auto_trader import AutoTrader
    trader = AutoTrader(kite, dry_run=not settings.AUTO_TRADE,
                        max_lots=settings.MAX_LOTS_AUTO)
    # On OEH signal:
    trader.on_entry(symbol, tsymbol, exchange, token, entry_ltp, lot_qty)
    # On every option tick:
    trader.on_tick(token, ltp)
    # At EOD:
    trader.force_close_all()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

log = logging.getLogger("auto_trader")
IST = ZoneInfo("Asia/Kolkata")

# ── Exit thresholds ────────────────────────────────────────────────────────────
HARD_STOP_PCT    = 50.0   # -50% → exit immediately
TRAIL_TRIGGER    = 30.0   # +30% → activate trailing stop
TRAIL_FLOOR_PCT  = 10.0   # once trailing activates, floor rises to entry + 10%
TRAIL_BELOW_PEAK = 15.0   # trail stop sits 15% below rolling peak
TARGET_PCT       = 55.0   # +55% → take profit


@dataclass
class Position:
    symbol:    str
    tsymbol:   str          # e.g. "NIFTY26MAY24000CE"
    exchange:  str          # "NFO" or "BFO"
    token:     int
    entry_px:  float
    qty:       int          # total shares (lots × lot_size)
    order_id:  str = ""     # Kite order ID of the entry order

    stop_px:          float = field(init=False)
    target_px:        float = field(init=False)
    peak_px:          float = field(init=False)
    trailing_active:  bool  = field(init=False, default=False)
    closed:           bool  = field(init=False, default=False)
    exit_reason:      str   = field(init=False, default="")
    exit_px:          float = field(init=False, default=0.0)
    last_ltp:         float = field(init=False, default=0.0)

    def __post_init__(self):
        self.stop_px   = round(self.entry_px * (1 - HARD_STOP_PCT   / 100), 2)
        self.target_px = round(self.entry_px * (1 + TARGET_PCT       / 100), 2)
        self.peak_px   = self.entry_px

    def check(self, ltp: float) -> str | None:
        """
        Feed latest LTP. Returns 'stop', 'trail_stop', or 'target' if exit
        should be triggered, else None.
        """
        if self.closed:
            return None

        self.peak_px = max(self.peak_px, ltp)

        # Hard stop
        if ltp <= self.stop_px:
            self.closed     = True
            self.exit_reason = "stop" if not self.trailing_active else "trail_stop"
            self.exit_px     = ltp
            return self.exit_reason

        # Activate trailing once +TRAIL_TRIGGER%
        if not self.trailing_active and ltp >= self.entry_px * (1 + TRAIL_TRIGGER / 100):
            self.trailing_active = True
            floor = self.entry_px * (1 + TRAIL_FLOOR_PCT / 100)
            self.stop_px = max(self.stop_px, round(floor, 2))
            log.info(f"{self.tsymbol}: trailing activated — floor ₹{self.stop_px:.2f}")

        # Move trail stop up with peak
        if self.trailing_active:
            trail = round(self.peak_px * (1 - TRAIL_BELOW_PEAK / 100), 2)
            if trail > self.stop_px:
                self.stop_px = trail

        # Target
        if ltp >= self.target_px:
            self.closed      = True
            self.exit_reason = "target"
            self.exit_px     = ltp
            return "target"

        return None

    def pnl(self) -> float:
        if self.exit_px:
            return round((self.exit_px - self.entry_px) * self.qty, 2)
        return 0.0

    def pnl_pct(self) -> float:
        if self.entry_px:
            return round((self.exit_px - self.entry_px) / self.entry_px * 100, 2)
        return 0.0


class AutoTrader:
    """
    Places and manages OEH trades via Kite Connect.

    dry_run=True  → log all actions but place no real orders (default).
    dry_run=False → live execution (set AUTO_TRADE=true in .env).
    """

    def __init__(self, kite, send_alert_fn, dry_run: bool = True,
                 max_lots: int = 1, daily_loss_cap: float = 3000.0):
        self.kite            = kite
        self.send_alert      = send_alert_fn
        self.dry_run         = dry_run
        self.max_lots        = max_lots
        self.daily_loss_cap  = daily_loss_cap   # stop taking new trades if loss hits this
        self._positions: dict[str, Position] = {}   # symbol → Position
        self._realized_pnl   = 0.0              # cumulative P&L from closed auto-trades
        self._cap_hit        = False            # True once daily loss cap is breached
        self._lock           = threading.Lock()

        mode = "DRY RUN" if dry_run else "⚡ LIVE"
        log.info(f"AutoTrader initialised — mode={mode}  max_lots={max_lots}  "
                 f"daily_loss_cap=₹{daily_loss_cap:,.0f}")

    # ── Entry ──────────────────────────────────────────────────────────────────

    def on_entry(self, symbol: str, tsymbol: str, exchange: str,
                 token: int, entry_ltp: float, lot_qty: int,
                 bias_score: int = 3, direction: str = "CE") -> None:
        """
        Called when OEH (CE) or OEL (PE) reversal is confirmed. Places a BUY order.

        bias_score: bullish indicators for CE (0–3), bearish indicators for PE (0–3).
                    Trade skipped if fewer than 2 required indicators align.
        direction:  "CE" (call, bullish) or "PE" (put, bearish).
        """
        with self._lock:
            if self._cap_hit:
                log.info(f"{symbol}: daily loss cap reached — no new trades today")
                return

            if bias_score < 2:
                side = "bullish" if direction == "CE" else "bearish"
                msg = (
                    f"⚠️ AUTO-TRADE SKIPPED — {symbol} {direction}\n"
                    f"Bias only {bias_score}/3 {side} at entry time.\n"
                    f"Market conditions not aligned — skipping."
                )
                self.send_alert(msg)
                log.info(f"{symbol}: {direction} skipped — bias {bias_score}/3 < 2")
                return

            pos_key = f"{symbol}_{direction}"
            if pos_key in self._positions and not self._positions[pos_key].closed:
                log.info(f"{symbol} {direction}: already in position — skipping")
                return

            qty = self.max_lots * lot_qty
            order_id = self._place_order(
                exchange=exchange,
                tsymbol=tsymbol,
                txn="BUY",
                qty=qty,
            )

            pos = Position(
                symbol   = symbol,
                tsymbol  = tsymbol,
                exchange = exchange,
                token    = token,
                entry_px = entry_ltp,
                qty      = qty,
                order_id = order_id,
            )
            self._positions[pos_key] = pos

            side_emoji = "🤖" if direction == "CE" else "🔻🤖"
            tag = "[DRY RUN] " if self.dry_run else ""
            msg = (
                f"{side_emoji} {tag}AUTO ENTRY — {tsymbol} ({direction})\n"
                f"BUY {self.max_lots} lot(s) @ ₹{entry_ltp:.2f}\n"
                f"Target: ₹{pos.target_px:.2f} (+{TARGET_PCT:.0f}%)\n"
                f"Stop:   ₹{pos.stop_px:.2f}  (-{HARD_STOP_PCT:.0f}%)\n"
                f"Trailing activates at ₹{entry_ltp*(1+TRAIL_TRIGGER/100):.2f} (+{TRAIL_TRIGGER:.0f}%)"
            )
            self.send_alert(msg)
            log.info(f"AutoTrader: entered {tsymbol} ({direction}) qty={qty} @ ₹{entry_ltp:.2f}")

    # ── Tick monitoring ────────────────────────────────────────────────────────

    def on_tick(self, token: int, ltp: float) -> None:
        """Feed every option tick. Checks and executes exits if thresholds are hit."""
        with self._lock:
            for pos in list(self._positions.values()):
                if pos.closed or pos.token != token:
                    continue

                pos.last_ltp = ltp
                reason = pos.check(ltp)
                if reason is None:
                    continue

                # Exit triggered
                self._place_order(
                    exchange=pos.exchange,
                    tsymbol=pos.tsymbol,
                    txn="SELL",
                    qty=pos.qty,
                )
                self._realized_pnl += pos.pnl()
                self._send_exit_alert(pos, ltp, reason)
                log.info(
                    f"AutoTrader: exited {pos.tsymbol} @ ₹{ltp:.2f} "
                    f"reason={reason} pnl=₹{pos.pnl():+,.0f} "
                    f"cumulative=₹{self._realized_pnl:+,.0f}"
                )

                # Check daily loss cap
                if self._realized_pnl <= -self.daily_loss_cap and not self._cap_hit:
                    self._cap_hit = True
                    msg = (
                        f"🛑 DAILY LOSS CAP HIT — ₹{self.daily_loss_cap:,.0f}\n"
                        f"Auto-trader is PAUSED for today.\n"
                        f"Total auto P&L: ₹{self._realized_pnl:+,.0f}\n"
                        f"No more trades will be taken today."
                    )
                    self.send_alert(msg)
                    log.warning("Daily loss cap reached — auto-trader paused")

    # ── Manual position watcher ───────────────────────────────────────────────

    def watch_position(self, symbol: str, tsymbol: str, exchange: str,
                       token: int, entry_px: float, qty: int,
                       direction: str = "CE") -> bool:
        """
        Register a manually-entered position for WebSocket monitoring.
        Uses the exact qty from Kite — bypasses lot calculation and bias check.
        Returns False if already tracked.
        """
        with self._lock:
            pos_key = f"{symbol}_{direction}"
            if pos_key in self._positions and not self._positions[pos_key].closed:
                return False   # already watching

            pos = Position(
                symbol   = symbol,
                tsymbol  = tsymbol,
                exchange = exchange,
                token    = token,
                entry_px = entry_px,
                qty      = qty,
                order_id = "MANUAL",
            )
            self._positions[pos_key] = pos

            tag = "[DRY RUN] " if self.dry_run else ""
            msg = (
                f"🔍 {tag}POSITION GUARDIAN — {tsymbol}\n"
                f"Manual position detected & now monitored\n"
                f"Entry ₹{entry_px:.2f} | Qty {qty}\n"
                f"Target ₹{pos.target_px:.2f} (+{TARGET_PCT:.0f}%) | "
                f"Stop ₹{pos.stop_px:.2f} (-{HARD_STOP_PCT:.0f}%)\n"
                f"Trailing activates at ₹{entry_px*(1+TRAIL_TRIGGER/100):.2f}"
            )
            self.send_alert(msg)
            log.info(f"Position guardian: watching {tsymbol} qty={qty} @ ₹{entry_px:.2f}")
            return True

    # ── EOD force-close ────────────────────────────────────────────────────────

    def force_close_all(self) -> None:
        """Force-close all open positions at market (called at 3:00 PM IST)."""
        with self._lock:
            for pos in self._positions.values():
                if pos.closed:
                    continue
                self._place_order(
                    exchange=pos.exchange,
                    tsymbol=pos.tsymbol,
                    txn="SELL",
                    qty=pos.qty,
                )
                pos.closed      = True
                pos.exit_reason = "force_close"
                tag = "[DRY RUN] " if self.dry_run else ""
                msg = (
                    f"🔔 {tag}FORCE CLOSE — {pos.tsymbol}\n"
                    f"Market close approaching — exiting position.\n"
                    f"Entry ₹{pos.entry_px:.2f} | Exit ~market"
                )
                self.send_alert(msg)

    # ── Position summary (Telegram broadcast) ─────────────────────────────────

    @staticmethod
    def _risk_badge(pos: "Position", indicators: dict) -> str:
        """
        Return a one-line risk assessment for this position based on current
        SuperTrend direction and spot vs EMA-50.

        ✅ Aligned   — ST and EMA both agree with the trade direction
        ⚠️ Caution   — One signal against the trade
        ❌ Exit zone — ST has flipped against the trade (consider closing)
        """
        sym      = "NIFTY50" if "NIFTY" in pos.tsymbol else "SENSEX"
        ind      = indicators.get(sym, {})
        st5      = ind.get("st5",     0)
        ema50    = ind.get("ema50",   0)
        last_cmp = pos.last_ltp if pos.last_ltp else pos.entry_px

        is_ce = "CE" in pos.tsymbol

        # SuperTrend alignment
        st_ok = (st5 == 1) if is_ce else (st5 == -1)

        # EMA-50 alignment (price above EMA for CE, below for PE)
        ema_ok = True
        if ema50 > 0:
            ema_ok = (last_cmp > ema50) if is_ce else (last_cmp < ema50)

        if st_ok and ema_ok:
            return "✅ Trend aligned — hold"
        elif st_ok and not ema_ok:
            return "⚠️ Caution — below EMA-50, monitor closely"
        elif not st_ok and ema_ok:
            return "⚠️ Caution — ST flipped, EMA still ok"
        else:
            return "❌ Against trend — ST + EMA both disagree, consider exiting"

    def position_summary(self, indicators: dict | None = None) -> str | None:
        """
        Build a Telegram-ready summary of all open positions.
        Returns None if no open positions exist.
        Each row shows: symbol, CMP, entry, P&L %, stop, target, trail status,
        and a risk badge based on current trend alignment.
        """
        with self._lock:
            open_pos = [p for p in self._positions.values() if not p.closed]

        if not open_pos:
            return None

        ind     = indicators or {}
        now_str = datetime.now(IST).strftime("%H:%M:%S")
        lines   = [f"📊 OPEN POSITIONS — {now_str}"]

        total_pnl = 0.0
        for pos in open_pos:
            cmp  = pos.last_ltp if pos.last_ltp else pos.entry_px
            pnl  = round((cmp - pos.entry_px) * pos.qty, 0)
            pct  = round((cmp - pos.entry_px) / pos.entry_px * 100, 1)
            total_pnl += pnl

            trail_str = (
                f"Trail SL ₹{pos.stop_px:.0f}"
                if pos.trailing_active
                else f"Hard SL ₹{pos.stop_px:.0f}"
            )
            dir_tag  = "🟢CE" if "CE" in pos.tsymbol else "🔴PE"
            risk     = self._risk_badge(pos, ind)

            lines.append(
                f"\n{dir_tag} {pos.tsymbol}\n"
                f"  CMP ₹{cmp:.2f}  Entry ₹{pos.entry_px:.2f}  "
                f"P&L ₹{pnl:+.0f} ({pct:+.1f}%)\n"
                f"  Tgt ₹{pos.target_px:.0f}  {trail_str}\n"
                f"  {risk}"
            )

        lines.append(f"\nNet P&L: ₹{total_pnl:+.0f}")
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _place_order(self, exchange: str, tsymbol: str,
                     txn: str, qty: int) -> str:
        """Place a market MIS order. Returns order ID (or 'DRY-RUN')."""
        if self.dry_run:
            log.info(f"[DRY RUN] Would place: {txn} {qty}x {tsymbol} @ MARKET")
            return "DRY-RUN"

        try:
            from kiteconnect import KiteConnect
            order_id = self.kite.place_order(
                variety          = self.kite.VARIETY_REGULAR,
                exchange         = exchange,
                tradingsymbol    = tsymbol,
                transaction_type = (self.kite.TRANSACTION_TYPE_BUY
                                    if txn == "BUY"
                                    else self.kite.TRANSACTION_TYPE_SELL),
                quantity         = qty,
                order_type       = self.kite.ORDER_TYPE_MARKET,
                product          = self.kite.PRODUCT_MIS,
            )
            log.info(f"Order placed: {txn} {qty}x {tsymbol} → order_id={order_id}")
            return str(order_id)
        except Exception as e:
            log.error(f"Order placement failed: {e}")
            self.send_alert(f"⚠️ ORDER FAILED — {txn} {tsymbol}: {e}")
            return "ERROR"

    def _send_exit_alert(self, pos: Position, exit_ltp: float, reason: str) -> None:
        emoji = {"target": "✅", "stop": "🛑", "trail_stop": "🔒", "force_close": "🔔"}
        label = {"target": "TARGET HIT", "stop": "STOP HIT",
                 "trail_stop": "TRAILING STOP", "force_close": "FORCE CLOSE"}
        tag   = "[DRY RUN] " if self.dry_run else ""
        pnl   = pos.pnl()
        pnl_pct = pos.pnl_pct()
        msg = (
            f"{emoji.get(reason,'🔔')} {tag}AUTO EXIT — {pos.tsymbol}\n"
            f"{label.get(reason, reason)}\n"
            f"Entry ₹{pos.entry_px:.2f} → Exit ₹{exit_ltp:.2f} "
            f"({pnl_pct:+.1f}%)\n"
            f"P&L: ₹{pnl:+,.0f}  ({pos.qty // (65 if 'NIFTY' in pos.tsymbol else 10)} lot(s))"
        )
        self.send_alert(msg)
