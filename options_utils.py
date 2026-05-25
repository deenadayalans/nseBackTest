"""
options_utils.py — Black-Scholes pricing, strike helpers, historical volatility.

All option prices are in index points (same units as the underlying).
Multiply by lot size to get rupee P&L.
"""

import numpy as np
from scipy.stats import norm
import pandas as pd


# ── Instrument config ─────────────────────────────────────────────────────────

INSTRUMENTS = {
    # symbol: (lot_size, strike_interval, expiry_weekday, pre_expiry_weekday)
    # weekday: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
    "NIFTY50":  {"lot": 65,  "strike_step": 50,  "expiry_dow": 1, "pre_dow": 0},  # Tue expiry
    "SENSEX":   {"lot": 10,  "strike_step": 100, "expiry_dow": 3, "pre_dow": 2},  # Thu expiry
    "BANKNIFTY":{"lot": 30,  "strike_step": 100, "expiry_dow": 4, "pre_dow": 3},  # Fri expiry (placeholder)
}

RISK_FREE_RATE = 0.065   # India 10-yr gilt approximate
MARKET_OPEN    = "09:15"
MARKET_CLOSE   = "15:30"
TRADING_HOURS  = 6.25    # hours per day
TRADING_DAYS   = 252     # per year


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, sigma: float,
             r: float = RISK_FREE_RATE, option_type: str = "call") -> float:
    """
    Black-Scholes option price.

    S     : spot price
    K     : strike price
    T     : time to expiry in years (e.g. 0.5/252 for half a trading day)
    sigma : annualised volatility (e.g. 0.15 for 15%)
    r     : risk-free rate (annual)
    """
    if T <= 0 or sigma <= 0:
        # At expiry: intrinsic value only
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, sigma: float,
              r: float = RISK_FREE_RATE, option_type: str = "call") -> dict:
    """Return delta, gamma, theta, vega."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100   # per 1% vol move

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / TRADING_DAYS
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / TRADING_DAYS

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ── Strike helpers ────────────────────────────────────────────────────────────

def atm_strike(spot: float, step: int) -> int:
    """Round spot to nearest valid strike."""
    return round(spot / step) * step


def otm_call_strike(spot: float, step: int, pct: float) -> int:
    """OTM call strike: above spot by pct%."""
    target = spot * (1 + pct / 100)
    return int(np.ceil(target / step) * step)


def otm_put_strike(spot: float, step: int, pct: float) -> int:
    """OTM put strike: below spot by pct%."""
    target = spot * (1 - pct / 100)
    return int(np.floor(target / step) * step)


# ── Volatility ────────────────────────────────────────────────────────────────

def hist_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling historical volatility, annualised.
    Uses log returns. window bars of daily data → annualised by sqrt(252).
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def intraday_time_to_expiry(current_time_str: str,
                             expiry_close: str = "15:30") -> float:
    """
    Time remaining to expiry as fraction of a year.
    current_time_str: "HH:MM"  (on expiry day)
    Returns T in years (e.g. 6 hours remaining ≈ 6/(6.25*252))
    """
    def to_minutes(t: str) -> int:
        h, m = map(int, t.split(":"))
        return h * 60 + m

    remaining_min = max(0, to_minutes(expiry_close) - to_minutes(current_time_str))
    remaining_hours = remaining_min / 60
    return remaining_hours / (TRADING_HOURS * TRADING_DAYS)


def overnight_time_to_expiry(trading_days_remaining: float) -> float:
    """Convert remaining trading days to fraction of year."""
    return trading_days_remaining / TRADING_DAYS


# ── Straddle / strangle pricing ───────────────────────────────────────────────

def straddle_premium(S: float, K: float, T: float, sigma: float,
                     r: float = RISK_FREE_RATE) -> dict:
    """ATM straddle = call + put at same strike."""
    call = bs_price(S, K, T, sigma, r, "call")
    put  = bs_price(S, K, T, sigma, r, "put")
    return {
        "call": round(call, 2),
        "put":  round(put, 2),
        "total": round(call + put, 2),
        "breakeven_up":   round(K + call + put, 1),
        "breakeven_down": round(K - call - put, 1),
    }


def strangle_premium(S: float, call_K: float, put_K: float,
                     T: float, sigma: float,
                     r: float = RISK_FREE_RATE) -> dict:
    """OTM strangle = OTM call + OTM put at different strikes."""
    call = bs_price(S, call_K, T, sigma, r, "call")
    put  = bs_price(S, put_K,  T, sigma, r, "put")
    return {
        "call": round(call, 2),
        "put":  round(put, 2),
        "total": round(call + put, 2),
        "call_strike": call_K,
        "put_strike":  put_K,
        "breakeven_up":   round(call_K + call + put, 1),
        "breakeven_down": round(put_K  - call - put, 1),
    }
